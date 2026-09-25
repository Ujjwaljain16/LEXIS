"""
HyPE (Hypothetical Document Embeddings) Generator for LEXIS.

Rationale: Generates synthetic queries that a user might ask, which would be answered by the chunk.
Source Inspiration: plan.md (HyPE Path B).
Deviations from Source Repos: Generates settings.hype_questions_per_chunk questions per chunk.
Expected Impact on Metrics: Increases recall for asymmetrical queries (short query vs long document chunk).
"""
import asyncio
import json
import logging
from typing import List

import litellm
from litellm import acompletion

from lexis.config import settings
from lexis.ingestion.rate_limiter import AsyncRateLimiter

logger = logging.getLogger(__name__)


def _is_daily_quota_exhaustion(error: Exception) -> bool:
    """A per-MINUTE rate limit is worth a short backoff retry -- it resets in
    under a minute. A per-DAY quota exhaustion (confirmed live: some Gemini
    free-tier keys cap gemini-2.5-flash at as few as 20 requests/day, far
    below what HyPE needs for even one small document's chunks) will not
    resolve within this run no matter how long we wait or how many times we
    retry -- litellm's RateLimitError message embeds the raw API error body,
    which names the specific quota that was hit (quotaId contains "PerDay"
    for a daily cap, "PerMinute" for the transient one this class is
    actually designed to ride out)."""
    return "PerDay" in str(error)


class HyPEGenerator:
    def __init__(self):
        # Shared across every generate_questions() call this instance makes -- pipeline.py
        # constructs one HyPEGenerator per IngestionPipeline, so this paces every chunk's call
        # across a whole document (or ingestion run), not per-chunk in isolation. A concurrency
        # bound (see pipeline.py's semaphore) alone is not enough: fast calls can still exceed
        # 20/min even with few in flight at once, confirmed live against Gemini's free tier.
        self._rate_limiter = AsyncRateLimiter(max_per_minute=settings.hype_requests_per_minute)
        # Once a daily quota exhaustion is confirmed once, every other call sharing the same key
        # this run will fail identically -- there is no reason to spend a full
        # retry-with-backoff cycle (and a real API round trip) on each of potentially hundreds
        # of remaining chunks just to rediscover the same fact. Sticky for this instance's
        # lifetime (matches its lifetime: one per IngestionPipeline / ingestion run).
        self._daily_quota_exhausted = False

    async def generate_questions(self, chunk_text: str) -> List[str]:
        if self._daily_quota_exhausted:
            return []

        n = settings.hype_questions_per_chunk
        system_prompt = (
            f"You are an expert search engine query generator. Read the text and generate exactly {n} "
            "hypothetical questions that a user might search for which this text directly answers. "
            "Output ONLY valid JSON matching this schema: {'questions': ['string', ...]}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk_text}
        ]

        for attempt in range(settings.hype_max_retries + 1):
            await self._rate_limiter.acquire()
            try:
                # Timeout matters here specifically because pipeline.py fans this call out
                # concurrently across every chunk of a document via asyncio.gather -- without a
                # per-call timeout, one slow/stuck request blocks the ENTIRE document's ingestion
                # indefinitely (gather waits for all tasks), not just its own 3 questions.
                response = await asyncio.wait_for(
                    acompletion(
                        model=settings.gemini_model_feature,
                        messages=messages,
                        response_format={"type": "json_object"},
                        temperature=0.1
                    ),
                    timeout=settings.hype_generation_timeout_s,
                )
                data = json.loads(response.choices[0].message.content)
                return data.get("questions", [])
            except litellm.RateLimitError as e:
                if _is_daily_quota_exhaustion(e):
                    self._daily_quota_exhausted = True
                    logger.warning(
                        "HyPE question generation hit a DAILY quota limit (not per-minute) -- "
                        "this will not reset during this run, so no further HyPE calls will be "
                        f"attempted for the rest of this ingestion. Original error: {e}"
                    )
                    return []
                if attempt < settings.hype_max_retries:
                    backoff = settings.hype_retry_backoff_s * (attempt + 1)
                    logger.warning(f"HyPE question generation rate-limited (attempt {attempt + 1}/"
                                    f"{settings.hype_max_retries + 1}), retrying in {backoff:.1f}s: {e}")
                    await asyncio.sleep(backoff)
                    continue
                logger.warning(f"HyPE question generation rate-limited, giving up after "
                                f"{settings.hype_max_retries + 1} attempts: {e}")
                return []
            except Exception as e:
                # Fail open: a chunk whose HyPE generation fails still gets ingested normally
                # (dense/BM25 paths unaffected), it just gets no hypothetical-question candidates
                # -- matches feature_extractor.py's fail-open-with-a-flag philosophy. Logged (not
                # silently swallowed) because a previously broken model reference here
                # (settings.llm_model, which never existed) went undetected for exactly this
                # reason. Non-rate-limit errors are not retried -- they are not expected to
                # resolve by waiting.
                logger.warning(f"HyPE question generation failed: {e}")
                return []
        return []  # unreachable (the loop always returns), but keeps this an obvious List[str]
