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


class HyPEGenerator:
    def __init__(self):
        # Shared across every generate_questions() call this instance makes -- pipeline.py
        # constructs one HyPEGenerator per IngestionPipeline, so this paces every chunk's call
        # across a whole document (or ingestion run), not per-chunk in isolation. A concurrency
        # bound (see pipeline.py's semaphore) alone is not enough: fast calls can still exceed
        # 20/min even with few in flight at once, confirmed live against Gemini's free tier.
        self._rate_limiter = AsyncRateLimiter(max_per_minute=settings.hype_requests_per_minute)

    async def generate_questions(self, chunk_text: str) -> List[str]:
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
