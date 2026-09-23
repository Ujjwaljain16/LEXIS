"""
HyPE (Hypothetical Document Embeddings) Generator for LEXIS.

Rationale: Generates synthetic queries that a user might ask, which would be answered by the chunk.
Source Inspiration: plan.md (HyPE Path B).
Deviations from Source Repos: Generates settings.hype_questions_per_chunk questions per chunk.
Expected Impact on Metrics: Increases recall for asymmetrical queries (short query vs long document chunk).
"""
import json
import logging
from typing import List

from litellm import acompletion

from lexis.config import settings

logger = logging.getLogger(__name__)


class HyPEGenerator:
    def __init__(self):
        pass

    async def generate_questions(self, chunk_text: str) -> List[str]:
        n = settings.hype_questions_per_chunk
        system_prompt = (
            f"You are an expert search engine query generator. Read the text and generate exactly {n} "
            "hypothetical questions that a user might search for which this text directly answers. "
            "Output ONLY valid JSON matching this schema: {'questions': ['string', ...]}"
        )

        try:
            response = await acompletion(
                model=settings.gemini_model_feature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("questions", [])
        except Exception as e:
            # Fail open: a chunk whose HyPE generation fails still gets ingested normally
            # (dense/BM25 paths unaffected), it just gets no hypothetical-question candidates --
            # matches feature_extractor.py's fail-open-with-a-flag philosophy. Logged (not
            # silently swallowed) because a previously broken model reference here
            # (settings.llm_model, which never existed) went undetected for exactly this reason.
            logger.warning(f"HyPE question generation failed: {e}")
            return []
