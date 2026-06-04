"""
HyPE (Hypothetical Document Embeddings) Generator for LEXIS.

Rationale: Generates synthetic queries that a user might ask, which would be answered by the chunk.
Source Inspiration: plan.md (HyPE Path B).
Deviations from Source Repos: Generates 3 specific questions per chunk.
Expected Impact on Metrics: Increases recall for asymmetrical queries (short query vs long document chunk).
"""
from litellm import acompletion
import json
from typing import List
from lexis.config import settings

class HyPEGenerator:
    def __init__(self):
        pass

    async def generate_questions(self, chunk_text: str) -> List[str]:
        system_prompt = (
            "You are an expert search engine query generator. Read the text and generate exactly 3 "
            "hypothetical questions that a user might search for which this text directly answers. "
            "Output ONLY valid JSON matching this schema: {'questions': ['string', 'string', 'string']}"
        )
        
        try:
            response = await acompletion(
                model=settings.llm_model,
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
            return []
