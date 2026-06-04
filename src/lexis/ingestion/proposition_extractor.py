"""
Proposition Extractor for LEXIS (LegalGraphRAG Foundation).

Rationale: Extracts atomic Subject-Predicate-Object triplets from chunks.
Source Inspiration: LegalGraphRAG `judge_dep` logic and PaperQA structured outputs.
Deviations from Source Repos: Modified to use Groq/Gemini via native JSON schema extraction to maintain high throughput.
Expected Impact on Metrics: Deep Mode retrieval precision will increase by traversing the knowledge graph instead of relying purely on vector similarity.
"""
from typing import List
from lexis.indexing.schema import Proposition
from lexis.config import settings
from litellm import acompletion
import json

class PropositionExtractor:
    def __init__(self):
        pass
        
    async def extract(self, chunk_text: str, chunk_id: str, doc_id: str) -> List[Proposition]:
        """
        Uses Groq to extract Propositions via JSON schema constraint.
        """
        system_prompt = (
            "You are an expert legal entity extractor. Extract subject-predicate-object triplets "
            "from the following text. Output ONLY valid JSON matching this schema: "
            "{'propositions': [{'subject': 'string', 'predicate': 'string', 'object': 'string'}]} "
        )
        
        try:
            response = await acompletion(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            data = json.loads(response.choices[0].message.content)
            propositions = []
            for p in data.get("propositions", []):
                propositions.append(Proposition(
                    subject=p["subject"],
                    predicate=p["predicate"],
                    object=p["object"],
                    chunk_id=chunk_id,
                    doc_id=doc_id
                ))
            return propositions
        except Exception as e:
            # Fallback or empty if extraction fails
            return []
