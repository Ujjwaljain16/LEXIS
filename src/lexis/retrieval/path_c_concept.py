import json
import logging
from typing import List, Any
from litellm import acompletion
from lexis.indexing.qdrant_client import LexisQdrantClient
from lexis.ingestion.embedder import BGEM3Embedder
from lexis.config import settings
from lexis.generation.prompts import CONCEPT_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

class ConceptRoutingRetrieval:
    """
    Path C: Concept Routing.
    Extracts conceptual entities from the query using an LLM, and searches the proposition graph.
    """
    def __init__(self):
        self.qdrant = LexisQdrantClient()
        self.embedder = BGEM3Embedder()
        self.collection_name = "propositions_v2"

    async def retrieve(self, query_text: str, top_k: int) -> List[Any]:
        try:
            response = await acompletion(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": CONCEPT_EXTRACTION_PROMPT},
                    {"role": "user", "content": query_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(response.choices[0].message.content)
            concepts = data.get("concepts", [])
        except Exception as e:
            logger.error(f"Failed to extract concepts: {e}")
            concepts = [query_text]

        if not concepts:
            concepts = [query_text]
            
        concept_query = " ".join(concepts)
        # Embed the concepts
        concept_emb = self.embedder.embed_text(concept_query).tolist()
        return await self.qdrant.search(self.collection_name, concept_emb, top_k=top_k)
