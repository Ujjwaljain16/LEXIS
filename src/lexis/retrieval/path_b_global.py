from typing import List
from lexis.retrieval.interfaces import RetrievalPath, Query, Candidate
from lexis.indexing.qdrant_client import LexisQdrantClient
from lexis.ingestion.embedder import BGEM3Embedder
from lexis.config import settings

class GlobalDenseRetrieval(RetrievalPath):
    """
    Path B: Global Dense Retrieval.
    Uses BGE-M3 embeddings to search the primary Qdrant vector space.
    Highly effective for semantic matches that lack exact keyword overlap.
    """
    def __init__(self):
        self.qdrant = LexisQdrantClient()
        self.embedder = BGEM3Embedder()
        self.collection_name = settings.qdrant_collection_primary

    async def retrieve(self, query: Query) -> List[Candidate]:
        # Embed the query text
        query_vector = self.embedder.embed_text(query.text).tolist()
        
        # Search Qdrant
        points = await self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=query.top_k
        )
        
        # Map to common Candidate protocol
        candidates = []
        for point in points:
            metadata = point.payload or {}
            candidates.append(
                Candidate(
                    # point.id is Qdrant's internal storage identifier (see
                    # IngestionPipeline._deterministic_uuid), not the application's
                    # citation-bearing chunk id. The real chunk_id is only in the
                    # payload, written by the ingestion pipeline. Falling back to
                    # point.id only guards against payload lacking the field.
                    chunk_id=metadata.get("chunk_id") or str(point.id),
                    score=point.score,
                    source_path="path_b_global",
                    metadata=metadata,
                    content=metadata.get("content", "")
                )
            )
            
        return candidates
