from typing import List
from lexis.retrieval.interfaces import RetrievalPath, Query, Candidate
from lexis.indexing.qdrant_client import LexisQdrantClient
from lexis.ingestion.embedder import BGEM3Embedder

class GlobalDenseRetrieval(RetrievalPath):
    """
    Path B: Global Dense Retrieval.
    Uses BGE-M3 embeddings to search the primary Qdrant vector space.
    Highly effective for semantic matches that lack exact keyword overlap.
    """
    def __init__(self):
        self.qdrant = LexisQdrantClient()
        self.embedder = BGEM3Embedder()
        self.collection_name = "primary_v2"

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
                    chunk_id=str(point.id),
                    score=point.score,
                    source_path="path_b_global",
                    metadata=metadata,
                    content=metadata.get("content", "")
                )
            )
            
        return candidates
