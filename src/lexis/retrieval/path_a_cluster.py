from typing import List, Any
from lexis.indexing.qdrant_client import LexisQdrantClient
from lexis.config import settings

class ClusterHierarchicalRetrieval:
    """
    Path A: RAPTOR Cluster Search.
    Searches across hierarchical tree nodes created by RAPTOR.
    """
    def __init__(self):
        self.qdrant = LexisQdrantClient()
        self.collection_name = settings.qdrant_collection_clusters

    async def retrieve(self, query_emb: List[float], top_k: int) -> List[Any]:
        return await self.qdrant.search(self.collection_name, query_emb, top_k=top_k)
