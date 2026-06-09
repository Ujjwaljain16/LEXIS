"""
Qdrant Vector Database Client for LEXIS.

Rationale: Provides an abstraction layer over Qdrant to manage vector collections securely.
Source Inspiration: RAGFlow vector indexing and general Qdrant best practices.
Deviations: Uses 4 distinct collections (primary, hype, propositions, clusters) as per the LEXIS multi-path architecture.
Expected Impact: Guarantees correct payload indexing which is required for Fast/Deep retrieval modes.
"""
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from lexis.config import settings

class LexisQdrantClient:
    def __init__(self):
        self.client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.vector_size = 384  # Using BAAI/bge-small-en-v1.5

    async def initialize_collections(self):
        # We append _v2 to force Qdrant to create new collections with the updated 384 vector size
        collections = ["primary_v2", "hype_v2", "propositions_v2", "clusters_v2"]
        
        collections_res = await self.client.get_collections()
        existing = [c.name for c in collections_res.collections]
        
        for name in collections:
            if name not in existing:
                await self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                
                # Payload Indices for fast filtering
                await self.client.create_payload_index(
                    collection_name=name,
                    field_name="doc_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                
                if name == "clusters":
                    await self.client.create_payload_index(
                        collection_name=name,
                        field_name="level",
                        field_schema=models.PayloadSchemaType.INTEGER,
                    )

    async def upsert_chunks(self, collection_name: str, points: list[models.PointStruct]):
        """Upsert a list of points to a specific collection."""
        await self.client.upsert(
            collection_name=collection_name,
            points=points
        )

    async def search(self, collection_name: str, query_vector: list[float], top_k: int = 5) -> list[models.ScoredPoint]:
        """Search a specific collection using cosine similarity."""
        response = await self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return response
