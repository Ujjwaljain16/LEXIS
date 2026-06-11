from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from lexis.config import settings
from typing import List

class LexisQdrantClient:
    def __init__(self):
        self.client = AsyncQdrantClient(url=settings.qdrant_url)

    async def initialize_collections(self):
        collections = [
            settings.qdrant_collection_primary,
            settings.qdrant_collection_hype,
            settings.qdrant_collection_propositions,
            settings.qdrant_collection_clusters
        ]
        
        collections_res = await self.client.get_collections()
        existing = [c.name for c in collections_res.collections]
        
        for name in collections:
            if name not in existing:
                await self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=settings.embedding_dim,
                        distance=models.Distance.COSINE
                    )
                )
                
                # Payload Indices for fast filtering
                await self.client.create_payload_index(
                    collection_name=name,
                    field_name="doc_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                
                if name == settings.qdrant_collection_clusters:
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

    async def get_points(self, collection_name: str, ids: list[str]) -> list[models.Record]:
        """Retrieve points by IDs."""
        return await self.client.retrieve(
            collection_name=collection_name,
            ids=ids
        )

    async def search(self, collection_name: str, query_vector: list[float], top_k: int = 10) -> list[models.ScoredPoint]:
        return await self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
