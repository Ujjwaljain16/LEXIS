from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from lexis.config import settings
from typing import List

class LexisQdrantClient:
    def __init__(self):
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=settings.qdrant_timeout_s,
        )

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

    async def find_by_payload(self, collection_name: str, must_match: dict, limit: int = 1) -> list[models.Record]:
        """
        Retrieve points by exact-match payload fields (e.g. doc_id + chunk_index).
        Point IDs in Qdrant are content-derived hashes (see Chunk.create) and cannot be
        constructed in advance, so lookups that only know a chunk's position must filter
        on payload instead of retrieving by id.
        """
        conditions = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in must_match.items()
        ]
        records, _ = await self.client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(must=conditions),
            limit=limit,
            with_payload=True,
        )
        return records

    async def search(self, collection_name: str, query_vector: list[float], top_k: int = 10) -> list[models.ScoredPoint]:
        # AsyncQdrantClient.search() was removed in qdrant-client 1.10+; the
        # query_points() replacement returns a QueryResponse wrapping .points
        # rather than the plain list search() used to return directly.
        response = await self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
        )
        return response.points
