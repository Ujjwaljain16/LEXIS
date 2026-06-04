"""
Elasticsearch Client for LEXIS (BM25 Retrieval).

Rationale: Provides BM25 lexical search capabilities required for Path D in the hybrid retriever.
Source Inspiration: Haystack BM25Retriever.
Deviations: Configured with standard analyzers initially (without complex legal synonyms) to establish a functional baseline.
Expected Impact: Improves recall for keyword-heavy queries where dense vectors fail (e.g., exact IDs or names).
"""
from elasticsearch import AsyncElasticsearch
from lexis.config import settings

class LexisElasticsearchClient:
    def __init__(self):
        self.client = AsyncElasticsearch(hosts=[settings.es_host])
        self.index_name = "lexis_primary"

    async def initialize_index(self):
        exists = await self.client.indices.exists(index=self.index_name)
        if not exists:
            mapping = {
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        }
                    }
                }
            }
            await self.client.indices.create(index=self.index_name, body=mapping)

    async def index_documents(self, documents: list[dict]):
        """Bulk index documents into ES."""
        from elasticsearch.helpers import async_bulk
        
        actions = [
            {
                "_index": self.index_name,
                "_id": doc["chunk_id"],
                "_source": doc
            }
            for doc in documents
        ]
        await async_bulk(self.client, actions)

    async def search(self, query_text: str, size: int = 5) -> list[dict]:
        """Search the primary index using BM25 lexical match."""
        body = {
            "query": {
                "match": {
                    "content": query_text
                }
            }
        }
        response = await self.client.search(index=self.index_name, body=body, size=size)
        return [hit["_source"] for hit in response["hits"]["hits"]]
