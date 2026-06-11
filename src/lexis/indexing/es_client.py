import re
from typing import List, Dict, Any, Optional
from elasticsearch import AsyncElasticsearch
from lexis.config import settings

INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "legal_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "legal_synonyms", "stop"]
                }
            },
            "filter": {
                "legal_synonyms": {
                    "type": "synonym",
                    "synonyms": [
                        "indemnification, indemnity, hold harmless",
                        "termination, cancellation, rescission",
                        "breach, violation, default",
                        "obligation, duty, requirement",
                        "agreement, contract, covenant",
                        "revenue, sales, income",
                        "ebitda, earnings before interest taxes depreciation amortization",
                        "ipo, initial public offering",
                        "sec, securities and exchange commission",
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "source_file": {"type": "keyword"},
            "page_num": {"type": "integer"},
            "content": {
                "type": "text",
                "analyzer": "legal_analyzer",
                "similarity": "BM25",
            },
            "title": {
                "type": "text",
                "analyzer": "legal_analyzer",
                "boost": 10,
            },
            "effective_date": {"type": "date"},
        }
    }
}

class LexisElasticsearchClient:
    def __init__(self):
        self.client = AsyncElasticsearch(hosts=[settings.elasticsearch_url])
        self.index_name = settings.elasticsearch_index

    async def initialize_index(self):
        exists = await self.client.indices.exists(index=self.index_name)
        if not exists:
            await self.client.indices.create(index=self.index_name, body=INDEX_MAPPING)

    async def index_documents(self, documents: List[Dict[str, Any]]):
        from elasticsearch.helpers import async_bulk
        
        actions = [
            {
                "_index": self.index_name,
                "_id": doc["chunk_id"],
                "_source": doc
            }
            for doc in documents
        ]
        if actions:
            await async_bulk(self.client, actions)

    def _sanitise_query(self, query: str) -> str:
        return re.sub(r'[\(\)\^"\'~\*\?:\\]', ' ', query).strip()

    async def search(self, query_text: str, size: int = 5, doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
        safe_query = self._sanitise_query(query_text)
        
        must_clauses = [
            {
                "multi_match": {
                    "query": safe_query,
                    "fields": ["content", "title^10"],
                    "type": "best_fields",
                    "analyzer": "legal_analyzer",
                }
            }
        ]

        filter_clauses = []
        if doc_type:
            filter_clauses.append({"term": {"doc_type": doc_type}})

        body = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses,
                }
            },
            "size": size,
        }

        response = await self.client.search(index=self.index_name, body=body)
        return [
            {
                "chunk_id": hit["_source"]["chunk_id"],
                "score": hit["_score"],
                "payload": hit["_source"],
            }
            for hit in response["hits"]["hits"]
        ]
