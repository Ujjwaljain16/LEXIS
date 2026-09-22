"""
Evidence-summary cache contract used by deep-mode map-reduce
(reranking/map_reduce_filter.py) and implemented by serving/cache.py.

These types were referenced by both modules but never defined, so neither
module could be imported. Fields mirror exactly how they are constructed
and read: (chunk_id, query_hash) is the cache key; reason/score are the
LLM-extracted evidence text and its confidence.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CachedEvidence(BaseModel):
    chunk_id: str
    query_hash: str
    reason: str
    score: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvidenceSummaryCache(ABC):
    @abstractmethod
    async def get(self, chunk_id: str, query_hash: str) -> Optional[CachedEvidence]:
        ...

    @abstractmethod
    async def set(self, evidence: CachedEvidence) -> None:
        ...
