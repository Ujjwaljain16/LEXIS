"""
WHAT: Protocol for Evidence Summary Caching to control Deep Mode costs.
"""
from typing import Protocol, Optional
from datetime import datetime
from pydantic import BaseModel

class CachedEvidence(BaseModel):
    chunk_id: str
    query_hash: str
    reason: str
    score: float
    timestamp: datetime

class EvidenceSummaryCache(Protocol):
    async def get(self, chunk_id: str, query_hash: str) -> Optional[CachedEvidence]:
        ...
        
    async def set(self, evidence: CachedEvidence) -> None:
        ...
