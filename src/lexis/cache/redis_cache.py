"""
WHAT: Redis-backed Evidence Summary Cache.
"""
import os
import json
import logging
from typing import Optional
import redis.asyncio as redis
from .base import EvidenceSummaryCache, CachedEvidence

logger = logging.getLogger(__name__)

class RedisEvidenceSummaryCache(EvidenceSummaryCache):
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = os.getenv("REDIS_URL", redis_url)
        self.client = redis.from_url(self.redis_url, decode_responses=True)
        self.ttl = 604800  # 7 days
        
    def _key(self, chunk_id: str, query_hash: str) -> str:
        return f"evidence:{chunk_id}:{query_hash}"

    async def get(self, chunk_id: str, query_hash: str) -> Optional[CachedEvidence]:
        try:
            raw = await self.client.get(self._key(chunk_id, query_hash))
            if raw:
                return CachedEvidence.parse_raw(raw)
        except Exception as e:
            logger.error(f"Redis get failed: {e}")
        return None

    async def set(self, evidence: CachedEvidence) -> None:
        try:
            key = self._key(evidence.chunk_id, evidence.query_hash)
            await self.client.setex(key, self.ttl, evidence.model_dump_json())
        except Exception as e:
            logger.error(f"Redis set failed: {e}")
            
class MemoryEvidenceSummaryCache(EvidenceSummaryCache):
    """Fallback for local testing without Redis."""
    def __init__(self):
        self.cache = {}
        
    async def get(self, chunk_id: str, query_hash: str) -> Optional[CachedEvidence]:
        key = f"{chunk_id}:{query_hash}"
        return self.cache.get(key)
        
    async def set(self, evidence: CachedEvidence) -> None:
        key = f"{evidence.chunk_id}:{evidence.query_hash}"
        self.cache[key] = evidence

def get_cache() -> EvidenceSummaryCache:
    if os.getenv("REDIS_URL"):
        return RedisEvidenceSummaryCache()
    return MemoryEvidenceSummaryCache()
