"""
Import/contract tests for the deep-mode evidence cache. These modules were
previously un-importable (missing base types), which silently disabled deep
mode's evidence filter; importing them is now a tested guarantee.
"""
import asyncio

from lexis.serving.base import CachedEvidence, EvidenceSummaryCache
from lexis.serving.cache import MemoryEvidenceSummaryCache, get_cache


def make(chunk_id="c1", query_hash="h1", reason="evidence", score=0.7):
    return CachedEvidence(chunk_id=chunk_id, query_hash=query_hash, reason=reason, score=score)


def test_map_reduce_module_is_importable():
    import lexis.reranking.map_reduce_filter as mrf  # noqa: F401
    assert hasattr(mrf, "map_reduce_deep_mode")


def test_memory_cache_roundtrip_keyed_by_chunk_and_query():
    cache = MemoryEvidenceSummaryCache()
    asyncio.run(cache.set(make("c1", "h1", "one")))
    asyncio.run(cache.set(make("c1", "h2", "two")))
    assert asyncio.run(cache.get("c1", "h1")).reason == "one"
    assert asyncio.run(cache.get("c1", "h2")).reason == "two"
    assert asyncio.run(cache.get("c2", "h1")) is None


def test_get_cache_returns_memory_cache_without_redis_env(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert isinstance(get_cache(), MemoryEvidenceSummaryCache)
    assert isinstance(get_cache(), EvidenceSummaryCache)


def test_cached_evidence_json_roundtrip():
    original = make()
    restored = CachedEvidence.model_validate_json(original.model_dump_json())
    assert restored == original
