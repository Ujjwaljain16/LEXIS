"""
Tests that RetrievalEngine._path_d_bm25 (hybrid_retriever.py) is wired to
the local bm25s index (ADR-003) rather than Elasticsearch, runs it off the
event loop, and builds Candidates with the real application chunk_id (not
a corpus index or any other identifier).
"""
import pytest

from lexis.retrieval.hybrid_retriever import RetrievalEngine


class FakeBM25Index:
    def __init__(self, hits):
        self._hits = hits
        self.calls = []

    def search(self, query_text, top_k):
        self.calls.append((query_text, top_k))
        return self._hits


@pytest.mark.asyncio
async def test_path_d_bm25_uses_the_local_index_not_elasticsearch():
    engine = RetrievalEngine()
    assert not hasattr(engine, "es"), "RetrievalEngine must not construct an Elasticsearch client (ADR-003)"
    assert hasattr(engine, "bm25")


@pytest.mark.asyncio
async def test_path_d_bm25_builds_candidates_from_real_chunk_ids():
    fake_index = FakeBM25Index([
        {"chunk_id": "pqac-real-app-id", "score": 3.14, "payload": {"content": "a generic passage", "doc_id": "doc-1"}},
    ])
    engine = RetrievalEngine()
    engine.bm25 = fake_index

    candidates = await engine._path_d_bm25("a generic query", top_k=5)

    assert len(candidates) == 1
    assert candidates[0].chunk_id == "pqac-real-app-id"
    assert candidates[0].score == 3.14
    assert candidates[0].content == "a generic passage"
    assert candidates[0].source_path == "path_d_bm25"
    assert fake_index.calls == [("a generic query", 5)]


@pytest.mark.asyncio
async def test_path_d_bm25_failure_is_isolated_like_the_old_es_path():
    class FailingIndex:
        def search(self, query_text, top_k):
            raise RuntimeError("simulated bm25 failure")

    engine = RetrievalEngine()
    engine.bm25 = FailingIndex()

    candidates = await engine._path_d_bm25("q", top_k=5)

    assert candidates == []
