"""
Regression tests for retrieval/hybrid_retriever.py's R4 rerank wiring
(RetrievalEngine._maybe_rerank, and its use in retrieve()/retrieve_with_trace()).

settings.rerank_enabled defaults to False: RetrievalEngine never constructs
a reranker, and retrieval must be byte-identical to before this path
existed.
"""
import pytest

from lexis.config import settings
from lexis.retrieval.hybrid_retriever import RetrievalEngine
from lexis.retrieval.interfaces import Candidate, Query


class FakeEmbedder:
    def embed_text(self, text):
        class _Vec:
            def tolist(self_inner):
                return [0.1, 0.2, 0.3]
        return _Vec()


class FakeQdrant:
    def __init__(self, results):
        self._results = results

    async def search(self, collection_name, query_vector, top_k=10):
        return self._results


class FakeBM25:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query_text, top_k):
        return self._hits


class FakePoint:
    def __init__(self, chunk_id, score, content, doc_id="doc-1"):
        self.score = score
        self.payload = {"chunk_id": chunk_id, "content": content, "doc_id": doc_id}


class ReversingReranker:
    """A fake reranker that just reverses whatever list it's given --
    deterministic, and provably NOT the same order apply_rrf would produce,
    so a test can tell whether reranking actually ran."""
    def __init__(self):
        self.calls = []

    async def transform(self, query: Query, candidates):
        self.calls.append((query.text, [c.chunk_id for c in candidates]))
        return list(reversed(candidates))


def build_engine(rerank=False):
    engine = RetrievalEngine()
    engine.embedder = FakeEmbedder()
    engine.qdrant = FakeQdrant([
        FakePoint("pqac-dense-1", 0.9, "dense hit one"),
        FakePoint("pqac-shared", 0.8, "found by both paths"),
    ])
    engine.bm25 = FakeBM25([
        {"chunk_id": "pqac-shared", "score": 5.0, "payload": {"content": "found by both paths", "doc_id": "doc-1"}},
        {"chunk_id": "pqac-bm25-1", "score": 4.0, "payload": {"content": "bm25 only hit", "doc_id": "doc-2"}},
    ])
    engine.reranker = ReversingReranker() if rerank else None
    return engine


@pytest.mark.asyncio
async def test_rerank_disabled_by_default_leaves_ordering_unchanged():
    monkeypatch_off = build_engine(rerank=False)
    trace = await monkeypatch_off.retrieve_with_trace("q", top_k_per_path=10, top_n_rrf=10)

    fused_order = [c.chunk_id for c in trace.fused_candidates]
    final_order = [c["id"] for c in trace.final_chunks]
    assert fused_order == final_order  # no reranking applied


@pytest.mark.asyncio
async def test_rerank_enabled_actually_changes_the_final_order():
    engine = build_engine(rerank=True)
    trace = await engine.retrieve_with_trace("q", top_k_per_path=10, top_n_rrf=10)

    fused_order = [c.chunk_id for c in trace.fused_candidates]
    final_order = [c["id"] for c in trace.final_chunks]
    assert final_order == list(reversed(fused_order))
    assert engine.reranker.calls  # the reranker was actually invoked


@pytest.mark.asyncio
async def test_rerank_does_not_corrupt_the_pre_rerank_fused_candidates_trace():
    """RetrievalTrace.fused_candidates documents the RRF-only stage --
    reranking must not retroactively change what that field reports."""
    engine = build_engine(rerank=True)
    trace = await engine.retrieve_with_trace("q", top_k_per_path=10, top_n_rrf=10)

    # fused_candidates must still be in RRF order (by RRF score), not reversed.
    fused_scores = [c.score for c in trace.fused_candidates]
    assert fused_scores == sorted(fused_scores, reverse=True)


@pytest.mark.asyncio
async def test_retrieve_and_retrieve_with_trace_agree_with_rerank_enabled():
    engine_a = build_engine(rerank=True)
    engine_b = build_engine(rerank=True)

    plain_result = await engine_a.retrieve("q", top_k_per_path=10, top_n_rrf=10)
    trace = await engine_b.retrieve_with_trace("q", top_k_per_path=10, top_n_rrf=10)

    assert plain_result == trace.final_chunks


@pytest.mark.asyncio
async def test_rerank_only_scores_the_top_rerank_top_k_and_appends_the_rest(monkeypatch):
    monkeypatch.setattr(settings, "rerank_top_k", 1)
    engine = build_engine(rerank=True)

    trace = await engine.retrieve_with_trace("q", top_k_per_path=10, top_n_rrf=10)

    # Only the single top-ranked candidate should have been handed to the reranker.
    assert len(engine.reranker.calls[0][1]) == 1
    # But the final list must still contain every fused candidate -- rerank never drops any.
    assert {c["id"] for c in trace.final_chunks} == {c.chunk_id for c in trace.fused_candidates}


@pytest.mark.asyncio
async def test_reranker_is_not_constructed_when_rerank_is_disabled():
    engine = RetrievalEngine()  # real construction path, settings.rerank_enabled is False here
    assert engine.reranker is None
