"""
Regression tests for retrieval/hybrid_retriever.py's R2 HyPE path.

settings.hype_enabled defaults to False: retrieve()/retrieve_with_trace()
must never query the HyPE collection and must produce byte-identical
results to before this path existed (see test_hybrid_retriever_trace.py,
whose fixtures/assertions this reuses unmodified). When enabled, a third
path runs concurrently and its candidates get fused by apply_rrf exactly
like path_b_global/path_d_bm25.
"""
import pytest

from lexis.config import settings
from lexis.retrieval.hybrid_retriever import RetrievalEngine


class FakeEmbedder:
    def embed_text(self, text):
        class _Vec:
            def tolist(self_inner):
                return [0.1, 0.2, 0.3]
        return _Vec()


class FakeQdrant:
    """Records which collection each search() call targeted, and returns a
    different canned result set per collection so a test can prove the
    HyPE path really queried the HyPE collection, not the primary one."""

    def __init__(self, results_by_collection):
        self._results_by_collection = results_by_collection
        self.queried_collections = []

    async def search(self, collection_name, query_vector, top_k=10):
        self.queried_collections.append(collection_name)
        return self._results_by_collection.get(collection_name, [])


class FakeBM25:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query_text, top_k):
        return self._hits


class FakePoint:
    def __init__(self, chunk_id, score, content, doc_id="doc-1", question=None):
        self.score = score
        payload = {"chunk_id": chunk_id, "content": content, "doc_id": doc_id}
        if question is not None:
            payload["question"] = question
        self.payload = payload


def build_engine(qdrant):
    engine = RetrievalEngine()
    engine.embedder = FakeEmbedder()
    engine.qdrant = qdrant
    engine.bm25 = FakeBM25([])
    return engine


@pytest.mark.asyncio
async def test_hype_path_default_disabled_never_queries_the_hype_collection(monkeypatch):
    monkeypatch.setattr(settings, "hype_enabled", False)
    qdrant = FakeQdrant({
        settings.qdrant_collection_primary: [FakePoint("pqac-dense-1", 0.9, "dense hit")],
        settings.qdrant_collection_hype: [FakePoint("pqac-dense-1", 0.99, "dense hit", question="a hype question")],
    })
    engine = build_engine(qdrant)

    trace = await engine.retrieve_with_trace("a query", top_k_per_path=10, top_n_rrf=10)

    assert settings.qdrant_collection_hype not in qdrant.queried_collections
    assert trace.hype_candidates == []


@pytest.mark.asyncio
async def test_hype_path_enabled_queries_the_hype_collection_and_is_fused(monkeypatch):
    monkeypatch.setattr(settings, "hype_enabled", True)
    qdrant = FakeQdrant({
        settings.qdrant_collection_primary: [FakePoint("pqac-dense-1", 0.9, "dense hit")],
        settings.qdrant_collection_hype: [
            FakePoint("pqac-hype-only", 0.95, "found only via a hypothetical question",
                      question="What is the notice period?"),
        ],
    })
    engine = build_engine(qdrant)

    trace = await engine.retrieve_with_trace("what is the notice period", top_k_per_path=10, top_n_rrf=10)

    assert settings.qdrant_collection_hype in qdrant.queried_collections
    hype_ids = [c.chunk_id for c in trace.hype_candidates]
    assert hype_ids == ["pqac-hype-only"]

    fused_ids = {c.chunk_id for c in trace.fused_candidates}
    assert "pqac-hype-only" in fused_ids

    # The HyPE candidate's content must be the real chunk content (denormalized at ingest
    # time), never the hypothetical question text itself -- a downstream citation must show
    # the actual contract language, not a generated question.
    hype_final = next(c for c in trace.fused_candidates if c.chunk_id == "pqac-hype-only")
    assert hype_final.content == "found only via a hypothetical question"


@pytest.mark.asyncio
async def test_retrieve_and_retrieve_with_trace_agree_with_hype_enabled(monkeypatch):
    monkeypatch.setattr(settings, "hype_enabled", True)

    def make_qdrant():
        return FakeQdrant({
            settings.qdrant_collection_primary: [FakePoint("pqac-dense-1", 0.9, "dense hit")],
            settings.qdrant_collection_hype: [FakePoint("pqac-hype-1", 0.8, "hype hit", question="q?")],
        })

    engine_a = build_engine(make_qdrant())
    engine_b = build_engine(make_qdrant())

    plain_result = await engine_a.retrieve("q", top_k_per_path=10, top_n_rrf=10)
    trace = await engine_b.retrieve_with_trace("q", top_k_per_path=10, top_n_rrf=10)

    assert plain_result == trace.final_chunks
    assert any(c["id"] == "pqac-hype-1" for c in plain_result)
