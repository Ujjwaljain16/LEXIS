"""
Proves RetrievalEngine.retrieve_with_trace() (Step 8 instrumentation) does
not change the behavior of the existing, baseline-verified retrieve()
method: for identical inputs and identical mocked dense/BM25 results, both
methods must return identical final_chunks. This is what makes it safe to
say the diagnostic instrumentation cannot have altered the Step 7 baseline.
"""
import pytest

from lexis.retrieval.hybrid_retriever import RetrievalEngine
from lexis.retrieval.interfaces import Candidate


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


def build_engine():
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
    return engine


@pytest.mark.asyncio
async def test_retrieve_and_retrieve_with_trace_produce_identical_final_chunks():
    engine_a = build_engine()
    engine_b = build_engine()

    plain_result = await engine_a.retrieve("a generic query", top_k_per_path=10, top_n_rrf=10)
    trace = await engine_b.retrieve_with_trace("a generic query", top_k_per_path=10, top_n_rrf=10)

    assert plain_result == trace.final_chunks


@pytest.mark.asyncio
async def test_trace_exposes_the_raw_per_path_results_retrieve_discards():
    engine = build_engine()
    trace = await engine.retrieve_with_trace("q", top_k_per_path=10, top_n_rrf=10)

    dense_ids = [c.chunk_id for c in trace.dense_candidates]
    bm25_ids = [c.chunk_id for c in trace.bm25_candidates]
    fused_ids = [c.chunk_id for c in trace.fused_candidates]

    assert dense_ids == ["pqac-dense-1", "pqac-shared"]
    assert bm25_ids == ["pqac-shared", "pqac-bm25-1"]
    # fused must include every candidate seen by either path (RRF unions paths).
    assert set(fused_ids) == {"pqac-dense-1", "pqac-shared", "pqac-bm25-1"}


@pytest.mark.asyncio
async def test_trace_final_chunks_respects_top_n_rrf_cutoff_same_as_retrieve():
    engine_a = build_engine()
    engine_b = build_engine()

    plain_result = await engine_a.retrieve("q", top_k_per_path=10, top_n_rrf=1)
    trace = await engine_b.retrieve_with_trace("q", top_k_per_path=10, top_n_rrf=1)

    assert len(plain_result) == 1
    assert plain_result == trace.final_chunks
