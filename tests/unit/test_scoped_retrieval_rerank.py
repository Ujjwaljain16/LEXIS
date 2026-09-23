"""
Regression tests for evaluation/scoped_retrieval.py's R4 rerank wiring.

This is the doc_scoped protocol used by the frozen CUAD test baseline, so
rerank must operate on the already-scoped fused candidates (never pulling
in anything outside the document scope -- reranking only reorders, it
can't introduce new candidates). The `engine`-like object only needs a
`reranker` attribute (getattr with a None default), matching the
lightweight fakes this module's tests already use elsewhere.
"""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from lexis.config import settings
from lexis.evaluation.scoped_retrieval import retrieve_scoped

DOCS = {"d1": ["c1", "c2"], "d2": ["c3"]}
CHUNK_DOC = {c: d for d, cs in DOCS.items() for c in cs}


def payload(cid):
    return {"chunk_id": cid, "doc_id": CHUNK_DOC[cid], "content": f"text {cid}"}


class FakeQdrantClient:
    async def query_points(self, collection_name, query, limit, query_filter):
        allowed = set(query_filter.must[0].match.any)
        pts = [SimpleNamespace(payload=payload(c), score=1.0 - i * 0.1)
               for i, c in enumerate(["c1", "c2"]) if CHUNK_DOC[c] in allowed][:limit]
        return SimpleNamespace(points=pts)


class FakeBM25:
    def corpus_size(self):
        return 0

    def search(self, query_text, top_k):
        return []


class ReversingReranker:
    def __init__(self):
        self.calls = []

    async def transform(self, query, candidates):
        self.calls.append([c.chunk_id for c in candidates])
        return list(reversed(candidates))


def make_engine(reranker=None):
    return SimpleNamespace(
        embedder=SimpleNamespace(embed_text=lambda q: np.array([0.1, 0.2])),
        qdrant=SimpleNamespace(client=FakeQdrantClient()),
        bm25=FakeBM25(),
        reranker=reranker,
    )


def run(engine, scope, k=10, n=10):
    return asyncio.run(retrieve_scoped(engine, "q", scope, k, n))


def test_no_reranker_attribute_at_all_behaves_like_disabled():
    """An engine-like object that doesn't even set .reranker (e.g. an older
    test fake) must not crash -- getattr(..., None) covers it."""
    engine = SimpleNamespace(
        embedder=SimpleNamespace(embed_text=lambda q: np.array([0.1, 0.2])),
        qdrant=SimpleNamespace(client=FakeQdrantClient()),
        bm25=FakeBM25(),
    )
    trace = run(engine, ["d1"])
    assert [c["id"] for c in trace.final_chunks] == [c.chunk_id for c in trace.fused_candidates]


def test_rerank_disabled_leaves_ordering_unchanged():
    engine = make_engine(reranker=None)
    trace = run(engine, ["d1"])
    assert [c["id"] for c in trace.final_chunks] == [c.chunk_id for c in trace.fused_candidates]


def test_rerank_enabled_changes_the_final_order_but_stays_within_scope():
    engine = make_engine(reranker=ReversingReranker())
    trace = run(engine, ["d1"])

    fused_order = [c.chunk_id for c in trace.fused_candidates]
    final_order = [c["id"] for c in trace.final_chunks]
    assert final_order == list(reversed(fused_order))
    assert all(CHUNK_DOC[cid] == "d1" for cid in final_order)  # never leaks outside the scope
    assert engine.reranker.calls


def test_rerank_never_corrupts_the_fused_candidates_trace():
    engine = make_engine(reranker=ReversingReranker())
    trace = run(engine, ["d1"])

    fused_scores = [c.score for c in trace.fused_candidates]
    assert fused_scores == sorted(fused_scores, reverse=True)
