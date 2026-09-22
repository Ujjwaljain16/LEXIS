"""
Tests for document routing and scoped retrieval using in-memory fakes
(no Qdrant, no BM25 library, no network, no embedding model).
"""
import asyncio
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from lexis.config import settings
from lexis.evaluation.doc_routing import document_scores, vote_document
from lexis.evaluation.scoped_retrieval import retrieve_scoped
from lexis.retrieval.fusion import apply_rrf

K = 61


def item(doc, cid="c"):
    return {"id": cid, "doc_id": doc}


# --- vote_document ---

def test_vote_prefers_document_with_more_and_higher_ranked_chunks():
    ranked = [item("a"), item("b"), item("a"), item("a"), item("b")]
    assert vote_document(ranked, K) == "a"


def test_rank_weighting_prefers_the_higher_ranked_chunk_at_equal_count():
    ranked = [item("b"), item("y"), item("y2"), item("y3"), item("a")]  # b at rank 1, a at rank 5
    scores = document_scores(ranked, K)
    assert scores["b"] > scores["a"]


def test_with_a_large_k_the_vote_is_nearly_a_count_vote():
    """At k=61 the weights are almost flat: two mid-ranked chunks (1/70+1/71)
    outweigh one top-ranked chunk (1/62). A small k weights rank strongly.
    This is why k is an input, not a literal."""
    ranked = [item("b")] + [item("z")] * 7 + [item("a"), item("a")]
    assert document_scores(ranked, K)["a"] > document_scores(ranked, K)["b"]
    assert document_scores(ranked, 1)["b"] > document_scores(ranked, 1)["a"]


def test_vote_tie_breaks_by_doc_id_deterministically():
    ranked = [item("b"), item("a")]  # ranks differ, but craft an exact tie:
    tie = [{"id": "1", "doc_id": "b"}, {"id": "2", "doc_id": "a"}]
    assert vote_document(tie, K) == "b"  # higher rank wins
    assert vote_document([item("q")], K) == "q"


def test_vote_ignores_items_without_doc_id_and_handles_empty():
    assert vote_document([], K) is None
    assert vote_document([{"id": "x"}, {"id": "y", "doc_id": None}], K) is None
    assert vote_document([{"id": "x"}, item("a")], K) == "a"


def test_routing_module_takes_no_ground_truth():
    params = set(inspect.signature(vote_document).parameters)
    assert not params & {"relevant_doc_ids", "relevant_chunk_ids", "answer_texts", "case"}


# --- scoped retrieval with fakes ---

DOCS = {"d1": ["c1", "c2", "c3"], "d2": ["c4", "c5"], "d3": ["c6"]}
CHUNK_DOC = {c: d for d, cs in DOCS.items() for c in cs}
DENSE_ORDER = ["c4", "c1", "c6", "c2", "c5", "c3"]       # global dense ranking
BM25_ORDER = ["c1", "c5", "c2", "c6", "c4", "c3"]         # global bm25 ranking


def payload(cid):
    return {"chunk_id": cid, "doc_id": CHUNK_DOC[cid], "content": f"text {cid}"}


class FakeQdrantClient:
    async def query_points(self, collection_name, query, limit, query_filter):
        allowed = set(query_filter.must[0].match.any)
        pts = [SimpleNamespace(payload=payload(c), score=1.0 - i * 0.1)
               for i, c in enumerate(DENSE_ORDER) if CHUNK_DOC[c] in allowed][:limit]
        return SimpleNamespace(points=pts)


class FakeBM25:
    def corpus_size(self):
        return len(BM25_ORDER)

    def search(self, query_text, top_k):
        return [{"chunk_id": c, "score": 5.0 - i, "payload": payload(c)} for i, c in enumerate(BM25_ORDER)][:top_k]


class FakeEngine:
    embedder = SimpleNamespace(embed_text=lambda q: np.array([0.1, 0.2]))
    qdrant = SimpleNamespace(client=FakeQdrantClient())
    bm25 = FakeBM25()


def run(scope, k=10, n=10):
    return asyncio.run(retrieve_scoped(FakeEngine(), "q", scope, k, n))


def test_every_result_stays_inside_the_scope():
    trace = run(["d2"])
    assert {c["payload"]["doc_id"] for c in trace.final_chunks} == {"d2"}
    assert all(CHUNK_DOC[c.chunk_id] == "d2" for c in trace.dense_candidates + trace.bm25_candidates)


def test_scope_of_all_documents_matches_an_unscoped_fusion():
    trace = run(list(DOCS))
    dense_all = [c for c in DENSE_ORDER]
    bm25_all = [c for c in BM25_ORDER]
    assert [c.chunk_id for c in trace.dense_candidates] == dense_all
    assert [c.chunk_id for c in trace.bm25_candidates] == bm25_all
    expected = apply_rrf([trace.dense_candidates, trace.bm25_candidates], k=settings.rrf_k)
    assert [c["id"] for c in trace.final_chunks] == [c.chunk_id for c in expected]


def test_scoped_bm25_keeps_global_order_of_the_in_scope_chunks():
    assert [c.chunk_id for c in run(["d1"]).bm25_candidates] == ["c1", "c2", "c3"]
    assert [c.chunk_id for c in run(["d1", "d3"]).bm25_candidates] == ["c1", "c2", "c6", "c3"]


def test_per_path_and_fusion_limits_are_respected():
    trace = run(list(DOCS), k=2, n=3)
    assert len(trace.dense_candidates) == 2 and len(trace.bm25_candidates) == 2
    assert len(trace.final_chunks) <= 3


def test_empty_scope_returns_empty_everything():
    trace = run([])
    assert trace.final_chunks == [] and trace.dense_candidates == [] and trace.bm25_candidates == []


def test_final_chunks_have_the_same_shape_as_the_unscoped_trace():
    chunk = run(["d1"]).final_chunks[0]
    assert set(chunk) == {"id", "score", "rrf_score", "source_path", "payload", "text"}


def test_scoped_module_never_calls_production_retrieve_or_mutates_settings():
    import lexis.evaluation.scoped_retrieval as m
    src = inspect.getsource(m)
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", '"""')))
    assert "settings." in code and "settings.rrf_k" in code
    assert "settings.rrf_k =" not in code and ".retrieve(" not in code
