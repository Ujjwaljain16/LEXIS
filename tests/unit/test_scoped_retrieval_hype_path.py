"""
Regression tests for evaluation/scoped_retrieval.py's R2 HyPE wiring.

This is the doc_scoped protocol used by the frozen CUAD test baseline
(evaluation/reports/cuad_doc_scoped_test.json), so HyPE must be scoped to
the same document set as the dense/BM25 paths -- never allowed to pull in
a hit from outside the case's relevant_doc_ids. settings.hype_enabled
defaults to False: existing doc_scoped behavior is unchanged unless a test
explicitly opts in (mirrors test_doc_scoping.py's fakes/fixtures).
"""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from lexis.config import settings
from lexis.evaluation.scoped_retrieval import retrieve_scoped

DOCS = {"d1": ["c1", "c2"], "d2": ["c3"]}
CHUNK_DOC = {c: d for d, cs in DOCS.items() for c in cs}


def payload(cid, question=None):
    p = {"chunk_id": cid, "doc_id": CHUNK_DOC[cid], "content": f"text {cid}"}
    if question is not None:
        p["question"] = question
    return p


class FakeQdrantClient:
    """Unlike test_doc_scoping.py's FakeQdrantClient, this one returns
    DIFFERENT results per collection_name so a test can prove the HyPE
    query really hit the HyPE collection, scoped the same way as dense."""

    def __init__(self, results_by_collection):
        self._results_by_collection = results_by_collection
        self.calls = []  # (collection_name, allowed_scope)

    async def query_points(self, collection_name, query, limit, query_filter):
        allowed = set(query_filter.must[0].match.any)
        self.calls.append((collection_name, allowed))
        all_points = self._results_by_collection.get(collection_name, [])
        in_scope = [p for p in all_points if p.payload["doc_id"] in allowed]
        return SimpleNamespace(points=in_scope[:limit])


class FakeBM25:
    def corpus_size(self):
        return 0

    def search(self, query_text, top_k):
        return []


def make_engine(client):
    return SimpleNamespace(
        embedder=SimpleNamespace(embed_text=lambda q: np.array([0.1, 0.2])),
        qdrant=SimpleNamespace(client=client),
        bm25=FakeBM25(),
    )


def run(engine, scope, k=10, n=10):
    return asyncio.run(retrieve_scoped(engine, "q", scope, k, n))


def test_hype_disabled_by_default_never_queries_the_hype_collection(monkeypatch):
    monkeypatch.setattr(settings, "hype_enabled", False)
    client = FakeQdrantClient({
        settings.qdrant_collection_primary: [SimpleNamespace(payload=payload("c1"), score=0.9)],
        settings.qdrant_collection_hype: [SimpleNamespace(payload=payload("c1", question="q?"), score=0.99)],
    })
    engine = make_engine(client)

    trace = run(engine, ["d1"])

    assert all(c != settings.qdrant_collection_hype for c, _ in client.calls)
    assert trace.hype_candidates == []


def test_hype_enabled_is_scoped_to_the_same_documents_as_dense(monkeypatch):
    monkeypatch.setattr(settings, "hype_enabled", True)
    client = FakeQdrantClient({
        settings.qdrant_collection_primary: [SimpleNamespace(payload=payload("c1"), score=0.9)],
        settings.qdrant_collection_hype: [
            SimpleNamespace(payload=payload("c1", question="in scope?"), score=0.95),
            SimpleNamespace(payload=payload("c3", question="out of scope?"), score=0.99),  # belongs to d2
        ],
    })
    engine = make_engine(client)

    trace = run(engine, ["d1"])  # scope excludes d2/c3

    hype_ids = [c.chunk_id for c in trace.hype_candidates]
    assert hype_ids == ["c1"]  # c3 filtered out even though it scored higher
    assert all(CHUNK_DOC[c.chunk_id] == "d1" for c in trace.hype_candidates)

    hype_calls = [scope for c, scope in client.calls if c == settings.qdrant_collection_hype]
    assert hype_calls == [{"d1"}]


def test_hype_candidates_are_included_in_the_final_fused_chunks(monkeypatch):
    monkeypatch.setattr(settings, "hype_enabled", True)
    client = FakeQdrantClient({
        settings.qdrant_collection_primary: [],
        settings.qdrant_collection_hype: [SimpleNamespace(payload=payload("c1", question="q?"), score=0.9)],
    })
    engine = make_engine(client)

    trace = run(engine, ["d1"])

    assert any(c["id"] == "c1" and c["source_path"] == "path_hype" for c in trace.final_chunks)


def test_empty_scope_returns_no_hype_candidates_either(monkeypatch):
    monkeypatch.setattr(settings, "hype_enabled", True)
    client = FakeQdrantClient({
        settings.qdrant_collection_hype: [SimpleNamespace(payload=payload("c1", question="q?"), score=0.9)],
    })
    engine = make_engine(client)

    trace = run(engine, [])

    assert trace.hype_candidates == []
    assert trace.final_chunks == []
