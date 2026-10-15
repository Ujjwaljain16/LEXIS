"""R7 document-preamble prior (retrieval/preamble.py + retrieval/scoped.py).

Fakes only: a Qdrant client that supports query_points and a paginated scroll, and a BM25 stub. Checks
that the flag is inert when off, that "first n" means document order by chunk_index (including the
non-contiguous base*100+sub values the chunker produces), that scope is respected, that results are
cached and cannot be corrupted by fusion, and that the prior lifts an opening chunk the other paths
rank poorly."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from lexis.config import settings
from lexis.retrieval.preamble import PreambleIndex
from lexis.retrieval.scoped import retrieve_scoped

# doc -> [(chunk_id, chunk_index)] in a deliberately shuffled storage order
DOCS = {
    "d1": [("d1-late", 7), ("d1-sub", 101), ("d1-first", 0), ("d1-second", 1)],
    "d2": [("d2-b", 3), ("d2-a", 2)],
}


def payload(doc, cid, idx):
    return {"chunk_id": cid, "doc_id": doc, "chunk_index": idx, "content": f"text {cid}"}


class FakeQdrant:
    def __init__(self, dense_order):
        self.dense_order = dense_order
        self.scroll_calls = 0

    async def query_points(self, collection_name, query, limit, query_filter):
        allowed = set(query_filter.must[0].match.any)
        pts = [SimpleNamespace(payload=payload(d, c, i), score=1.0 - n * 0.01)
               for n, (d, c, i) in enumerate(self.dense_order) if d in allowed][:limit]
        return SimpleNamespace(points=pts)

    async def scroll(self, collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
        self.scroll_calls += 1
        doc = scroll_filter.must[0].match.value
        records = [SimpleNamespace(payload=payload(doc, c, i)) for c, i in DOCS[doc]]
        start = offset or 0
        page = records[start:start + limit]
        return page, (start + limit if start + limit < len(records) else None)


class FakeBM25:
    def corpus_size(self):
        return 0

    def search(self, q, k):
        return []


def engine_with(dense_order):
    return SimpleNamespace(embedder=SimpleNamespace(embed_text=lambda q: np.array([0.1])),
                           qdrant=SimpleNamespace(client=FakeQdrant(dense_order)), bm25=FakeBM25(), reranker=None)


# the dense path ranks the opening chunk LAST within d1 -- the failure mode the prior exists for
DENSE = [("d1", "d1-late", 7), ("d1", "d1-sub", 101), ("d1", "d1-second", 1), ("d1", "d1-first", 0)]


def run(engine, scope=("d1",), n=10):
    return asyncio.run(retrieve_scoped(engine, "what is this agreement called", list(scope), 10, n))


def test_prior_is_inert_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "preamble_prior_chunks", 0)
    eng = engine_with(DENSE)
    trace = run(eng)
    assert trace.preamble_candidates == []
    assert eng.qdrant.client.scroll_calls == 0
    assert [c["id"] for c in trace.final_chunks] == ["d1-late", "d1-sub", "d1-second", "d1-first"]


def test_prior_lifts_the_opening_chunk(monkeypatch):
    monkeypatch.setattr(settings, "preamble_prior_chunks", 1)
    trace = run(engine_with(DENSE))
    assert [c.chunk_id for c in trace.preamble_candidates] == ["d1-first"]
    ids = [c["id"] for c in trace.final_chunks]
    assert ids[0] == "d1-first"   # promoted from last place (dense) to first (dense rank 4 + preamble rank 1)


def test_first_n_follows_document_order_not_storage_order_or_a_range_filter(monkeypatch):
    monkeypatch.setattr(settings, "preamble_prior_chunks", 3)
    trace = run(engine_with(DENSE))
    # indices 0, 1, 7 -- NOT 101 (the sub-split chunk sorts after 7) and not the stored order
    assert [c.chunk_id for c in trace.preamble_candidates] == ["d1-first", "d1-second", "d1-late"]


def test_prior_respects_scope_and_orders_documents_as_given(monkeypatch):
    monkeypatch.setattr(settings, "preamble_prior_chunks", 1)
    dense = DENSE + [("d2", "d2-a", 2), ("d2", "d2-b", 3)]
    trace = run(engine_with(dense), scope=("d2", "d1"))
    assert [c.chunk_id for c in trace.preamble_candidates] == ["d2-a", "d1-first"]
    only_d1 = run(engine_with(dense), scope=("d1",))
    assert all(c.metadata["doc_id"] == "d1" for c in only_d1.preamble_candidates)


def test_document_chunks_are_fetched_once_and_cached(monkeypatch):
    monkeypatch.setattr(settings, "preamble_prior_chunks", 1)
    eng = engine_with(DENSE)
    run(eng)
    run(eng)
    assert eng.qdrant.client.scroll_calls == 1


def test_pagination_is_followed(monkeypatch):
    monkeypatch.setattr(settings, "preamble_scroll_batch", 2)
    idx = PreambleIndex(FakeQdrant(DENSE), "c")
    got = asyncio.run(idx.first_chunks("d1", 4))
    assert [c.chunk_id for c in got] == ["d1-first", "d1-second", "d1-late", "d1-sub"]
    assert idx._client.scroll_calls == 2   # 4 records / batch of 2


def test_fusion_cannot_corrupt_the_cache(monkeypatch):
    monkeypatch.setattr(settings, "preamble_prior_chunks", 1)
    eng = engine_with(DENSE)
    run(eng)
    run(eng)
    cached = eng._preamble_index._cache["d1"][0]
    assert cached.source_path == "path_preamble"   # not "path_preamble,path_b_global"
