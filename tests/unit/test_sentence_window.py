"""
Regression tests for SentenceWindowExpansion (reranking/sentence_window.py).

Bugs fixed:
1. ChunkMetadata/Qdrant payload never carried a chunk_index, so the
   doc_id/chunk_index guard always short-circuited and expansion never ran.
2. Even with chunk_index present, the old code constructed a synthetic
   neighbour id (f"{doc_id}_chunk_{idx}") and fetched it by id. Real chunk
   ids are content-derived hashes (see Chunk.create), so that id could never
   match a real point -- expansion was a structural no-op regardless of
   metadata. The fix looks neighbours up by (doc_id, chunk_index) instead.

These tests use a synthetic, non-legal, variable-length corpus (arbitrary
doc ids, arbitrary chunk counts, arbitrary positions) to make sure nothing
here assumes a fixed document structure or a minimum number of chunks.
"""
import pytest

from lexis.reranking.sentence_window import SentenceWindowExpansion
from lexis.retrieval.interfaces import Query, Candidate


def make_candidate(doc_id: str, chunk_index: int, content: str, source_path: str = "path_b_global") -> Candidate:
    return Candidate(
        chunk_id=f"pqac-fake-{doc_id}-{chunk_index}",
        score=1.0,
        source_path=source_path,
        metadata={"doc_id": doc_id, "chunk_index": chunk_index},
        content=content,
    )


class FakeCorpusIndex:
    """Simulates a Qdrant lookup by (doc_id, chunk_index) payload match."""

    def __init__(self, chunks_by_doc: dict):
        self._chunks_by_doc = chunks_by_doc
        self.calls = []

    async def fetch(self, doc_id: str, chunk_index: int):
        self.calls.append((doc_id, chunk_index))
        return self._chunks_by_doc.get(doc_id, {}).get(chunk_index)


@pytest.mark.asyncio
async def test_expands_using_position_based_lookup_not_constructed_id():
    # An arbitrary 3-chunk "report" -- not a contract/clause structure.
    corpus = {
        "report-42": {
            0: make_candidate("report-42", 0, "Quarterly summary opens here."),
            1: make_candidate("report-42", 1, "Middle section with the key figure."),
            2: make_candidate("report-42", 2, "Closing remarks follow."),
        }
    }
    index = FakeCorpusIndex(corpus)
    expander = SentenceWindowExpansion(fetch_chunk_fn=index.fetch, window_size=1)

    anchor = corpus["report-42"][1]
    result = await expander.transform(Query(text="q"), [anchor])

    assert len(result) == 1
    expanded = result[0]
    assert expanded.chunk_id == anchor.chunk_id  # anchor id preserved as the citation key
    assert "Quarterly summary opens here." in expanded.content
    assert "Middle section with the key figure." in expanded.content
    assert "Closing remarks follow." in expanded.content
    assert expanded.source_path.endswith(",window_expanded")

    # Lookup happened by (doc_id, index) -- proves neighbours are found by
    # position, not by guessing a chunk id string.
    assert ("report-42", 0) in index.calls
    assert ("report-42", 2) in index.calls


@pytest.mark.asyncio
async def test_respects_window_size_larger_than_one():
    corpus = {"doc-w": {i: make_candidate("doc-w", i, f"chunk number {i}") for i in range(5)}}
    index = FakeCorpusIndex(corpus)
    expander = SentenceWindowExpansion(fetch_chunk_fn=index.fetch, window_size=2)

    anchor = corpus["doc-w"][2]
    result = await expander.transform(Query(text="q"), [anchor])

    content = result[0].content
    for i in range(5):
        assert f"chunk number {i}" in content


@pytest.mark.asyncio
async def test_no_expansion_when_document_has_a_single_chunk():
    """No assumption of a minimum document length: a one-chunk document must
    not error and must not be 'expanded' with anything."""
    corpus = {"solo-doc": {0: make_candidate("solo-doc", 0, "The only chunk in this document.")}}
    index = FakeCorpusIndex(corpus)
    expander = SentenceWindowExpansion(fetch_chunk_fn=index.fetch, window_size=2)

    anchor = corpus["solo-doc"][0]
    result = await expander.transform(Query(text="q"), [anchor])

    assert len(result) == 1
    assert result[0].content == "The only chunk in this document."
    assert not result[0].source_path.endswith(",window_expanded")


@pytest.mark.asyncio
async def test_passthrough_when_chunk_index_missing_from_metadata():
    """e.g. a candidate sourced from a path that doesn't populate chunk_index
    (such as the current BM25/ES path) must pass through unchanged, not crash."""
    candidate = Candidate(
        chunk_id="pqac-no-position-metadata",
        score=1.0,
        source_path="path_d_bm25",
        metadata={"doc_id": "some-doc"},  # no chunk_index
        content="Content without positional metadata.",
    )

    async def fetch_chunk_fn(doc_id, chunk_index):
        raise AssertionError("fetch_chunk_fn must not be called when chunk_index is absent")

    expander = SentenceWindowExpansion(fetch_chunk_fn=fetch_chunk_fn, window_size=1)
    result = await expander.transform(Query(text="q"), [candidate])

    assert result == [candidate]


@pytest.mark.asyncio
async def test_neighbor_fetch_failure_is_isolated_and_does_not_crash():
    async def flaky_fetch(doc_id, chunk_index):
        raise RuntimeError("simulated transient lookup failure")

    anchor = make_candidate("doc-x", 5, "Anchor content that must survive.")
    expander = SentenceWindowExpansion(fetch_chunk_fn=flaky_fetch, window_size=1)

    result = await expander.transform(Query(text="q"), [anchor])

    assert len(result) == 1
    assert result[0].content == "Anchor content that must survive."


@pytest.mark.asyncio
async def test_duplicate_candidates_are_processed_once():
    anchor = make_candidate("doc-y", 3, "Some content.")
    index = FakeCorpusIndex({"doc-y": {3: anchor}})
    expander = SentenceWindowExpansion(fetch_chunk_fn=index.fetch, window_size=1)

    result = await expander.transform(Query(text="q"), [anchor, anchor])

    assert len(result) == 1
