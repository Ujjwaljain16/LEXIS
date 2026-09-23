"""
Tests for indexing/bm25_index.py::LexisBM25Index (ADR-003: replaces
Elasticsearch as Path D's keyword-search backend). Uses the real bm25s
library -- it's pure Python/numpy, no server or model download, so there's
no reason to fake it. Every index lives in a pytest tmp_path so nothing
here touches the real data/bm25_index/ directory or leaves files behind.
"""
import pytest

from lexis.indexing.bm25_index import LexisBM25Index


def make_doc(chunk_id, doc_id, content):
    return {"chunk_id": chunk_id, "doc_id": doc_id, "content": content}


def test_search_before_any_documents_added_returns_empty(tmp_path):
    index = LexisBM25Index(index_dir=str(tmp_path / "bm25"))
    assert index.search("anything", top_k=10) == []


def test_added_documents_are_immediately_searchable(tmp_path):
    index = LexisBM25Index(index_dir=str(tmp_path / "bm25"))
    index.add_documents([
        make_doc("c1", "doc-1", "a cat is a feline and likes to purr"),
        make_doc("c2", "doc-1", "a dog is the best friend and loves to play fetch"),
        make_doc("c3", "doc-2", "a fish is a creature that lives in water and swims"),
    ])

    results = index.search("does the fish like water", top_k=2)

    assert len(results) > 0
    top_hit = results[0]
    assert top_hit["chunk_id"] == "c3"
    assert top_hit["payload"]["content"] == "a fish is a creature that lives in water and swims"
    assert isinstance(top_hit["score"], float)


def test_results_match_the_es_client_shape(tmp_path):
    """chunk_id/score/payload -- the same shape the (now retired)
    Elasticsearch client returned, so callers needed no changes beyond the
    swap itself."""
    index = LexisBM25Index(index_dir=str(tmp_path / "bm25"))
    index.add_documents([make_doc("c1", "doc-1", "quarterly staffing review policy")])
    results = index.search("staffing policy", top_k=1)
    assert set(results[0].keys()) == {"chunk_id", "score", "payload"}


def test_top_k_is_respected_and_capped_to_corpus_size(tmp_path):
    index = LexisBM25Index(index_dir=str(tmp_path / "bm25"))
    index.add_documents([make_doc(f"c{i}", "doc-1", f"generic passage number {i} about routing") for i in range(3)])

    results = index.search("routing", top_k=100)  # more than the corpus has
    assert len(results) == 3


def test_persisted_corpus_survives_a_new_instance_without_readding(tmp_path):
    """Ingestion and retrieval construct separate LexisBM25Index objects
    (see pipeline.py / hybrid_retriever.py) -- a fresh instance pointed at
    the same directory must see documents added by an earlier one."""
    index_dir = str(tmp_path / "bm25")
    writer = LexisBM25Index(index_dir=index_dir)
    writer.add_documents([make_doc("c1", "doc-1", "a report about quarterly incident counts")])

    reader = LexisBM25Index(index_dir=index_dir)  # brand new instance, nothing added yet
    results = reader.search("incident counts", top_k=5)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"


def test_documents_from_multiple_add_calls_are_all_searchable(tmp_path):
    """Simulates two separate documents ingested one after another --
    proves the rebuild-on-write approach doesn't drop earlier documents."""
    index = LexisBM25Index(index_dir=str(tmp_path / "bm25"))
    index.add_documents([make_doc("c1", "doc-1", "first document about renewal timelines")])
    index.add_documents([make_doc("c2", "doc-2", "second document about staffing levels")])

    renewal_hits = index.search("renewal timelines", top_k=5)
    staffing_hits = index.search("staffing levels", top_k=5)

    assert any(h["chunk_id"] == "c1" for h in renewal_hits)
    assert any(h["chunk_id"] == "c2" for h in staffing_hits)


def test_readding_the_same_chunk_id_replaces_it_instead_of_duplicating(tmp_path):
    """Regression: re-ingesting the same document (e.g. a retried or
    idempotent re-run) must not double the corpus -- Qdrant upserts by
    point id; this index must match that semantics by chunk_id."""
    index = LexisBM25Index(index_dir=str(tmp_path / "bm25"))
    index.add_documents([make_doc("c1", "doc-1", "original content about routing")])
    index.add_documents([make_doc("c1", "doc-1", "original content about routing")])  # same chunk_id again

    results = index.search("routing", top_k=10)
    assert len(results) == 1  # not 2

    with open(str(tmp_path / "bm25" / "corpus.jsonl"), encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 1


def test_readding_the_same_chunk_id_with_new_content_updates_it(tmp_path):
    index = LexisBM25Index(index_dir=str(tmp_path / "bm25"))
    index.add_documents([make_doc("c1", "doc-1", "old wording about staffing")])
    index.add_documents([make_doc("c1", "doc-1", "revised wording about scheduling")])

    results = index.search("scheduling", top_k=10)
    assert len(results) == 1
    assert results[0]["payload"]["content"] == "revised wording about scheduling"


def test_empty_query_returns_no_results_not_a_crash(tmp_path):
    index = LexisBM25Index(index_dir=str(tmp_path / "bm25"))
    index.add_documents([make_doc("c1", "doc-1", "some content")])
    assert index.search("   ", top_k=5) == []


def test_index_text_is_tokenized_but_content_is_returned_unmodified(tmp_path):
    """Regression: pipeline.py sets index_text to a CCH-prefixed version of
    the chunk (doc title/type/section header) so BM25 matching benefits
    from that context the same way dense embedding already does, but
    "content" must stay the clean, unprefixed text -- path_d_bm25.py reads
    payload["content"] as the candidate's actual displayed/cited text, so
    leaking the header into it would corrupt citations."""
    index = LexisBM25Index(index_dir=str(tmp_path / "bm25"))
    index.add_documents([{
        "chunk_id": "c1",
        "doc_id": "doc-1",
        "content": "the term of this agreement is five years",
        "index_text": "Document: Master Supply Agreement\nType: contract\nSection: Term\n\n"
                       "the term of this agreement is five years",
    }])

    # A query that only matches words from the header (not "content") must still hit,
    # proving index_text -- not content -- is what got tokenized.
    results = index.search("master supply agreement term section", top_k=5)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["payload"]["content"] == "the term of this agreement is five years"
    assert "Document:" not in results[0]["payload"]["content"]


def test_missing_index_text_falls_back_to_content(tmp_path):
    """Older/other callers that only ever set "content" (no index_text) --
    e.g. this test file's own make_doc() -- must keep working unchanged."""
    index = LexisBM25Index(index_dir=str(tmp_path / "bm25"))
    index.add_documents([make_doc("c1", "doc-1", "a policy about staffing levels")])
    results = index.search("staffing levels", top_k=5)
    assert len(results) == 1
    assert results[0]["payload"]["content"] == "a policy about staffing levels"
