"""
Tests for the generic ground-truth mapping utilities
(evaluation/dataset/mapping.py) used by benchmark adapters. Content here is
deliberately generic (a policy/report/memo), not contract/clause specific,
since these utilities must work for any future benchmark.
"""
from lexis.evaluation.dataset.mapping import (
    build_answer_derived_query,
    deterministic_document_id,
    map_answer_texts_to_chunk_ids,
    normalize_whitespace,
    texts_matching_chunk,
)
from lexis.indexing.schema import Chunk, ChunkMetadata

GENERIC_META = ChunkMetadata(source_file="generic.txt", page_num=1, document_type="unknown")


def make_chunk(doc_id: str, split_idx: int, raw_content: str) -> Chunk:
    return Chunk.create(doc_id=doc_id, split_idx=split_idx, raw_content=raw_content, metadata=GENERIC_META)


# --- deterministic_document_id ---

def test_same_source_identifier_always_yields_the_same_doc_id():
    a = deterministic_document_id("Quarterly Safety Report 2024")
    b = deterministic_document_id("Quarterly Safety Report 2024")
    assert a == b


def test_different_source_identifiers_yield_different_doc_ids():
    a = deterministic_document_id("Quarterly Safety Report 2024")
    b = deterministic_document_id("Employee Handbook v3")
    assert a != b


def test_no_hardcoded_lookup_table_arbitrary_titles_all_resolve():
    """Any arbitrary title must resolve deterministically -- there is no
    finite hand-maintained mapping to exhaust."""
    titles = [f"Arbitrary Document Title {i}" for i in range(25)]
    ids = {deterministic_document_id(t) for t in titles}
    assert len(ids) == len(titles)  # all distinct, none collided


def test_prefix_is_applied_and_configurable():
    assert deterministic_document_id("X", prefix="cuad").startswith("cuad-")
    assert deterministic_document_id("X", prefix="other-benchmark").startswith("other-benchmark-")


# --- map_answer_texts_to_chunk_ids ---

def test_exact_verbatim_match_is_found():
    chunk = make_chunk("doc-1", 0, "The routing procedure requires manager approval within 48 hours.")
    result = map_answer_texts_to_chunk_ids(["manager approval within 48 hours"], [chunk])
    assert result == {chunk.chunk_id}


def test_whitespace_differences_still_match():
    """Chunking re-joins sentences with a single space (see
    ingestion/chunker.py), so a chunk's raw_content will not preserve the
    source document's original line breaks/spacing even though the words
    are unchanged. The mapping must normalize whitespace on both sides."""
    chunk = make_chunk("doc-1", 0, "Staffing levels:\n   must not fall   below\ntwo people per shift.")
    result = map_answer_texts_to_chunk_ids(["must not fall below\ntwo people per shift"], [chunk])
    assert result == {chunk.chunk_id}


def test_no_match_returns_empty_set_not_a_guess():
    chunk = make_chunk("doc-1", 0, "This passage is about something entirely unrelated.")
    result = map_answer_texts_to_chunk_ids(["a phrase that never appears anywhere"], [chunk])
    assert result == set()


def test_multiple_answer_texts_match_across_different_chunks():
    chunk_a = make_chunk("doc-1", 0, "First topic: renewal timelines are thirty days.")
    chunk_b = make_chunk("doc-1", 1, "Second topic: staffing must be reviewed quarterly.")
    result = map_answer_texts_to_chunk_ids(
        ["renewal timelines are thirty days", "staffing must be reviewed quarterly"],
        [chunk_a, chunk_b],
    )
    assert result == {chunk_a.chunk_id, chunk_b.chunk_id}


def test_answer_text_present_in_one_chunk_does_not_falsely_match_another():
    chunk_a = make_chunk("doc-1", 0, "This chunk contains the real answer text about renewals.")
    chunk_b = make_chunk("doc-1", 1, "This chunk is completely unrelated content.")
    result = map_answer_texts_to_chunk_ids(["real answer text about renewals"], [chunk_a, chunk_b])
    assert result == {chunk_a.chunk_id}
    assert chunk_b.chunk_id not in result


def test_empty_answer_text_is_ignored_not_matched_to_everything():
    chunk = make_chunk("doc-1", 0, "Some content.")
    result = map_answer_texts_to_chunk_ids([""], [chunk])
    assert result == set()


# --- normalize_whitespace / texts_matching_chunk / build_answer_derived_query
# (Step 11: offline oracle query representation) ---

def test_normalize_whitespace_collapses_runs_like_the_internal_rule():
    assert normalize_whitespace("a   b\n\nc") == "a b c"


def test_texts_matching_chunk_single_match():
    # The underlying rule is whitespace-normalized substring containment,
    # not case-insensitive -- use a matching-case candidate.
    chunk = make_chunk("doc-1", 0, "The lease term is five years and renews annually.")
    result = texts_matching_chunk(["The lease term is five years and renews annually"], chunk)
    assert result == ["The lease term is five years and renews annually"]


def test_texts_matching_chunk_no_match_returns_empty_list():
    chunk = make_chunk("doc-1", 0, "Unrelated content entirely.")
    assert texts_matching_chunk(["something that never appears"], chunk) == []


def test_texts_matching_chunk_multiple_answer_spans_only_matching_ones_returned():
    chunk = make_chunk("doc-1", 0, "Rent is due monthly. Pets are not allowed on the premises.")
    result = texts_matching_chunk(["Rent is due monthly", "something absent", "Pets are not allowed on the premises"], chunk)
    assert result == ["Rent is due monthly", "Pets are not allowed on the premises"]


def test_texts_matching_chunk_preserves_input_order():
    chunk = make_chunk("doc-1", 0, "Alpha clause. Beta clause. Gamma clause.")
    result = texts_matching_chunk(["Gamma clause", "Alpha clause"], chunk)
    assert result == ["Gamma clause", "Alpha clause"]  # order of candidate_texts, not of appearance in chunk


def test_build_answer_derived_query_single_match_returned_as_is():
    chunk = make_chunk("doc-1", 0, "Either party may terminate with thirty days notice.")
    query = build_answer_derived_query(["Either party may terminate with thirty days notice"], chunk)
    assert query == "Either party may terminate with thirty days notice"


def test_build_answer_derived_query_multiple_matches_joined_deterministically():
    chunk = make_chunk("doc-1", 0, "Rent is due monthly. Pets are not allowed.")
    query = build_answer_derived_query(["Rent is due monthly", "Pets are not allowed"], chunk)
    assert query == "Rent is due monthly Pets are not allowed"


def test_build_answer_derived_query_no_match_returns_none():
    chunk = make_chunk("doc-1", 0, "Something else entirely.")
    assert build_answer_derived_query(["never appears"], chunk) is None


def test_build_answer_derived_query_empty_candidate_list_returns_none():
    chunk = make_chunk("doc-1", 0, "Some content.")
    assert build_answer_derived_query([], chunk) is None


def test_build_answer_derived_query_empty_string_candidate_is_ignored():
    chunk = make_chunk("doc-1", 0, "Some content here.")
    assert build_answer_derived_query([""], chunk) is None
