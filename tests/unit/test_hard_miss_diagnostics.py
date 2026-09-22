"""
Tests for the generic hard-miss diagnostic layer (evaluation/hard_miss_diagnostics.py).

All content here is generic (memos, policies, made-up sentences), never
CUAD/legal-specific, to prove this layer has no benchmark-specific
assumptions. No Qdrant/BM25/embedding model/network is used -- dense/BM25
"evidence" objects are constructed directly, exactly as
run_hard_miss_diagnostics.py would after its own I/O.
"""
from lexis.evaluation.hard_miss_diagnostics import (
    BM25Evidence,
    BM25TextStatus,
    DenseEvidence,
    VectorStatus,
    build_hard_miss_record,
    build_hard_miss_report,
    compute_chunk_characteristics,
    compute_lexical_diagnostic,
    format_hard_miss_summary,
    tokenize,
)


# --- tokenize / compute_lexical_diagnostic: the required scenarios ---

def test_tokenize_empty_string_returns_empty_list():
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_lexical_diagnostic_empty_query():
    diag = compute_lexical_diagnostic("", "the warehouse shipped the order on time")
    assert diag.query_token_count == 0
    assert diag.query_token_coverage is None  # undefined when the query has no tokens
    assert diag.shared_token_count == 0


def test_lexical_diagnostic_empty_chunk():
    diag = compute_lexical_diagnostic("where did the shipment go", "")
    assert diag.chunk_token_count == 0
    assert diag.chunk_token_coverage is None
    assert diag.shared_token_count == 0


def test_lexical_diagnostic_both_empty():
    diag = compute_lexical_diagnostic("", "")
    assert diag.overlap_ratio_jaccard is None
    assert diag.shared_token_count == 0


def test_lexical_diagnostic_no_shared_tokens():
    diag = compute_lexical_diagnostic("bicycle repair shop hours", "the orchestra rehearsed symphony movements")
    assert diag.shared_token_count == 0
    assert diag.query_token_coverage == 0.0
    assert diag.overlap_ratio_jaccard == 0.0


def test_lexical_diagnostic_full_token_overlap():
    diag = compute_lexical_diagnostic("mountain river valley", "mountain river valley")
    assert diag.shared_token_count == diag.query_token_count
    assert diag.query_token_coverage == 1.0
    assert diag.chunk_token_coverage == 1.0
    assert diag.overlap_ratio_jaccard == 1.0


def test_lexical_diagnostic_repeated_tokens_counted_as_distinct():
    # query_token_count is the raw (non-deduped) token count -- "river"
    # repeating three times must show up as 4, not be silently collapsed --
    # but shared_token_count/coverage are computed over DISTINCT tokens, so
    # the repetition must not inflate or deflate coverage.
    diag = compute_lexical_diagnostic("river river river valley", "river valley")
    assert diag.query_token_count == 4
    assert diag.shared_token_count == 2  # {"river", "valley"}
    assert diag.query_token_coverage == 1.0


def test_lexical_diagnostic_punctuation_and_case_normalized():
    diag = compute_lexical_diagnostic("What Is The Refund Policy?!", "the refund policy is thirty days.")
    # "what", "is", "the" are English stopwords and dropped by both sides;
    # "refund"/"policy" should match regardless of case/punctuation.
    assert diag.shared_token_count >= 2
    assert diag.overlap_ratio_jaccard is not None and diag.overlap_ratio_jaccard > 0.0


def test_lexical_diagnostic_is_deterministic():
    a = compute_lexical_diagnostic("annual maintenance schedule", "the annual maintenance schedule is published quarterly")
    b = compute_lexical_diagnostic("annual maintenance schedule", "the annual maintenance schedule is published quarterly")
    assert a == b


# --- compute_chunk_characteristics ---

def test_chunk_characteristics_no_matched_answer_texts():
    chars = compute_chunk_characteristics("some chunk content here", [])
    assert chars.answer_span_char_length is None
    assert chars.answer_share_of_chunk_chars is None
    assert chars.chunk_char_length == len("some chunk content here")


def test_chunk_characteristics_uses_max_matched_answer_length():
    chunk = "The tenant shall pay rent monthly. The lease term is five years."
    chars = compute_chunk_characteristics(chunk, ["rent monthly", "The lease term is five years"])
    assert chars.answer_span_char_length == len("The lease term is five years")
    assert chars.answer_share_of_chunk_chars == chars.answer_span_char_length / len(chunk)


def test_chunk_characteristics_empty_chunk():
    chars = compute_chunk_characteristics("", ["something"])
    assert chars.chunk_char_length == 0
    assert chars.answer_share_of_chunk_chars is None  # denominator is zero, not fabricated


# --- build_hard_miss_record: missing optional dense/bm25 evidence ---

def test_build_record_with_missing_dense_and_bm25_evidence():
    record = build_hard_miss_record(
        case_id="c1", chunk_id="chunk-1", doc_id="doc-1",
        query="what is the notice period",
        chunk_content="either party may terminate with thirty days notice",
        matched_answer_texts=["thirty days notice"],
        index_integrity="indexed_normally",
        dense_evidence=None,
        bm25_evidence=None,
    )
    assert record.dense_evidence is None
    assert record.bm25_evidence is None
    assert record.lexical.shared_token_count > 0


def test_build_record_with_present_dense_and_bm25_evidence():
    dense = DenseEvidence(
        vector_status=VectorStatus.FOUND.value, query_chunk_similarity=0.42,
        population_size=100, exact_rank=57, rank_unavailable_reason=None,
    )
    bm25 = BM25Evidence(
        text_status=BM25TextStatus.FOUND.value, score=0.0,
        corpus_size=100, exact_rank=88, rank_unavailable_reason=None,
    )
    record = build_hard_miss_record(
        case_id="c1", chunk_id="chunk-1", doc_id="doc-1",
        query="what is the notice period",
        chunk_content="either party may terminate with thirty days notice",
        matched_answer_texts=["thirty days notice"],
        index_integrity="indexed_normally",
        dense_evidence=dense,
        bm25_evidence=bm25,
    )
    assert record.dense_evidence.exact_rank == 57
    assert record.bm25_evidence.score == 0.0


# --- build_hard_miss_report / format_hard_miss_summary ---

def _record(chunk_id, dense=None, bm25=None, integrity="indexed_normally", query="apple banana", chunk="apple banana cherry"):
    return build_hard_miss_record(
        case_id="c", chunk_id=chunk_id, doc_id="d", query=query, chunk_content=chunk,
        matched_answer_texts=[], index_integrity=integrity, dense_evidence=dense, bm25_evidence=bm25,
    )


def test_report_counts_valid_vectors_and_nonzero_overlap():
    records = [
        _record("a", dense=DenseEvidence(VectorStatus.FOUND.value, 0.1, 10, 3, None)),
        _record("b", dense=DenseEvidence(VectorStatus.MISSING_FROM_VECTOR_STORE.value, None, 10, None, "no vector")),
        _record("c", query="zzz yyy", chunk="qqq www"),  # no lexical overlap, no dense evidence
    ]
    report = build_hard_miss_report(records, benchmark="synthetic", run_config={})
    assert report.num_hard_misses == 3
    assert report.num_dense_evidence_available == 2
    assert report.num_with_valid_dense_vector == 1
    assert report.num_with_nonzero_lexical_overlap == 2  # "a" and "b" share apple/banana


def test_report_handles_no_evidence_at_all():
    records = [_record("a"), _record("b")]
    report = build_hard_miss_report(records, benchmark="synthetic", run_config={})
    assert report.num_dense_evidence_available == 0
    assert report.num_bm25_evidence_available == 0
    assert report.dense_similarity_distribution["count"] == 0
    assert report.dense_similarity_distribution["mean"] is None


def test_report_flags_index_integrity_anomalies():
    records = [
        _record("a", integrity="indexed_normally"),
        _record("b", integrity="missing_from_chunk_map"),
    ]
    report = build_hard_miss_report(records, benchmark="synthetic", run_config={})
    assert report.num_index_integrity_anomalies == 1


def test_format_hard_miss_summary_is_a_nonempty_string():
    records = [_record("a")]
    report = build_hard_miss_report(records, benchmark="synthetic", run_config={})
    summary = format_hard_miss_summary(report)
    assert isinstance(summary, str)
    assert "Hard misses analyzed: 1" in summary
