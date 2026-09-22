"""
Tests for the generic retrieval-depth sensitivity module
(evaluation/depth_sensitivity.py). All retrieval-stage data is synthetic
(plain Candidate lists standing in for RetrievalTrace) -- no Qdrant/BM25/
embedding model/network is used. Content is generic (reports, memos),
never CUAD/legal-specific.
"""
import inspect
from dataclasses import dataclass
from typing import List

import pytest

from lexis.evaluation.depth_sensitivity import (
    IndexIntegrityStatus,
    build_chunk_profile,
    build_rank_distribution,
    check_index_integrity,
    compute_depth_coverage,
    format_depth_table,
    set_rrf_hit_depths,
)
from lexis.evaluation.types import BenchmarkCase
from lexis.retrieval.interfaces import Candidate


def make_candidate(chunk_id: str, doc_id: str = "doc-x") -> Candidate:
    return Candidate(chunk_id=chunk_id, score=1.0, source_path="test", metadata={"doc_id": doc_id}, content="text")


class FakeChunkWithId:
    """A minimal stand-in exposing exactly what check_index_integrity reads."""
    def __init__(self, chunk_id, raw_content):
        self.chunk_id = chunk_id
        self.raw_content = raw_content


@dataclass
class FakeTrace:
    dense_candidates: List[Candidate]
    bm25_candidates: List[Candidate]
    fused_candidates: List[Candidate]
    final_chunks: List[dict]


def make_case(case_id="case-1", relevant_chunk_ids=("relevant-1",), relevant_doc_ids=("doc-x",)):
    return BenchmarkCase(
        case_id=case_id,
        query="a generic query",
        relevant_chunk_ids=frozenset(relevant_chunk_ids),
        relevant_doc_ids=frozenset(relevant_doc_ids),
        source_benchmark="synthetic",
    )


DEPTHS = [10, 30, 60, 100, 200]


def ranked_dense(rank_of_relevant, target_chunk_id="relevant-1", total=250):
    """Builds a candidate list where target_chunk_id sits at 1-indexed rank_of_relevant."""
    cands = [make_candidate(f"noise-{i}") for i in range(total)]
    cands[rank_of_relevant - 1] = make_candidate(target_chunk_id)
    return cands


# --- index integrity: existing ground-truth chunk map only, no new matching ---

def test_index_integrity_normal_chunk():
    chunks_by_doc_id = {"doc-x": [FakeChunkWithId("relevant-1", "real content here")]}
    assert check_index_integrity("relevant-1", "doc-x", chunks_by_doc_id) == IndexIntegrityStatus.INDEXED_NORMALLY


def test_index_integrity_missing_chunk():
    chunks_by_doc_id = {"doc-x": [FakeChunkWithId("other-chunk", "content")]}
    assert check_index_integrity("relevant-1", "doc-x", chunks_by_doc_id) == IndexIntegrityStatus.MISSING_FROM_CHUNK_MAP


def test_index_integrity_missing_doc():
    assert check_index_integrity("relevant-1", "doc-not-present", {}) == IndexIntegrityStatus.MISSING_FROM_CHUNK_MAP


def test_index_integrity_empty_content():
    chunks_by_doc_id = {"doc-x": [FakeChunkWithId("relevant-1", "   ")]}
    assert check_index_integrity("relevant-1", "doc-x", chunks_by_doc_id) == IndexIntegrityStatus.EMPTY_CONTENT


def test_index_integrity_no_doc_id():
    assert check_index_integrity("relevant-1", None, {"doc-x": []}) == IndexIntegrityStatus.MISSING_FROM_CHUNK_MAP


# --- chunk profile: dense/bm25/both/absent, and rank-between-two-depths ---

def test_relevant_item_in_dense_but_not_bm25():
    trace = FakeTrace(
        dense_candidates=ranked_dense(5),
        bm25_candidates=[make_candidate("noise-1")],
        fused_candidates=[],
        final_chunks=[],
    )
    case = make_case()
    profile = build_chunk_profile(case, "relevant-1", trace, DEPTHS, {"doc-x": [FakeChunkWithId("relevant-1", "text")]})
    assert profile.dense_rank_at_max_depth == 5
    assert profile.bm25_rank_at_max_depth is None
    assert profile.first_depth_dense_hit == 10  # smallest configured depth >= 5
    assert profile.first_depth_bm25_hit is None


def test_relevant_item_in_bm25_but_not_dense():
    trace = FakeTrace(
        dense_candidates=[make_candidate("noise-1")],
        bm25_candidates=ranked_dense(45)[:50],  # relevant-1 at rank 45 within a shorter list
        fused_candidates=[],
        final_chunks=[],
    )
    case = make_case()
    profile = build_chunk_profile(case, "relevant-1", trace, DEPTHS, {"doc-x": [FakeChunkWithId("relevant-1", "text")]})
    assert profile.dense_rank_at_max_depth is None
    assert profile.bm25_rank_at_max_depth == 45
    assert profile.first_depth_bm25_hit == 60  # smallest configured depth >= 45 (between 30 and 60)


def test_relevant_item_in_both_paths():
    trace = FakeTrace(
        dense_candidates=ranked_dense(3),
        bm25_candidates=ranked_dense(7),
        fused_candidates=[],
        final_chunks=[],
    )
    case = make_case()
    profile = build_chunk_profile(case, "relevant-1", trace, DEPTHS, {"doc-x": [FakeChunkWithId("relevant-1", "text")]})
    assert profile.dense_rank_at_max_depth == 3
    assert profile.bm25_rank_at_max_depth == 7
    assert profile.first_depth_dense_hit == 10
    assert profile.first_depth_bm25_hit == 10


def test_relevant_item_absent_at_largest_diagnostic_depth():
    trace = FakeTrace(
        dense_candidates=[make_candidate(f"noise-{i}") for i in range(250)],  # relevant-1 never present
        bm25_candidates=[make_candidate(f"noise-{i}") for i in range(250)],
        fused_candidates=[],
        final_chunks=[],
    )
    case = make_case()
    profile = build_chunk_profile(case, "relevant-1", trace, DEPTHS, {"doc-x": [FakeChunkWithId("relevant-1", "text")]})
    assert profile.dense_rank_at_max_depth is None
    assert profile.bm25_rank_at_max_depth is None
    assert profile.first_depth_dense_hit is None
    assert profile.first_depth_bm25_hit is None


def test_relevant_item_between_two_depth_values_exact_boundary():
    """Rank 31 must land in the 60 bucket, not the 30 bucket -- an
    off-by-one here would silently misreport coverage."""
    trace = FakeTrace(dense_candidates=ranked_dense(31), bm25_candidates=[], fused_candidates=[], final_chunks=[])
    case = make_case()
    profile = build_chunk_profile(case, "relevant-1", trace, DEPTHS, {"doc-x": [FakeChunkWithId("relevant-1", "text")]})
    assert profile.first_depth_dense_hit == 60

    trace_at_30 = FakeTrace(dense_candidates=ranked_dense(30), bm25_candidates=[], fused_candidates=[], final_chunks=[])
    profile_at_30 = build_chunk_profile(case, "relevant-1", trace_at_30, DEPTHS, {"doc-x": [FakeChunkWithId("relevant-1", "text")]})
    assert profile_at_30.first_depth_dense_hit == 30


# --- determinism ---

def test_profile_building_is_deterministic():
    trace = FakeTrace(dense_candidates=ranked_dense(15), bm25_candidates=[], fused_candidates=[], final_chunks=[])
    case = make_case()
    chunks_by_doc_id = {"doc-x": [FakeChunkWithId("relevant-1", "text")]}
    first = build_chunk_profile(case, "relevant-1", trace, DEPTHS, chunks_by_doc_id)
    second = build_chunk_profile(case, "relevant-1", trace, DEPTHS, chunks_by_doc_id)
    assert first == second


# --- rrf hit depths and coverage math ---

def test_set_rrf_hit_depths_picks_smallest_covering_depth():
    profile = build_chunk_profile(
        make_case(), "relevant-1",
        FakeTrace([], [], [], []), DEPTHS, {"doc-x": [FakeChunkWithId("relevant-1", "text")]},
    )
    set_rrf_hit_depths([profile], {10: set(), 30: {("case-1", "relevant-1")}, 60: {("case-1", "relevant-1")}, 100: {("case-1", "relevant-1")}, 200: {("case-1", "relevant-1")}})
    assert profile.first_depth_rrf_hit == 30


def test_shared_chunk_id_across_two_cases_tracked_independently():
    """Regression: the same chunk_id can be a relevant target for two
    different cases (e.g. two questions about the same document whose
    answers fall in the same passage). A hit for one case's query must not
    be misreported as a hit for a different case that happens to share the
    chunk_id -- each case's own query can retrieve it independently."""
    shared_chunk_id = "shared-chunk"
    chunks_by_doc_id = {"doc-x": [FakeChunkWithId(shared_chunk_id, "text")]}

    case_hit = make_case(case_id="case-hit", relevant_chunk_ids=(shared_chunk_id,))
    case_miss = make_case(case_id="case-miss", relevant_chunk_ids=(shared_chunk_id,))

    profile_hit = build_chunk_profile(case_hit, shared_chunk_id, FakeTrace([], [], [], []), DEPTHS, chunks_by_doc_id)
    profile_miss = build_chunk_profile(case_miss, shared_chunk_id, FakeTrace([], [], [], []), DEPTHS, chunks_by_doc_id)

    # Only case-hit's query actually retrieved the chunk at depth 30; case-miss's did not.
    rrf_hits_by_depth = {30: {("case-hit", shared_chunk_id)}}
    set_rrf_hit_depths([profile_hit, profile_miss], rrf_hits_by_depth)

    assert profile_hit.first_depth_rrf_hit == 30
    assert profile_miss.first_depth_rrf_hit is None  # must NOT inherit case-hit's hit

    coverage = compute_depth_coverage(30, [profile_hit, profile_miss], rrf_hits_by_depth[30], per_case_metrics_at_depth=[(1.0, 1.0), (0.0, 0.0)])
    assert coverage.rrf_coverage == pytest.approx(0.5)  # exactly one of the two pairs hit, not both


def test_compute_depth_coverage_multiple_depths_and_mixed_outcomes():
    chunks_by_doc_id = {"doc-x": [FakeChunkWithId("a", "t"), FakeChunkWithId("b", "t"), FakeChunkWithId("c", "t"), FakeChunkWithId("d", "t")]}
    case = make_case()

    profile_a = build_chunk_profile(case, "a", FakeTrace(ranked_dense(3, "a"), [], [], []), DEPTHS, chunks_by_doc_id)   # dense@3
    profile_b = build_chunk_profile(case, "b", FakeTrace([], ranked_dense(45, "b", total=50), [], []), DEPTHS, chunks_by_doc_id)  # bm25@45
    profile_c = build_chunk_profile(case, "c", FakeTrace(ranked_dense(5, "c"), ranked_dense(5, "c"), [], []), DEPTHS, chunks_by_doc_id)  # both@5
    profile_d = build_chunk_profile(case, "d", FakeTrace([], [], [], []), DEPTHS, chunks_by_doc_id)  # absent from both

    profiles = [profile_a, profile_b, profile_c, profile_d]

    # At depth 10: a (dense@3) and c (both@5) covered; b (bm25@45) and d not.
    cov_10 = compute_depth_coverage(10, profiles, rrf_hit_pairs_at_depth=set(), per_case_metrics_at_depth=[(0.5, 0.5)])
    assert cov_10.num_relevant_chunks == 4
    assert cov_10.dense_coverage == pytest.approx(2 / 4)   # a, c
    assert cov_10.bm25_coverage == pytest.approx(1 / 4)    # c
    assert cov_10.either_coverage == pytest.approx(2 / 4)  # a, c
    assert cov_10.both_coverage == pytest.approx(1 / 4)    # c
    assert cov_10.neither_coverage == pytest.approx(2 / 4)  # b, d

    # At depth 60: b's rank-45 is now covered too.
    cov_60 = compute_depth_coverage(60, profiles, rrf_hit_pairs_at_depth=set(), per_case_metrics_at_depth=[(1.0, 1.0)])
    assert cov_60.dense_coverage == pytest.approx(2 / 4)   # a, c (b/d have no dense rank)
    assert cov_60.bm25_coverage == pytest.approx(2 / 4)    # b, c
    assert cov_60.either_coverage == pytest.approx(3 / 4)  # a, b, c
    assert cov_60.neither_coverage == pytest.approx(1 / 4)  # d only


def test_recall_and_mrr_at_depth_come_from_the_metrics_module_unmodified():
    """Proves this module does not reimplement Recall@k or MRR."""
    import lexis.evaluation.depth_sensitivity as ds_module
    source = inspect.getsource(ds_module)
    assert "def recall_at_k" not in source
    assert "def reciprocal_rank" not in source

    coverage = compute_depth_coverage(30, [], rrf_hit_pairs_at_depth=set(), per_case_metrics_at_depth=[(1.0, 1.0), (0.0, 0.0)])
    assert coverage.mean_recall_at_depth == pytest.approx(0.5)
    assert coverage.mean_reciprocal_rank_at_depth == pytest.approx(0.5)


# --- rank distribution buckets derive from the configured depths, not a hardcoded scheme ---

def test_rank_distribution_buckets_follow_configured_depths():
    chunks_by_doc_id = {"doc-x": [FakeChunkWithId("relevant-1", "t")]}
    case = make_case()
    profile_5 = build_chunk_profile(case, "relevant-1", FakeTrace(ranked_dense(5), [], [], []), [10, 30], chunks_by_doc_id)
    profile_20 = build_chunk_profile(case, "relevant-1", FakeTrace(ranked_dense(20), [], [], []), [10, 30], chunks_by_doc_id)
    profile_none = build_chunk_profile(case, "relevant-1", FakeTrace([], [], [], []), [10, 30], chunks_by_doc_id)

    dist = build_rank_distribution([profile_5, profile_20, profile_none], [10, 30])
    assert dist["dense"]["<= 10"] == 1
    assert dist["dense"]["11-30"] == 1
    assert dist["dense"][">30 or absent"] == 1


# --- no mutation of production retrieval configuration ---

def test_module_never_imports_or_touches_settings():
    """depth_sensitivity.py must be pure measurement over already-computed
    traces -- it has no business reading or writing production Settings."""
    import lexis.evaluation.depth_sensitivity as ds_module
    source = inspect.getsource(ds_module)
    assert "from lexis.config import settings" not in source
    assert "import settings" not in source


def test_format_depth_table_contains_every_depth_row():
    coverages = [
        compute_depth_coverage(d, [], rrf_hit_pairs_at_depth=set(), per_case_metrics_at_depth=[])
        for d in DEPTHS
    ]
    table = format_depth_table(coverages)
    for d in DEPTHS:
        assert f"| {d} |" in table
