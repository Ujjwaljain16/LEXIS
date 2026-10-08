"""
Tests for the generic retrieval error-analysis layer (evaluation/diagnostics.py).

All retrieval-stage data here is synthetic (a lightweight FakeTrace standing
in for RetrievalEngine.retrieve_with_trace's RetrievalTrace, and plain
Candidate objects) -- no Qdrant/BM25/embedding model/network is used.
Content is generic (reports, memos, policies), never CUAD/legal-specific,
to prove this layer has no benchmark-specific assumptions.
"""
from dataclasses import dataclass
from typing import List

import pytest

from lexis.evaluation.diagnostics import (
    CaseDocumentOutcome,
    ChunkOutcome,
    build_diagnostic_report,
    classify_chunk,
    diagnose_case,
)
from lexis.evaluation.types import BenchmarkCase
from lexis.retrieval.interfaces import Candidate


def make_candidate(chunk_id: str, doc_id: str = "doc-x", content: str = "text") -> Candidate:
    return Candidate(chunk_id=chunk_id, score=1.0, source_path="test", metadata={"doc_id": doc_id, "content": content}, content=content)


@dataclass
class FakeTrace:
    """Mirrors retrieval.hybrid_retriever.RetrievalTrace's shape."""
    dense_candidates: List[Candidate]
    bm25_candidates: List[Candidate]
    fused_candidates: List[Candidate]
    final_chunks: List[dict]


def final_chunks_from(fused: List[Candidate], top_n: int) -> List[dict]:
    return [{"id": c.chunk_id, "payload": c.metadata} for c in fused[:top_n]]


def make_case(case_id="case-1", relevant_chunk_ids=("relevant-1",), relevant_doc_ids=("doc-x",)):
    return BenchmarkCase(
        case_id=case_id,
        query="a generic query",
        relevant_chunk_ids=frozenset(relevant_chunk_ids),
        relevant_doc_ids=frozenset(relevant_doc_ids),
        source_benchmark="synthetic",
    )


# --- classify_chunk: the six required scenarios ---

def test_relevant_chunk_present_in_dense_and_hits_top_k():
    fused = [make_candidate("relevant-1"), make_candidate("noise-1")]
    trace = FakeTrace(
        dense_candidates=[make_candidate("relevant-1"), make_candidate("noise-2")],
        bm25_candidates=[make_candidate("noise-3")],
        fused_candidates=fused,
        final_chunks=final_chunks_from(fused, top_n=2),
    )
    diag = classify_chunk("relevant-1", trace, top_n_rrf=2)
    assert diag.dense_rank == 1
    assert diag.bm25_rank is None
    assert diag.rrf_rank == 1
    assert diag.in_final_topk is True
    assert diag.outcome == ChunkOutcome.HIT


def test_relevant_chunk_present_only_in_bm25_and_hits_top_k():
    fused = [make_candidate("relevant-1"), make_candidate("noise-1")]
    trace = FakeTrace(
        dense_candidates=[make_candidate("noise-2")],
        bm25_candidates=[make_candidate("relevant-1"), make_candidate("noise-3")],
        fused_candidates=fused,
        final_chunks=final_chunks_from(fused, top_n=2),
    )
    diag = classify_chunk("relevant-1", trace, top_n_rrf=2)
    assert diag.dense_rank is None
    assert diag.bm25_rank == 1
    assert diag.rrf_rank == 1
    assert diag.outcome == ChunkOutcome.HIT


def test_relevant_chunk_present_in_both_paths():
    fused = [make_candidate("relevant-1")]
    trace = FakeTrace(
        dense_candidates=[make_candidate("relevant-1")],
        bm25_candidates=[make_candidate("noise-1"), make_candidate("relevant-1")],
        fused_candidates=fused,
        final_chunks=final_chunks_from(fused, top_n=1),
    )
    diag = classify_chunk("relevant-1", trace, top_n_rrf=1)
    assert diag.dense_rank == 1
    assert diag.bm25_rank == 2
    assert diag.outcome == ChunkOutcome.HIT


def test_relevant_chunk_retrieved_but_below_final_cutoff():
    # relevant-1 is fused-ranked 2nd, but the final cutoff only keeps the top 1.
    fused = [make_candidate("noise-1"), make_candidate("relevant-1")]
    trace = FakeTrace(
        dense_candidates=[make_candidate("noise-1"), make_candidate("relevant-1")],
        bm25_candidates=[],
        fused_candidates=fused,
        final_chunks=final_chunks_from(fused, top_n=1),
    )
    diag = classify_chunk("relevant-1", trace, top_n_rrf=1)
    assert diag.dense_rank == 2
    assert diag.rrf_rank == 2
    assert diag.in_final_topk is False
    assert diag.outcome == ChunkOutcome.RANKED_TOO_LOW


def test_relevant_chunk_absent_from_both_paths():
    fused = [make_candidate("noise-1")]
    trace = FakeTrace(
        dense_candidates=[make_candidate("noise-1")],
        bm25_candidates=[make_candidate("noise-2")],
        fused_candidates=fused,
        final_chunks=final_chunks_from(fused, top_n=30),
    )
    diag = classify_chunk("relevant-1", trace, top_n_rrf=30)
    assert diag.dense_rank is None
    assert diag.bm25_rank is None
    assert diag.rrf_rank is None
    assert diag.outcome == ChunkOutcome.ABSENT_FROM_BOTH_PATHS


def test_relevant_chunk_lost_in_fusion_edge_case():
    """Constructed directly (not something apply_rrf can currently produce)
    to prove the category is detected correctly if the data ever shows it,
    per the instruction not to assume it can't happen."""
    trace = FakeTrace(
        dense_candidates=[make_candidate("relevant-1")],
        bm25_candidates=[],
        fused_candidates=[make_candidate("noise-1")],  # relevant-1 absent despite being in dense_candidates
        final_chunks=final_chunks_from([make_candidate("noise-1")], top_n=30),
    )
    diag = classify_chunk("relevant-1", trace, top_n_rrf=30)
    assert diag.dense_rank == 1
    assert diag.rrf_rank is None
    assert diag.outcome == ChunkOutcome.LOST_IN_FUSION


# --- multiple relevant chunks per case, mixed outcomes ---

def test_case_with_multiple_relevant_chunks_gets_independent_outcomes():
    fused = [make_candidate("hit-chunk", doc_id="doc-x"), make_candidate("noise", doc_id="doc-y")]
    trace = FakeTrace(
        dense_candidates=[make_candidate("hit-chunk", doc_id="doc-x")],
        bm25_candidates=[],
        fused_candidates=fused,
        final_chunks=final_chunks_from(fused, top_n=2),
    )
    case = make_case(relevant_chunk_ids=("hit-chunk", "missing-chunk"), relevant_doc_ids=("doc-x",))

    diag = diagnose_case(case, trace, recall_at_k=0.5, reciprocal_rank=1.0, top_n_rrf=2)

    outcomes = {cd.chunk_id: cd.outcome for cd in diag.chunk_diagnostics}
    assert outcomes["hit-chunk"] == ChunkOutcome.HIT
    assert outcomes["missing-chunk"] == ChunkOutcome.ABSENT_FROM_BOTH_PATHS
    # Not a full hit (one chunk missed) -> document_outcome IS computed.
    assert diag.document_outcome == CaseDocumentOutcome.RIGHT_DOC_WRONG_CHUNK  # doc-x IS represented via hit-chunk


def test_case_document_outcome_absent_from_topk_when_no_chunk_of_right_doc_present():
    fused = [make_candidate("other-1", doc_id="doc-y"), make_candidate("other-2", doc_id="doc-z")]
    trace = FakeTrace(
        dense_candidates=[make_candidate("other-1", doc_id="doc-y")],
        bm25_candidates=[make_candidate("other-2", doc_id="doc-z")],
        fused_candidates=fused,
        final_chunks=final_chunks_from(fused, top_n=2),
    )
    case = make_case(relevant_chunk_ids=("missing-chunk",), relevant_doc_ids=("doc-x",))

    diag = diagnose_case(case, trace, recall_at_k=0.0, reciprocal_rank=0.0, top_n_rrf=2)

    assert diag.document_outcome == CaseDocumentOutcome.RIGHT_DOC_ABSENT_FROM_TOPK


def test_full_hit_case_has_no_document_outcome():
    fused = [make_candidate("relevant-1", doc_id="doc-x")]
    trace = FakeTrace(
        dense_candidates=[make_candidate("relevant-1", doc_id="doc-x")],
        bm25_candidates=[],
        fused_candidates=fused,
        final_chunks=final_chunks_from(fused, top_n=1),
    )
    case = make_case(relevant_chunk_ids=("relevant-1",), relevant_doc_ids=("doc-x",))

    diag = diagnose_case(case, trace, recall_at_k=1.0, reciprocal_rank=1.0, top_n_rrf=1)

    assert diag.document_outcome is None  # nothing to diagnose -- it was a hit


# --- determinism ---

def test_classification_is_deterministic():
    fused = [make_candidate("noise-1"), make_candidate("relevant-1")]
    trace = FakeTrace(
        dense_candidates=[make_candidate("noise-1"), make_candidate("relevant-1")],
        bm25_candidates=[],
        fused_candidates=fused,
        final_chunks=final_chunks_from(fused, top_n=1),
    )
    first = classify_chunk("relevant-1", trace, top_n_rrf=1)
    second = classify_chunk("relevant-1", trace, top_n_rrf=1)
    assert first == second


# --- aggregate report: mutually exclusive, no double counting ---

def test_aggregate_counts_sum_to_total_relevant_chunks_diagnosed():
    fused = [make_candidate("hit-1", doc_id="doc-a")]
    trace_hit = FakeTrace(
        dense_candidates=[make_candidate("hit-1", doc_id="doc-a")],
        bm25_candidates=[],
        fused_candidates=fused,
        final_chunks=final_chunks_from(fused, top_n=1),
    )
    trace_miss = FakeTrace(
        dense_candidates=[],
        bm25_candidates=[],
        fused_candidates=[],
        final_chunks=[],
    )

    case_a = diagnose_case(make_case("case-a", ("hit-1",), ("doc-a",)), trace_hit, 1.0, 1.0, top_n_rrf=1)
    case_b = diagnose_case(make_case("case-b", ("missing-1", "missing-2"), ("doc-b",)), trace_miss, 0.0, 0.0, top_n_rrf=1)

    report = build_diagnostic_report([case_a, case_b], benchmark="synthetic", run_config={"top_k": 1})

    total_relevant_chunks_diagnosed = 1 + 2  # case_a has 1, case_b has 2
    assert sum(report.aggregate_chunk_outcomes.values()) == total_relevant_chunks_diagnosed
    assert report.aggregate_chunk_outcomes[ChunkOutcome.HIT.value] == 1
    assert report.aggregate_chunk_outcomes[ChunkOutcome.ABSENT_FROM_BOTH_PATHS.value] == 2
    # Document outcomes are a SEPARATE dimension -- only case_b contributes one (case_a was a full hit).
    assert sum(report.aggregate_document_outcomes.values()) == 1
    assert report.aggregate_document_outcomes[CaseDocumentOutcome.RIGHT_DOC_ABSENT_FROM_TOPK.value] == 1
    assert report.num_cases == 2


def test_report_carries_category_definitions_for_every_outcome():
    report = build_diagnostic_report([], benchmark="synthetic", run_config={})
    for outcome in list(ChunkOutcome) + list(CaseDocumentOutcome):
        assert outcome.value in report.category_definitions


# --- does not touch the existing metric definitions ---

def test_diagnostics_module_does_not_reimplement_recall_or_mrr():
    """diagnose_case must take recall/MRR as inputs, not compute them --
    proves this module cannot silently define them differently from
    evaluation/metrics.py (used unmodified by the harness)."""
    import inspect
    from lexis.evaluation import diagnostics as diagnostics_module

    source = inspect.getsource(diagnostics_module)
    assert "def recall_at_k" not in source
    assert "def reciprocal_rank" not in source
