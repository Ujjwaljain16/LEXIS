"""
Generic, document-agnostic retrieval error-analysis layer.

Operates purely on BenchmarkCase (query + relevant chunk/doc ids -- see
evaluation/types.py) and RetrievalTrace (per-stage candidate lists from
RetrievalEngine.retrieve_with_trace -- see retrieval/hybrid_retriever.py).
It has no knowledge of CUAD, legal documents, clauses, or any other
benchmark-specific concept, and does not compute Recall@k/MRR itself --
those come from the existing evaluation/metrics.py, unchanged, and are
passed in by the caller so this module cannot silently diverge from the
baseline's own metric definitions.

Outcome categories are derived strictly from stage evidence (dense rank,
BM25 rank, fused rank, final cutoff membership) -- never invented or
assumed to occur.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from lexis.evaluation.types import BenchmarkCase


class ChunkOutcome(str, Enum):
    """Mutually exclusive, exhaustive classification for one (case,
    relevant_chunk_id) pair. Every relevant chunk in every diagnosed case
    gets exactly one of these -- aggregate counts sum to the total number
    of (case, relevant_chunk) pairs diagnosed, so they cannot be
    double-counted or misleading."""
    HIT = "hit"
    ABSENT_FROM_BOTH_PATHS = "absent_from_both_paths"
    LOST_IN_FUSION = "lost_in_fusion"
    RANKED_TOO_LOW = "ranked_too_low"


class CaseDocumentOutcome(str, Enum):
    """A second, independent diagnostic dimension at case granularity,
    computed ONLY for cases with at least one missed relevant chunk (a full
    hit has nothing to diagnose). This OVERLAPS with ChunkOutcome by design
    -- a case can have RANKED_TOO_LOW chunks and still be
    RIGHT_DOC_WRONG_CHUNK if a different chunk from the same document made
    the final top-k. Report the two dimensions side by side; never sum them
    together into one total."""
    RIGHT_DOC_WRONG_CHUNK = "right_doc_wrong_chunk"
    RIGHT_DOC_ABSENT_FROM_TOPK = "right_doc_absent_from_topk"


CATEGORY_DEFINITIONS: Dict[str, str] = {
    ChunkOutcome.HIT.value:
        "The relevant chunk appeared in the final top-k results.",
    ChunkOutcome.ABSENT_FROM_BOTH_PATHS.value:
        "The relevant chunk was not returned by either the dense or BM25 retrieval path "
        "at the configured per-path depth.",
    ChunkOutcome.LOST_IN_FUSION.value:
        "The relevant chunk was returned by at least one retrieval path but has no entry "
        "in the fused RRF ranking. Given the current fusion implementation (which unions "
        "every candidate from every path), this should not occur; it is reported, not "
        "assumed impossible, in case a data-integrity edge case (e.g. an empty chunk_id) "
        "produces it.",
    ChunkOutcome.RANKED_TOO_LOW.value:
        "The relevant chunk was returned by a retrieval path and is present in the fused "
        "RRF ranking, but its fused rank exceeds the final top-k cutoff.",
    CaseDocumentOutcome.RIGHT_DOC_WRONG_CHUNK.value:
        "For a case that missed at least one relevant chunk: at least one chunk from the "
        "same relevant document(s) still appears in the final top-k -- a passage-selection "
        "issue, not a document-relevance issue. Computed only for cases with a miss; "
        "overlaps with, and is independent of, the per-chunk outcome counts.",
    CaseDocumentOutcome.RIGHT_DOC_ABSENT_FROM_TOPK.value:
        "For a case that missed at least one relevant chunk: no chunk from the same "
        "relevant document(s) appears anywhere in the final top-k -- the correct document "
        "itself was not surfaced. Computed only for cases with a miss; overlaps with, and "
        "is independent of, the per-chunk outcome counts.",
}


def rank_of(chunk_id: str, ordered_candidates: List[Any]) -> Optional[int]:
    """1-indexed rank of chunk_id within an already-ordered candidate list
    (Candidate objects, which expose .chunk_id), or None if absent."""
    for i, candidate in enumerate(ordered_candidates):
        if candidate.chunk_id == chunk_id:
            return i + 1
    return None


@dataclass
class ChunkDiagnostic:
    chunk_id: str
    dense_rank: Optional[int]
    bm25_rank: Optional[int]
    rrf_rank: Optional[int]
    in_final_topk: bool
    outcome: ChunkOutcome


@dataclass
class CaseDiagnostic:
    case_id: str
    query: str
    relevant_chunk_ids: List[str]
    recall_at_k: float
    reciprocal_rank: float
    final_topk_doc_ids: List[str]
    chunk_diagnostics: List[ChunkDiagnostic]
    document_outcome: Optional[CaseDocumentOutcome]


@dataclass
class DiagnosticReport:
    benchmark: str
    run_config: dict
    num_cases: int
    aggregate_chunk_outcomes: Dict[str, int]
    aggregate_document_outcomes: Dict[str, int]
    category_definitions: Dict[str, str]
    per_case: List[CaseDiagnostic]


def classify_chunk(chunk_id: str, trace, top_n_rrf: int) -> ChunkDiagnostic:
    """trace must expose .dense_candidates, .bm25_candidates,
    .fused_candidates as ordered lists of objects with a .chunk_id
    attribute (see retrieval/hybrid_retriever.py::RetrievalTrace)."""
    dense_rank = rank_of(chunk_id, trace.dense_candidates)
    bm25_rank = rank_of(chunk_id, trace.bm25_candidates)
    rrf_rank = rank_of(chunk_id, trace.fused_candidates)
    in_final_topk = rrf_rank is not None and rrf_rank <= top_n_rrf

    if in_final_topk:
        outcome = ChunkOutcome.HIT
    elif dense_rank is None and bm25_rank is None:
        outcome = ChunkOutcome.ABSENT_FROM_BOTH_PATHS
    elif rrf_rank is None:
        outcome = ChunkOutcome.LOST_IN_FUSION
    else:
        outcome = ChunkOutcome.RANKED_TOO_LOW

    return ChunkDiagnostic(
        chunk_id=chunk_id,
        dense_rank=dense_rank,
        bm25_rank=bm25_rank,
        rrf_rank=rrf_rank,
        in_final_topk=in_final_topk,
        outcome=outcome,
    )


def _document_outcome(case: BenchmarkCase, final_topk_doc_ids: List[str]) -> Optional[CaseDocumentOutcome]:
    if not case.relevant_doc_ids:
        return None
    right_doc_present = any(doc_id in case.relevant_doc_ids for doc_id in final_topk_doc_ids)
    return (
        CaseDocumentOutcome.RIGHT_DOC_WRONG_CHUNK
        if right_doc_present
        else CaseDocumentOutcome.RIGHT_DOC_ABSENT_FROM_TOPK
    )


def diagnose_case(
    case: BenchmarkCase,
    trace,
    recall_at_k: float,
    reciprocal_rank: float,
    top_n_rrf: int,
) -> CaseDiagnostic:
    """
    recall_at_k/reciprocal_rank must be computed by the caller using
    evaluation/metrics.py (the same functions the harness uses) -- this
    module never computes them itself, so it cannot define them
    differently from the baseline.
    """
    chunk_diagnostics = [
        classify_chunk(chunk_id, trace, top_n_rrf) for chunk_id in case.relevant_chunk_ids
    ]
    final_topk_doc_ids = [c.get("payload", {}).get("doc_id") for c in trace.final_chunks]

    is_full_hit = bool(chunk_diagnostics) and all(cd.outcome == ChunkOutcome.HIT for cd in chunk_diagnostics)
    document_outcome = None if is_full_hit else _document_outcome(case, final_topk_doc_ids)

    return CaseDiagnostic(
        case_id=case.case_id,
        query=case.query,
        relevant_chunk_ids=list(case.relevant_chunk_ids),
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
        final_topk_doc_ids=final_topk_doc_ids,
        chunk_diagnostics=chunk_diagnostics,
        document_outcome=document_outcome,
    )


def build_diagnostic_report(
    case_diagnostics: List[CaseDiagnostic],
    benchmark: str,
    run_config: dict,
) -> DiagnosticReport:
    chunk_counts: Dict[str, int] = {outcome.value: 0 for outcome in ChunkOutcome}
    doc_counts: Dict[str, int] = {outcome.value: 0 for outcome in CaseDocumentOutcome}

    for case in case_diagnostics:
        for chunk_diag in case.chunk_diagnostics:
            chunk_counts[chunk_diag.outcome.value] += 1
        if case.document_outcome is not None:
            doc_counts[case.document_outcome.value] += 1

    return DiagnosticReport(
        benchmark=benchmark,
        run_config=run_config,
        num_cases=len(case_diagnostics),
        aggregate_chunk_outcomes=chunk_counts,
        aggregate_document_outcomes=doc_counts,
        category_definitions=dict(CATEGORY_DEFINITIONS),
        per_case=case_diagnostics,
    )


def format_summary(report: DiagnosticReport) -> str:
    """Concise human-readable summary, safe to print to logs/console."""
    lines = [
        f"=== Retrieval Diagnostic Summary ({report.benchmark}) ===",
        f"Cases diagnosed: {report.num_cases}",
        "",
        "Per-relevant-chunk outcomes (mutually exclusive, sums to total relevant chunks diagnosed):",
    ]
    total_chunks = sum(report.aggregate_chunk_outcomes.values())
    for outcome in ChunkOutcome:
        count = report.aggregate_chunk_outcomes.get(outcome.value, 0)
        lines.append(f"  {outcome.value:28s} {count:4d} / {total_chunks}")
    lines.append("")
    lines.append("Case-level document outcomes (only for cases with a miss; overlaps with the above, reported separately):")
    for outcome in CaseDocumentOutcome:
        count = report.aggregate_document_outcomes.get(outcome.value, 0)
        lines.append(f"  {outcome.value:28s} {count:4d}")
    return "\n".join(lines)
