"""
Generic, document-agnostic retrieval-depth sensitivity analysis.

Answers a narrow question left open by Step 8's diagnostics: for relevant
chunks absent from both retrieval paths at the CURRENT production per-path
depth, are they genuinely unretrievable, or merely ranked beyond that
depth? This module measures coverage/rank/metric behavior across a
configurable set of retrieval depths. It does not infer a cause, tune
anything, or change production retrieval behavior -- measurement only.

Operates purely on BenchmarkCase (evaluation/types.py), RetrievalTrace
(retrieval/hybrid_retriever.py), and the chunks_by_doc_id map already built
to derive ground truth (evaluation/dataset/cuad_loader.py's caller
constructs this the same way regardless of benchmark) -- no CUAD-specific,
legal-specific, or otherwise benchmark-specific concept appears here.

Coverage definitions (pooled across every relevant chunk diagnosed, NOT
case-averaged -- see recall_at_depth/mrr_at_depth below for the
case-averaged numbers, and do not mix the two):
    dense_coverage   = fraction of relevant chunks with dense_rank <= depth
    bm25_coverage    = fraction of relevant chunks with bm25_rank <= depth
    either_coverage  = fraction with dense_rank <= depth OR bm25_rank <= depth
    both_coverage    = fraction with dense_rank <= depth AND bm25_rank <= depth
    neither_coverage = 1 - either_coverage
    rrf_coverage     = fraction present in the fused ranking's own top-`depth`
                       when retrieval is run with that depth as BOTH the
                       per-path depth and the fusion cutoff (i.e. what
                       Recall@depth's underlying "did we retrieve it at all"
                       question reduces to at the chunk level)

recall_at_depth / mrr_at_depth are the existing, case-averaged
evaluation/metrics.py functions, unmodified, applied with k=depth -- a
different aggregation from the pooled coverage figures above.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from lexis.evaluation.diagnostics import rank_of
from lexis.evaluation.types import BenchmarkCase


class IndexIntegrityStatus(str, Enum):
    """Whether a relevant chunk's own indexed record looks structurally
    sound, independent of whether retrieval ever surfaces it. Derived only
    from the same chunk map already used to build ground truth (see
    dataset/mapping.py::map_answer_texts_to_chunk_ids) -- no new matching
    logic, no fuzzy/semantic comparison."""
    INDEXED_NORMALLY = "indexed_normally"
    MISSING_FROM_CHUNK_MAP = "missing_from_chunk_map"
    EMPTY_CONTENT = "empty_content"


def check_index_integrity(chunk_id: str, doc_id: Optional[str], chunks_by_doc_id: Dict[str, list]) -> IndexIntegrityStatus:
    if doc_id is None:
        return IndexIntegrityStatus.MISSING_FROM_CHUNK_MAP
    for chunk in chunks_by_doc_id.get(doc_id, []):
        if chunk.chunk_id == chunk_id:
            if not chunk.raw_content or not chunk.raw_content.strip():
                return IndexIntegrityStatus.EMPTY_CONTENT
            return IndexIntegrityStatus.INDEXED_NORMALLY
    return IndexIntegrityStatus.MISSING_FROM_CHUNK_MAP


@dataclass
class ChunkDepthProfile:
    case_id: str
    chunk_id: str
    doc_id: Optional[str]
    dense_rank_at_max_depth: Optional[int]
    bm25_rank_at_max_depth: Optional[int]
    first_depth_dense_hit: Optional[int]
    first_depth_bm25_hit: Optional[int]
    first_depth_rrf_hit: Optional[int]
    index_integrity: IndexIntegrityStatus


@dataclass
class DepthCoverage:
    depth: int
    num_relevant_chunks: int
    dense_coverage: float
    bm25_coverage: float
    either_coverage: float
    both_coverage: float
    neither_coverage: float
    rrf_coverage: float
    num_cases: int
    mean_recall_at_depth: float
    mean_reciprocal_rank_at_depth: float


@dataclass
class DepthSensitivityReport:
    benchmark: str
    run_config: dict
    depths: List[int]
    per_depth_coverage: List[DepthCoverage]
    rank_distribution: Dict[str, Dict[str, int]]
    chunk_profiles: List[ChunkDepthProfile]
    definitions: Dict[str, str]


def _first_depth_at_or_below(rank: Optional[int], depths: List[int]) -> Optional[int]:
    """Smallest configured depth D such that rank <= D, or None if rank is
    None or exceeds every configured depth."""
    if rank is None:
        return None
    for d in sorted(depths):
        if rank <= d:
            return d
    return None


def build_chunk_profile(
    case: BenchmarkCase,
    chunk_id: str,
    max_depth_trace,
    depths: List[int],
    chunks_by_doc_id: Dict[str, list],
) -> ChunkDepthProfile:
    """max_depth_trace must be a RetrievalTrace obtained by calling
    retrieve_with_trace() with top_k_per_path == max(depths); its
    dense_candidates/bm25_candidates order is depth-invariant (a search
    engine's top-N ranking does not change based on how many results were
    requested), so ranks derived from it are valid for every smaller
    configured depth too. Its fused_candidates, however, are NOT reused
    for smaller depths -- see compute_depth_coverage's rrf figures, which
    the caller must derive from a depth-matched trace instead."""
    dense_rank = rank_of(chunk_id, max_depth_trace.dense_candidates)
    bm25_rank = rank_of(chunk_id, max_depth_trace.bm25_candidates)

    doc_id = next(iter(case.relevant_doc_ids), None)
    integrity = check_index_integrity(chunk_id, doc_id, chunks_by_doc_id)

    return ChunkDepthProfile(
        case_id=case.case_id,
        chunk_id=chunk_id,
        doc_id=doc_id,
        dense_rank_at_max_depth=dense_rank,
        bm25_rank_at_max_depth=bm25_rank,
        first_depth_dense_hit=_first_depth_at_or_below(dense_rank, depths),
        first_depth_bm25_hit=_first_depth_at_or_below(bm25_rank, depths),
        first_depth_rrf_hit=None,  # filled in by set_rrf_hit_depths once all depth-matched traces are available
        index_integrity=integrity,
    )


def set_rrf_hit_depths(chunk_profiles: List[ChunkDepthProfile], rrf_hits_by_depth: Dict[int, set]) -> None:
    """rrf_hits_by_depth: {depth: set((case_id, chunk_id)) pairs present in
    THAT CASE's own fused top-`depth` for its own query}. Keyed by
    (case_id, chunk_id) rather than chunk_id alone, because the same
    chunk_id can be a relevant target for more than one case (e.g. two
    different questions about the same document whose answers fall in the
    same chunk) -- each case's own query can retrieve it differently, so a
    flat chunk_id set would incorrectly mark it "hit" for every case that
    references it as soon as any ONE of them retrieves it.
    Mutates each profile's first_depth_rrf_hit to the smallest depth at
    which its OWN (case_id, chunk_id) pair was covered."""
    depths_sorted = sorted(rrf_hits_by_depth.keys())
    for profile in chunk_profiles:
        key = (profile.case_id, profile.chunk_id)
        for d in depths_sorted:
            if key in rrf_hits_by_depth[d]:
                profile.first_depth_rrf_hit = d
                break


def compute_depth_coverage(
    depth: int,
    chunk_profiles: List[ChunkDepthProfile],
    rrf_hit_pairs_at_depth: set,
    per_case_metrics_at_depth: List[Tuple[float, float]],
) -> DepthCoverage:
    """rrf_hit_pairs_at_depth: set of (case_id, chunk_id) pairs -- see
    set_rrf_hit_depths for why this must not be a flat chunk_id set."""
    total = len(chunk_profiles)
    if total == 0:
        dense_cov = bm25_cov = either_cov = both_cov = neither_cov = rrf_cov = 0.0
    else:
        dense_hit = lambda p: p.dense_rank_at_max_depth is not None and p.dense_rank_at_max_depth <= depth
        bm25_hit = lambda p: p.bm25_rank_at_max_depth is not None and p.bm25_rank_at_max_depth <= depth

        dense_hits = sum(1 for p in chunk_profiles if dense_hit(p))
        bm25_hits = sum(1 for p in chunk_profiles if bm25_hit(p))
        both_hits = sum(1 for p in chunk_profiles if dense_hit(p) and bm25_hit(p))
        either_hits = sum(1 for p in chunk_profiles if dense_hit(p) or bm25_hit(p))
        rrf_hits = sum(1 for p in chunk_profiles if (p.case_id, p.chunk_id) in rrf_hit_pairs_at_depth)

        dense_cov = dense_hits / total
        bm25_cov = bm25_hits / total
        both_cov = both_hits / total
        either_cov = either_hits / total
        neither_cov = 1.0 - either_cov
        rrf_cov = rrf_hits / total

    recalls = [r for r, _ in per_case_metrics_at_depth]
    rrs = [rr for _, rr in per_case_metrics_at_depth]
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mean_rr = sum(rrs) / len(rrs) if rrs else 0.0

    return DepthCoverage(
        depth=depth,
        num_relevant_chunks=total,
        dense_coverage=dense_cov,
        bm25_coverage=bm25_cov,
        either_coverage=either_cov,
        both_coverage=both_cov,
        neither_coverage=neither_cov,
        rrf_coverage=rrf_cov,
        num_cases=len(per_case_metrics_at_depth),
        mean_recall_at_depth=mean_recall,
        mean_reciprocal_rank_at_depth=mean_rr,
    )


def _bucket_label(rank: Optional[int], sorted_depths: List[int]) -> str:
    if rank is None:
        return f">{sorted_depths[-1]} or absent"
    prev = 0
    for d in sorted_depths:
        if rank <= d:
            return f"<= {d}" if prev == 0 else f"{prev + 1}-{d}"
        prev = d
    return f">{sorted_depths[-1]} or absent"


def build_rank_distribution(chunk_profiles: List[ChunkDepthProfile], depths: List[int]) -> Dict[str, Dict[str, int]]:
    """Rank-distribution bucket boundaries come directly from the
    configured depths list (sorted) -- not a separately hardcoded binning
    scheme."""
    sorted_depths = sorted(depths)
    dist: Dict[str, Dict[str, int]] = {"dense": {}, "bm25": {}}
    for path_key, attr in (("dense", "dense_rank_at_max_depth"), ("bm25", "bm25_rank_at_max_depth")):
        counts: Dict[str, int] = {}
        for profile in chunk_profiles:
            label = _bucket_label(getattr(profile, attr), sorted_depths)
            counts[label] = counts.get(label, 0) + 1
        dist[path_key] = counts
    return dist


DEFINITIONS: Dict[str, str] = {
    "dense_coverage": "Fraction of relevant chunks (pooled across all diagnosed cases) whose dense-path rank is <= the given depth.",
    "bm25_coverage": "Fraction of relevant chunks whose BM25-path rank is <= the given depth.",
    "either_coverage": "Fraction of relevant chunks found by dense OR BM25 within the given depth.",
    "both_coverage": "Fraction of relevant chunks found by dense AND BM25 within the given depth.",
    "neither_coverage": "Fraction of relevant chunks found by neither path within the given depth (1 - either_coverage).",
    "rrf_coverage": "Fraction of relevant chunks present in the fused top-K when retrieval is actually run with K == this depth as both the per-path depth and the fusion cutoff.",
    "mean_recall_at_depth": "Case-averaged Recall@depth (evaluation/metrics.py::recall_at_k, unmodified) -- NOT the same aggregation as the coverage figures above; do not compare them directly.",
    "mean_reciprocal_rank_at_depth": "Case-averaged MRR at this depth (evaluation/metrics.py::reciprocal_rank, unmodified).",
    IndexIntegrityStatus.INDEXED_NORMALLY.value: "The relevant chunk exists in the ground-truth chunk map with non-empty content -- its absence from retrieval, if any, is not an indexing/data problem as far as this check can tell.",
    IndexIntegrityStatus.MISSING_FROM_CHUNK_MAP.value: "The relevant chunk_id could not be found in the chunk map used to build ground truth for its document -- a genuine data-integrity anomaly, since this same map is what produced the ground truth in the first place.",
    IndexIntegrityStatus.EMPTY_CONTENT.value: "The relevant chunk exists but its raw_content is empty or whitespace-only.",
}


def format_depth_table(coverages: List[DepthCoverage]) -> str:
    header = "| Depth | Dense coverage | BM25 coverage | Either-path coverage | Both-path coverage | Neither-path coverage | RRF coverage | Recall@depth | MRR@depth |"
    sep = "|" + "---:|" * 9
    rows = [header, sep]
    for c in sorted(coverages, key=lambda x: x.depth):
        rows.append(
            f"| {c.depth} | {c.dense_coverage:.3f} | {c.bm25_coverage:.3f} | {c.either_coverage:.3f} | "
            f"{c.both_coverage:.3f} | {c.neither_coverage:.3f} | {c.rrf_coverage:.3f} | "
            f"{c.mean_recall_at_depth:.3f} | {c.mean_reciprocal_rank_at_depth:.3f} |"
        )
    return "\n".join(rows)
