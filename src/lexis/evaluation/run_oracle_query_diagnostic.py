"""
CLI: offline query-representation diagnostic -- `oracle_query_diagnostic`
(Step 11).

Holds the retrieval mechanism completely fixed (same indexed corpus, same
embedding model, same Qdrant collection/distance metric, same BM25 index,
same RRF implementation, same top_k/depth checkpoints, via the unmodified
RetrievalEngine.retrieve_with_trace()) and varies ONLY the query text, to
measure how much query representation affects retrievability. This is an
ORACLE diagnostic: the "answer-derived" representation is built from
ground-truth answer text (already used to build ground truth in the first
place -- see dataset/mapping.py) and is used HERE ONLY, never as a
production query. Every metric/artifact this script produces is labeled
`oracle_query_diagnostic` and must never be reported as a baseline or an
improvement.

Three representations are compared, per (case, relevant_chunk) pair:
  A. existing_query        -- the real benchmark query (case.query)
  B. answer_derived        -- the exact mapped ground-truth answer text(s)
                               for that specific chunk (dataset/mapping.py::
                               build_answer_derived_query -- reuses the
                               existing whitespace-normalized substring rule,
                               no new matching logic, no invented wording)
  C. answer_derived_normalized -- B run through the same whitespace
                               normalization already used to build ground
                               truth (dataset/mapping.py::normalize_whitespace).
                               Included only if it would differ from B for at
                               least one pair; otherwise omitted (logged).

Two aggregations of "existing_query" are reported:
  - case-level: the standard aggregation (case-averaged Recall/MRR over each
    case's FULL relevant_chunk_ids set) -- this reuses depth_sensitivity.py
    exactly as run_depth_sensitivity.py does, and is expected to reproduce
    the established baseline at depth 30 as an internal validation that this
    script's retrieval mechanism is unmodified.
  - singleton-unit: the SAME underlying traces, but each (case, chunk_id)
    pair evaluated as its own unit with a singleton relevant set -- this is
    what makes "existing_query" directly comparable to "answer_derived",
    which is inherently singleton (one answer text per one target chunk).
    Pooled dense/BM25/RRF coverage is numerically IDENTICAL between the
    case-level and singleton-unit framings (coverage is already per-chunk);
    only case-averaged Recall/MRR differs, because the denominator (the
    relevant set size) differs.

Uses ONLY retrieve_with_trace() (never RetrievalEngine.retrieve()) and never
mutates settings/RRF/top-k/embedding model/chunking/production data -- a
live settings-snapshot guard (same fields as run_depth_sensitivity.py)
raises if anything changes.

Example:
    python -m lexis.evaluation.run_oracle_query_diagnostic --benchmark cuad \\
        --cuad-path data/cuad_raw/CUAD_v1/CUAD_v1.json \\
        --num-contracts 10 --max-questions 50 --depths 30,100,200
"""
import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from lexis.config import settings
from lexis.evaluation.dataset.cuad_loader import CUADAdapter, CUADLoader, select_contracts
from lexis.evaluation.dataset.mapping import build_answer_derived_query, deterministic_document_id, normalize_whitespace
from lexis.evaluation.depth_sensitivity import (
    build_chunk_profile,
    check_index_integrity,
    compute_depth_coverage,
    set_rrf_hit_depths,
)
from lexis.evaluation.diagnostics import rank_of
from lexis.evaluation.metrics import reciprocal_rank, recall_at_k
from lexis.evaluation.oracle_query_diagnostic import build_representation_case, classify_recovery
from lexis.evaluation.run_eval import chunk_document_text
from lexis.indexing.schema import Chunk
from lexis.ingestion.chunker import SemanticChunker
from lexis.ingestion.embedder import BGEM3Embedder
from lexis.ingestion.parser import LexisParser
from lexis.retrieval.hybrid_retriever import RetrievalEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASELINE_RECALL_AT_30 = 0.5579597701149425
BASELINE_MRR = 0.22195082956259424


async def _run_representation(
    retriever: RetrievalEngine,
    pairs: List[Tuple[str, str, str]],  # (case_id, chunk_id, query_text) -- one query per pair
    depths: List[int],
    base_case_by_id: Dict[str, object],
    chunks_by_doc_id: Dict[str, List[Chunk]],
    representation_label: str,
):
    """Runs retrieve_with_trace() once per (pair, depth), builds a
    ChunkDepthProfile per pair from the max-depth trace, and computes
    per-depth pooled coverage + singleton-unit case-averaged Recall/MRR --
    exactly depth_sensitivity.py's own method, applied to a representation's
    own set of (case, chunk, query) diagnostic units."""
    max_depth = max(depths)
    traces_by_depth_pair: Dict[int, Dict[Tuple[str, str], object]] = {d: {} for d in depths}

    for case_id, chunk_id, query_text in pairs:
        for depth in depths:
            trace = await retriever.retrieve_with_trace(query_text, top_k_per_path=depth, top_n_rrf=depth)
            traces_by_depth_pair[depth][(case_id, chunk_id)] = trace

    chunk_profiles = []
    for case_id, chunk_id, _ in pairs:
        base_case = base_case_by_id[case_id]
        singleton_case = build_representation_case(base_case, chunk_id, base_case.query, representation_label)
        max_trace = traces_by_depth_pair[max_depth][(case_id, chunk_id)]
        chunk_profiles.append(build_chunk_profile(singleton_case, chunk_id, max_trace, depths, chunks_by_doc_id))

    rrf_hits_by_depth: Dict[int, set] = {}
    per_case_metrics_by_depth: Dict[int, List[Tuple[float, float]]] = {}
    for depth in depths:
        hit_pairs = set()
        per_case_metrics = []
        for case_id, chunk_id, _ in pairs:
            trace = traces_by_depth_pair[depth][(case_id, chunk_id)]
            retrieved_ids = [c["id"] for c in trace.final_chunks]
            if chunk_id in retrieved_ids:
                hit_pairs.add((case_id, chunk_id))
            per_case_metrics.append((
                recall_at_k([chunk_id], retrieved_ids, k=depth),
                reciprocal_rank([chunk_id], retrieved_ids),
            ))
        rrf_hits_by_depth[depth] = hit_pairs
        per_case_metrics_by_depth[depth] = per_case_metrics

    set_rrf_hit_depths(chunk_profiles, rrf_hits_by_depth)
    coverages = [
        compute_depth_coverage(d, chunk_profiles, rrf_hits_by_depth[d], per_case_metrics_by_depth[d])
        for d in depths
    ]
    return chunk_profiles, coverages, rrf_hits_by_depth


def _coverage_to_dict(c) -> dict:
    return {
        "depth": c.depth, "num_relevant_chunks": c.num_relevant_chunks,
        "dense_coverage": c.dense_coverage, "bm25_coverage": c.bm25_coverage,
        "either_coverage": c.either_coverage, "both_coverage": c.both_coverage,
        "neither_coverage": c.neither_coverage, "rrf_coverage": c.rrf_coverage,
        "num_cases": c.num_cases, "mean_recall_at_depth": c.mean_recall_at_depth,
        "mean_reciprocal_rank_at_depth": c.mean_reciprocal_rank_at_depth,
    }


def _profile_to_dict(p) -> dict:
    return {
        "case_id": p.case_id, "chunk_id": p.chunk_id, "doc_id": p.doc_id,
        "dense_rank_at_max_depth": p.dense_rank_at_max_depth, "bm25_rank_at_max_depth": p.bm25_rank_at_max_depth,
        "first_depth_dense_hit": p.first_depth_dense_hit, "first_depth_bm25_hit": p.first_depth_bm25_hit,
        "first_depth_rrf_hit": p.first_depth_rrf_hit, "index_integrity": p.index_integrity.value,
    }


async def run_cuad_oracle_query_diagnostic(
    cuad_path: str,
    num_contracts: int,
    max_questions: int,
    depths: List[int],
    output_path: str,
):
    depths = sorted(set(depths))
    max_depth = depths[-1]

    logger.info(f"Loading CUAD dataset from {cuad_path}")
    raw = CUADLoader().load(cuad_path)
    contracts = select_contracts(raw, num_contracts)
    logger.info(f"Selected {len(contracts)} contracts (deterministic, sorted by title).")

    parser = LexisParser()
    embedder = BGEM3Embedder()
    chunker = SemanticChunker(embedder=embedder)

    chunks_by_doc_id: Dict[str, List[Chunk]] = {}
    chunk_by_id: Dict[str, Chunk] = {}
    for contract in contracts:
        title = contract["title"]
        doc_id = deterministic_document_id(title, prefix="cuad")
        context = contract["paragraphs"][0]["context"]
        logger.info(f"Chunking contract '{title}' (doc_id={doc_id})...")
        chunks = chunk_document_text(parser, chunker, context, doc_id)
        chunks_by_doc_id[doc_id] = chunks
        for c in chunks:
            chunk_by_id[c.chunk_id] = c
        logger.info(f"  -> {len(chunks)} chunks")

    cases, unmapped = CUADAdapter().build_cases(raw, chunks_by_doc_id, num_contracts, max_questions)
    logger.info(f"Built {len(cases)} scoreable benchmark cases; {len(unmapped)} unmapped.")
    if not cases:
        logger.error("No scoreable benchmark cases were produced. Cannot run the oracle query diagnostic.")
        return None

    scored_cases = [c for c in cases if c.has_chunk_level_ground_truth()]
    base_case_by_id = {c.case_id: c for c in scored_cases}

    all_pairs = [(c.case_id, chunk_id) for c in scored_cases for chunk_id in c.relevant_chunk_ids]
    logger.info(f"Total (case, relevant_chunk) pairs: {len(all_pairs)}")

    # --- Construct answer-derived query text per pair (CUAD-specific glue:
    # only HERE does this script know about case.metadata["answer_texts"];
    # the actual matching rule is the same generic, existing one). ---
    answer_derived_pairs: List[Tuple[str, str, str]] = []
    answer_unavailable_pairs: List[Tuple[str, str]] = []
    for case_id, chunk_id in all_pairs:
        base_case = base_case_by_id[case_id]
        chunk = chunk_by_id.get(chunk_id)
        answer_texts = base_case.metadata.get("answer_texts", [])
        query_text = build_answer_derived_query(answer_texts, chunk) if chunk is not None else None
        if query_text is None:
            answer_unavailable_pairs.append((case_id, chunk_id))
        else:
            answer_derived_pairs.append((case_id, chunk_id, query_text))
    logger.info(f"Answer-derived query available for {len(answer_derived_pairs)} / {len(all_pairs)} pairs "
                f"({len(answer_unavailable_pairs)} unavailable).")

    # --- Optional representation C: only if normalization would change ANY
    # answer-derived query text; otherwise omitted entirely. ---
    normalized_pairs: List[Tuple[str, str, str]] = []
    any_normalization_changed = False
    for case_id, chunk_id, query_text in answer_derived_pairs:
        normalized = normalize_whitespace(query_text)
        normalized_pairs.append((case_id, chunk_id, normalized))
        if normalized != query_text:
            any_normalization_changed = True
    include_normalized = any_normalization_changed
    logger.info(f"Representation C (answer_derived_normalized) included: {include_normalized}")

    retriever = RetrievalEngine()
    settings_snapshot_before = (
        settings.retrieval_top_k_per_path, settings.retrieval_fusion_top_k, settings.rrf_k,
        settings.qdrant_collection_primary, settings.embedding_model,
    )

    # --- Representation A, case-level (validation) ---
    existing_traces_by_depth: Dict[int, Dict[str, object]] = {d: {} for d in depths}
    for case in scored_cases:
        for depth in depths:
            trace = await retriever.retrieve_with_trace(case.query, top_k_per_path=depth, top_n_rrf=depth)
            existing_traces_by_depth[depth][case.case_id] = trace

    existing_profiles_case_level = []
    for case in scored_cases:
        max_trace = existing_traces_by_depth[max_depth][case.case_id]
        for chunk_id in case.relevant_chunk_ids:
            existing_profiles_case_level.append(build_chunk_profile(case, chunk_id, max_trace, depths, chunks_by_doc_id))

    existing_rrf_hits_by_depth: Dict[int, set] = {}
    existing_case_metrics_by_depth: Dict[int, List[Tuple[float, float]]] = {}
    retrieved_ids_by_depth_case: Dict[int, Dict[str, List[str]]] = {d: {} for d in depths}
    for depth in depths:
        hit_pairs = set()
        case_metrics = []
        for case in scored_cases:
            trace = existing_traces_by_depth[depth][case.case_id]
            retrieved_ids = [c["id"] for c in trace.final_chunks]
            retrieved_ids_by_depth_case[depth][case.case_id] = retrieved_ids
            relevant_ids = list(case.relevant_chunk_ids)
            for cid in relevant_ids:
                if cid in retrieved_ids:
                    hit_pairs.add((case.case_id, cid))
            case_metrics.append((recall_at_k(relevant_ids, retrieved_ids, k=depth), reciprocal_rank(relevant_ids, retrieved_ids)))
        existing_rrf_hits_by_depth[depth] = hit_pairs
        existing_case_metrics_by_depth[depth] = case_metrics

    set_rrf_hit_depths(existing_profiles_case_level, existing_rrf_hits_by_depth)
    existing_coverages_case_level = [
        compute_depth_coverage(d, existing_profiles_case_level, existing_rrf_hits_by_depth[d], existing_case_metrics_by_depth[d])
        for d in depths
    ]

    baseline_reproduced_recall30 = next((c.mean_recall_at_depth for c in existing_coverages_case_level if c.depth == 30), None)
    baseline_reproduced_mrr30 = next((c.mean_reciprocal_rank_at_depth for c in existing_coverages_case_level if c.depth == 30), None)

    # --- Representation A, singleton-unit (reuses the SAME traces -- no new network calls) ---
    existing_profiles_singleton = []
    for case_id, chunk_id in all_pairs:
        base_case = base_case_by_id[case_id]
        singleton_case = build_representation_case(base_case, chunk_id, base_case.query, "existing_query")
        max_trace = existing_traces_by_depth[max_depth][case_id]
        existing_profiles_singleton.append(build_chunk_profile(singleton_case, chunk_id, max_trace, depths, chunks_by_doc_id))

    existing_singleton_metrics_by_depth: Dict[int, List[Tuple[float, float]]] = {}
    for depth in depths:
        metrics = []
        for case_id, chunk_id in all_pairs:
            retrieved_ids = retrieved_ids_by_depth_case[depth][case_id]
            metrics.append((recall_at_k([chunk_id], retrieved_ids, k=depth), reciprocal_rank([chunk_id], retrieved_ids)))
        existing_singleton_metrics_by_depth[depth] = metrics
    set_rrf_hit_depths(existing_profiles_singleton, existing_rrf_hits_by_depth)
    existing_coverages_singleton = [
        compute_depth_coverage(d, existing_profiles_singleton, existing_rrf_hits_by_depth[d], existing_singleton_metrics_by_depth[d])
        for d in depths
    ]

    # --- Representation B: answer_derived ---
    answer_profiles, answer_coverages, answer_rrf_hits_by_depth = await _run_representation(
        retriever, answer_derived_pairs, depths, base_case_by_id, chunks_by_doc_id, "answer_derived",
    )

    # --- Representation C: answer_derived_normalized (conditional) ---
    normalized_profiles, normalized_coverages, normalized_rrf_hits_by_depth = ([], [], {})
    if include_normalized:
        normalized_profiles, normalized_coverages, normalized_rrf_hits_by_depth = await _run_representation(
            retriever, normalized_pairs, depths, base_case_by_id, chunks_by_doc_id, "answer_derived_normalized",
        )

    settings_snapshot_after = (
        settings.retrieval_top_k_per_path, settings.retrieval_fusion_top_k, settings.rrf_k,
        settings.qdrant_collection_primary, settings.embedding_model,
    )
    if settings_snapshot_before != settings_snapshot_after:
        raise RuntimeError(
            "Production Settings changed during the oracle query diagnostic -- "
            "this must never happen; refusing to write a report."
        )

    # --- Hard-miss set (Step 10's exact definition, re-derived here, not
    # read from a prior artifact): absent from BOTH raw paths under the
    # EXISTING query at max_depth. ---
    existing_profile_by_pair = {(p.case_id, p.chunk_id): p for p in existing_profiles_case_level}
    answer_profile_by_pair = {(p.case_id, p.chunk_id): p for p in answer_profiles}

    hard_miss_pairs = [
        (p.case_id, p.chunk_id) for p in existing_profiles_case_level
        if p.dense_rank_at_max_depth is None and p.bm25_rank_at_max_depth is None
    ]
    logger.info(f"Hard-miss set (existing query, absent from both raw paths at depth {max_depth}): {len(hard_miss_pairs)}")

    hard_miss_recovery = []
    for case_id, chunk_id in hard_miss_pairs:
        answer_profile = answer_profile_by_pair.get((case_id, chunk_id))
        if answer_profile is None:
            hard_miss_recovery.append({
                "case_id": case_id, "chunk_id": chunk_id,
                "answer_derived_available": False,
                "raw_path_recovered": None, "rrf_recovered_by_depth": {d: None for d in depths},
            })
            continue
        raw_recovered = answer_profile.dense_rank_at_max_depth is not None or answer_profile.bm25_rank_at_max_depth is not None
        rrf_recovered_by_depth = {
            d: (answer_profile.first_depth_rrf_hit is not None and answer_profile.first_depth_rrf_hit <= d)
            for d in depths
        }
        hard_miss_recovery.append({
            "case_id": case_id, "chunk_id": chunk_id,
            "answer_derived_available": True,
            "raw_path_recovered": raw_recovered,
            "rrf_recovered_by_depth": rrf_recovered_by_depth,
        })

    num_hard_miss_raw_recovered = sum(1 for r in hard_miss_recovery if r["raw_path_recovered"] is True)
    num_hard_miss_rrf_recovered_by_depth = {
        d: sum(1 for r in hard_miss_recovery if r["rrf_recovered_by_depth"].get(d) is True) for d in depths
    }
    num_hard_miss_answer_unavailable = sum(1 for r in hard_miss_recovery if not r["answer_derived_available"])

    # --- Per-pair recovery classification across ALL pairs (not just hard misses), at each depth ---
    recovery_by_depth = {}
    for depth in depths:
        records = []
        for case_id, chunk_id in all_pairs:
            existing_profile = existing_profile_by_pair.get((case_id, chunk_id))
            answer_profile = answer_profile_by_pair.get((case_id, chunk_id))
            if existing_profile is None or answer_profile is None:
                continue
            original_hit = existing_profile.first_depth_rrf_hit is not None and existing_profile.first_depth_rrf_hit <= depth
            alternate_hit = answer_profile.first_depth_rrf_hit is not None and answer_profile.first_depth_rrf_hit <= depth
            records.append(classify_recovery(case_id, chunk_id, depth, original_hit, alternate_hit))
        recovery_by_depth[depth] = records

    run_config = {
        "num_contracts": num_contracts, "max_questions": max_questions, "depths": depths,
        "embedding_model": settings.embedding_model, "rrf_k": settings.rrf_k,
        "include_normalized_representation": include_normalized,
    }

    def _recovery_summary(depth):
        recs = recovery_by_depth[depth]
        return {
            "num_pairs": len(recs),
            "num_became_retrievable": sum(1 for r in recs if r.became_retrievable),
            "num_remains_missing": sum(1 for r in recs if r.remains_missing),
            "num_original_succeeds": sum(1 for r in recs if r.original_hit),
            "num_alternate_succeeds": sum(1 for r in recs if r.alternate_hit),
        }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_type": "oracle_query_diagnostic",
        "benchmark": "cuad",
        "cuad_source_path": cuad_path,
        "run_config": run_config,
        "baseline_reference": {
            "recall_at_30": BASELINE_RECALL_AT_30,
            "mrr": BASELINE_MRR,
            "note": "Established Steps 7-9 baseline, unmodified by this diagnostic.",
        },
        "baseline_reproduction_check": {
            "recall_at_30_reproduced": baseline_reproduced_recall30,
            "mrr_reproduced": baseline_reproduced_mrr30,
            "matches_baseline": (
                baseline_reproduced_recall30 == BASELINE_RECALL_AT_30 and baseline_reproduced_mrr30 == BASELINE_MRR
            ),
        },
        "num_total_pairs": len(all_pairs),
        "num_answer_derived_available": len(answer_derived_pairs),
        "num_answer_derived_unavailable": len(answer_unavailable_pairs),
        "representations": {
            "existing_query_case_level": [_coverage_to_dict(c) for c in existing_coverages_case_level],
            "existing_query_singleton": [_coverage_to_dict(c) for c in existing_coverages_singleton],
            "answer_derived": [_coverage_to_dict(c) for c in answer_coverages],
            "answer_derived_normalized": [_coverage_to_dict(c) for c in normalized_coverages] if include_normalized else None,
        },
        "hard_miss_set_size": len(hard_miss_pairs),
        "hard_miss_recovery": {
            "num_raw_path_recovered": num_hard_miss_raw_recovered,
            "num_rrf_recovered_by_depth": num_hard_miss_rrf_recovered_by_depth,
            "num_answer_derived_unavailable_for_hard_miss": num_hard_miss_answer_unavailable,
            "per_pair": hard_miss_recovery,
        },
        "recovery_by_depth": {str(d): _recovery_summary(d) for d in depths},
        "chunk_profiles": {
            "existing_query_case_level": [_profile_to_dict(p) for p in existing_profiles_case_level],
            "answer_derived": [_profile_to_dict(p) for p in answer_profiles],
            "answer_derived_normalized": [_profile_to_dict(p) for p in normalized_profiles] if include_normalized else None,
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Oracle query diagnostic report written to {output_path}")

    # --- Human-readable summary ---
    lines = ["=== oracle_query_diagnostic Summary (cuad) ===", ""]
    lines.append(f"Baseline reference (unchanged): Recall@30={BASELINE_RECALL_AT_30}, MRR={BASELINE_MRR}")
    lines.append(f"Baseline reproduction check (existing_query, case-level, this run): "
                 f"Recall@30={baseline_reproduced_recall30}, MRR={baseline_reproduced_mrr30}")
    lines.append("")
    lines.append("| Representation | Depth | Dense cov | BM25 cov | Either cov | RRF Recall@depth | RRF MRR@depth |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, coverages in [
        ("existing_query (singleton)", existing_coverages_singleton),
        ("answer_derived", answer_coverages),
    ] + ([("answer_derived_normalized", normalized_coverages)] if include_normalized else []):
        for c in coverages:
            lines.append(f"| {label} | {c.depth} | {c.dense_coverage:.3f} | {c.bm25_coverage:.3f} | {c.either_coverage:.3f} | "
                         f"{c.mean_recall_at_depth:.3f} | {c.mean_reciprocal_rank_at_depth:.3f} |")
    lines.append("")
    lines.append(f"Hard-miss set (from this run, same definition as Step 10): {len(hard_miss_pairs)}")
    lines.append(f"  Answer-derived query unavailable for: {num_hard_miss_answer_unavailable}")
    lines.append(f"  Recovered into raw dense/BM25 candidate lists (depth {max_depth}): {num_hard_miss_raw_recovered}")
    for d in depths:
        lines.append(f"  Recovered into final RRF top-{d}: {num_hard_miss_rrf_recovered_by_depth[d]}")
    lines.append(f"  Remain unrecovered (raw-path) even under answer-derived query: {len(hard_miss_pairs) - num_hard_miss_raw_recovered - num_hard_miss_answer_unavailable}")

    summary = "\n".join(lines)
    logger.info("\n" + summary)
    return output


def main():
    arg_parser = argparse.ArgumentParser(description="Offline oracle query-representation diagnostic -- measurement only, never a production baseline/improvement claim.")
    arg_parser.add_argument("--benchmark", choices=["cuad"], default="cuad")
    arg_parser.add_argument("--cuad-path", default="data/cuad_raw/CUAD_v1/CUAD_v1.json")
    arg_parser.add_argument("--num-contracts", type=int, default=10)
    arg_parser.add_argument("--max-questions", type=int, default=50)
    arg_parser.add_argument("--depths", default="30,100,200", help="Comma-separated diagnostic depths.")
    arg_parser.add_argument("--output", default="evaluation/reports/cuad_oracle_query_diagnostic.json")
    args = arg_parser.parse_args()

    depths = [int(d.strip()) for d in args.depths.split(",") if d.strip()]

    if args.benchmark == "cuad":
        asyncio.run(run_cuad_oracle_query_diagnostic(
            args.cuad_path, args.num_contracts, args.max_questions, depths, args.output,
        ))
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()
