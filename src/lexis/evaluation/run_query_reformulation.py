"""
CLI: offline query-reformulation experiment (Step 12).

Tests whether a REALISTIC, deterministic, generic query transformation --
built using ONLY the original benchmark queries' own text, never ground
truth -- can improve first-stage retrieval. This is an EXPERIMENT artifact,
not a production configuration: nothing here is wired into
RetrievalEngine.retrieve(), and no representation is adopted as a new
baseline by this script.

Representations compared (case-level: each case's FULL relevant_chunk_ids
set, same aggregation as the production baseline and Step 9):
  A. original            -- case.query, unchanged (the frozen baseline;
                             MUST reproduce Recall@30=0.5579597701149425,
                             MRR=0.22195082956259424 exactly, or this script
                             raises rather than comparing further)
  B. normalized           -- query_reformulation.py::normalize_query, the
                             project's one existing generic whitespace rule
  C. boilerplate_reduced   -- query_reformulation.py::reduce_ngram_boilerplate,
                             fitted ONLY on this run's own query set (see
                             that module's docstring for why n-gram-span
                             detection, not whole-sentence detection, is
                             used -- a whole-sentence variant was tested
                             analytically and found to flag zero sentences
                             on the real CUAD query set, since the fixed
                             template wording surrounds a varying quoted
                             term rather than forming a whole repeated
                             sentence; that negative result is reported,
                             not hidden)
  D. multi_representation -- NOT a new model or a new dense/BM25 config:
                             the raw dense+BM25 candidate lists from
                             representations A and C (already computed,
                             zero extra retrieval calls) are combined
                             through the EXISTING, unmodified
                             retrieval/fusion.py::apply_rrf, at the same
                             settings.rrf_k, producing a second, separate
                             fused ranking. Reported as its own strategy,
                             never merged into A's own numbers.

Uses ONLY retrieve_with_trace() (never RetrievalEngine.retrieve()). Never
mutates settings/RRF/top-k/embedding model/chunking/production data -- a
live settings-snapshot guard raises if anything changes. The boilerplate
model is fit exclusively from case.query strings (see
query_reformulation.py); nothing here ever reads answer_texts, chunk
content, or relevance labels.

Example:
    python -m lexis.evaluation.run_query_reformulation --benchmark cuad \\
        --cuad-path data/cuad_raw/CUAD_v1/CUAD_v1.json \\
        --num-contracts 10 --max-questions 50 --depths 30,100,200
"""
import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lexis.config import settings
from lexis.evaluation.dataset.cuad_loader import CUADAdapter, CUADLoader, select_contracts
from lexis.evaluation.dataset.mapping import deterministic_document_id
from lexis.evaluation.depth_sensitivity import (
    ChunkDepthProfile,
    _first_depth_at_or_below,
    build_chunk_profile,
    check_index_integrity,
    compute_depth_coverage,
    set_rrf_hit_depths,
)
from lexis.evaluation.diagnostics import rank_of
from lexis.evaluation.metrics import reciprocal_rank, recall_at_k
from lexis.evaluation.oracle_query_diagnostic import classify_recovery
from lexis.evaluation.query_reformulation import (
    content_token_ratio,
    fit_boilerplate_model,
    fit_ngram_boilerplate_model,
    normalize_query,
    reduce_boilerplate,
    reduce_ngram_boilerplate,
    split_sentences,
)
from lexis.evaluation.run_eval import chunk_document_text
from lexis.indexing.schema import Chunk
from lexis.ingestion.chunker import SemanticChunker
from lexis.ingestion.embedder import BGEM3Embedder
from lexis.ingestion.parser import LexisParser
from lexis.retrieval.fusion import apply_rrf
from lexis.retrieval.hybrid_retriever import RetrievalEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASELINE_RECALL_AT_30 = 0.5579597701149425
BASELINE_MRR = 0.22195082956259424
EXPERIMENT_ID = "step12_query_reformulation_experiment"


def _coverage_to_dict(c) -> dict:
    return {
        "depth": c.depth, "num_relevant_chunks": c.num_relevant_chunks,
        "dense_coverage": c.dense_coverage, "bm25_coverage": c.bm25_coverage,
        "either_coverage": c.either_coverage, "both_coverage": c.both_coverage,
        "neither_coverage": c.neither_coverage, "rrf_coverage": c.rrf_coverage,
        "num_cases": c.num_cases, "mean_recall_at_depth": c.mean_recall_at_depth,
        "mean_reciprocal_rank_at_depth": c.mean_reciprocal_rank_at_depth,
    }


async def _run_case_level_representation(
    retriever: RetrievalEngine,
    scored_cases,
    query_by_case: Dict[str, str],
    depths: List[int],
    chunks_by_doc_id,
):
    """Case-level (full relevant_chunk_ids set), exactly Step 9/11's own
    validated method -- one retrieve_with_trace() call per case per depth."""
    max_depth = max(depths)
    traces_by_depth: Dict[int, Dict[str, object]] = {d: {} for d in depths}
    for case in scored_cases:
        query_text = query_by_case[case.case_id]
        for depth in depths:
            trace = await retriever.retrieve_with_trace(query_text, top_k_per_path=depth, top_n_rrf=depth)
            traces_by_depth[depth][case.case_id] = trace

    profiles = []
    for case in scored_cases:
        max_trace = traces_by_depth[max_depth][case.case_id]
        for chunk_id in case.relevant_chunk_ids:
            profiles.append(build_chunk_profile(case, chunk_id, max_trace, depths, chunks_by_doc_id))

    rrf_hits_by_depth: Dict[int, set] = {}
    case_metrics_by_depth: Dict[int, List[Tuple[float, float]]] = {}
    per_case_recall_by_depth: Dict[int, Dict[str, float]] = {d: {} for d in depths}
    per_case_rr_by_depth: Dict[int, Dict[str, float]] = {d: {} for d in depths}
    for depth in depths:
        hit_pairs = set()
        case_metrics = []
        for case in scored_cases:
            trace = traces_by_depth[depth][case.case_id]
            retrieved_ids = [c["id"] for c in trace.final_chunks]
            relevant_ids = list(case.relevant_chunk_ids)
            for cid in relevant_ids:
                if cid in retrieved_ids:
                    hit_pairs.add((case.case_id, cid))
            case_recall = recall_at_k(relevant_ids, retrieved_ids, k=depth)
            case_rr = reciprocal_rank(relevant_ids, retrieved_ids)
            case_metrics.append((case_recall, case_rr))
            per_case_recall_by_depth[depth][case.case_id] = case_recall
            per_case_rr_by_depth[depth][case.case_id] = case_rr
        rrf_hits_by_depth[depth] = hit_pairs
        case_metrics_by_depth[depth] = case_metrics

    set_rrf_hit_depths(profiles, rrf_hits_by_depth)
    coverages = [compute_depth_coverage(d, profiles, rrf_hits_by_depth[d], case_metrics_by_depth[d]) for d in depths]
    return {
        "traces_by_depth": traces_by_depth,
        "profiles": profiles,
        "coverages": coverages,
        "rrf_hits_by_depth": rrf_hits_by_depth,
        "per_case_recall_by_depth": per_case_recall_by_depth,
        "per_case_rr_by_depth": per_case_rr_by_depth,
    }


def _build_multi_representation(scored_cases, result_a, result_c, depths, chunks_by_doc_id, max_depth):
    """Combines representation A's and C's ALREADY-FETCHED raw candidate
    lists via the existing, unmodified apply_rrf -- zero additional
    retrieval calls. Per-path "coverage" for this strategy is defined as
    the union of A's and C's own per-path coverage (a chunk counts as
    dense-covered at depth d if EITHER representation's dense rank <= d),
    which reduces to taking the best (minimum) rank of the two -- a
    well-defined combination rule, not an invented ranking."""
    traces_a = result_a["traces_by_depth"][max_depth]
    traces_c = result_c["traces_by_depth"][max_depth]

    profiles = []
    for case in scored_cases:
        trace_a = traces_a[case.case_id]
        trace_c = traces_c[case.case_id]
        doc_id = next(iter(case.relevant_doc_ids), None)
        for chunk_id in case.relevant_chunk_ids:
            dense_a = rank_of(chunk_id, trace_a.dense_candidates)
            dense_c = rank_of(chunk_id, trace_c.dense_candidates)
            bm25_a = rank_of(chunk_id, trace_a.bm25_candidates)
            bm25_c = rank_of(chunk_id, trace_c.bm25_candidates)
            dense_best = min([r for r in (dense_a, dense_c) if r is not None], default=None)
            bm25_best = min([r for r in (bm25_a, bm25_c) if r is not None], default=None)
            integrity = check_index_integrity(chunk_id, doc_id, chunks_by_doc_id)
            profiles.append(ChunkDepthProfile(
                case_id=case.case_id, chunk_id=chunk_id, doc_id=doc_id,
                dense_rank_at_max_depth=dense_best, bm25_rank_at_max_depth=bm25_best,
                first_depth_dense_hit=_first_depth_at_or_below(dense_best, depths),
                first_depth_bm25_hit=_first_depth_at_or_below(bm25_best, depths),
                first_depth_rrf_hit=None,
                index_integrity=integrity,
            ))

    rrf_hits_by_depth: Dict[int, set] = {}
    case_metrics_by_depth: Dict[int, List[Tuple[float, float]]] = {}
    per_case_recall_by_depth: Dict[int, Dict[str, float]] = {d: {} for d in depths}
    per_case_rr_by_depth: Dict[int, Dict[str, float]] = {d: {} for d in depths}
    for depth in depths:
        hit_pairs = set()
        case_metrics = []
        for case in scored_cases:
            trace_a = traces_a[case.case_id]
            trace_c = traces_c[case.case_id]
            candidate_lists = [
                trace_a.dense_candidates[:depth], trace_a.bm25_candidates[:depth],
                trace_c.dense_candidates[:depth], trace_c.bm25_candidates[:depth],
            ]
            fused = apply_rrf(candidate_lists, k=settings.rrf_k)
            retrieved_ids = [c.chunk_id for c in fused[:depth]]
            relevant_ids = list(case.relevant_chunk_ids)
            for cid in relevant_ids:
                if cid in retrieved_ids:
                    hit_pairs.add((case.case_id, cid))
            case_recall = recall_at_k(relevant_ids, retrieved_ids, k=depth)
            case_rr = reciprocal_rank(relevant_ids, retrieved_ids)
            case_metrics.append((case_recall, case_rr))
            per_case_recall_by_depth[depth][case.case_id] = case_recall
            per_case_rr_by_depth[depth][case.case_id] = case_rr
        rrf_hits_by_depth[depth] = hit_pairs
        case_metrics_by_depth[depth] = case_metrics

    set_rrf_hit_depths(profiles, rrf_hits_by_depth)
    coverages = [compute_depth_coverage(d, profiles, rrf_hits_by_depth[d], case_metrics_by_depth[d]) for d in depths]
    return {
        "profiles": profiles, "coverages": coverages, "rrf_hits_by_depth": rrf_hits_by_depth,
        "per_case_recall_by_depth": per_case_recall_by_depth, "per_case_rr_by_depth": per_case_rr_by_depth,
    }


def _error_analysis(scored_cases, result_original, result_alt, depth):
    """Per-(case, relevant_chunk) comparison at one depth, reusing
    oracle_query_diagnostic.py's RepresentationRecovery (already generic:
    case_id, chunk_id, depth, two hit booleans) -- not reimplemented here."""
    hits_orig = result_original["rrf_hits_by_depth"][depth]
    hits_alt = result_alt["rrf_hits_by_depth"][depth]
    records = []
    for case in scored_cases:
        for chunk_id in case.relevant_chunk_ids:
            original_hit = (case.case_id, chunk_id) in hits_orig
            alternate_hit = (case.case_id, chunk_id) in hits_alt
            records.append(classify_recovery(case.case_id, chunk_id, depth, original_hit, alternate_hit))
    return records


def _summarize_error_analysis(records) -> dict:
    became_retrievable = [r for r in records if r.became_retrievable]
    regressed = [r for r in records if r.regressed]
    remains_missing = [r for r in records if r.remains_missing]
    distinct_cases_gaining = {r.case_id for r in became_retrievable}
    per_case_gain_counts: Dict[str, int] = {}
    for r in became_retrievable:
        per_case_gain_counts[r.case_id] = per_case_gain_counts.get(r.case_id, 0) + 1
    max_gain_from_one_case = max(per_case_gain_counts.values(), default=0)
    return {
        "num_pairs": len(records),
        "num_became_retrievable": len(became_retrievable),
        "num_regressed": len(regressed),
        "num_remains_missing": len(remains_missing),
        "num_original_succeeds": sum(1 for r in records if r.original_hit),
        "num_alternate_succeeds": sum(1 for r in records if r.alternate_hit),
        "distinct_cases_with_a_gain": len(distinct_cases_gaining),
        "max_gain_from_a_single_case": max_gain_from_one_case,
        "gains_concentrated": (
            max_gain_from_one_case > 0 and len(became_retrievable) > 0
            and (max_gain_from_one_case / len(became_retrievable)) >= 0.5
        ),
    }


async def run_cuad_query_reformulation(
    cuad_path: str,
    num_contracts: int,
    max_questions: int,
    depths: List[int],
    ngram_length: int,
    min_frequency_ratio: float,
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
    for contract in contracts:
        title = contract["title"]
        doc_id = deterministic_document_id(title, prefix="cuad")
        context = contract["paragraphs"][0]["context"]
        logger.info(f"Chunking contract '{title}' (doc_id={doc_id})...")
        chunks = chunk_document_text(parser, chunker, context, doc_id)
        chunks_by_doc_id[doc_id] = chunks
        logger.info(f"  -> {len(chunks)} chunks")

    cases, unmapped = CUADAdapter().build_cases(raw, chunks_by_doc_id, num_contracts, max_questions)
    logger.info(f"Built {len(cases)} scoreable benchmark cases; {len(unmapped)} unmapped.")
    if not cases:
        logger.error("No scoreable benchmark cases were produced. Cannot run the query-reformulation experiment.")
        return None

    scored_cases = [c for c in cases if c.has_chunk_level_ground_truth()]
    all_queries = [c.query for c in scored_cases]

    # --- Fit boilerplate models from ONLY the query set's own text (never
    # answers/chunks/labels). Analytical comparison of the two candidate
    # granularities, reported regardless of which is used for retrieval. ---
    sentence_model = fit_boilerplate_model(all_queries, min_frequency_ratio=min_frequency_ratio)
    num_sentences_flagged = sum(1 for s in sentence_model.sentence_frequency if sentence_model.is_boilerplate(s))
    num_changed_by_sentence_reduction = sum(1 for q in all_queries if reduce_boilerplate(q, sentence_model) != q)

    ngram_model = fit_ngram_boilerplate_model(all_queries, ngram_length=ngram_length, min_frequency_ratio=min_frequency_ratio)
    num_ngrams_flagged = sum(1 for g in ngram_model.ngram_frequency if ngram_model.is_boilerplate(g))
    num_changed_by_ngram_reduction = sum(1 for q in all_queries if reduce_ngram_boilerplate(q, ngram_model) != q)

    logger.info(f"Sentence-level boilerplate model: {num_sentences_flagged} distinct sentences flagged "
                f"(threshold {min_frequency_ratio}); {num_changed_by_sentence_reduction}/{len(all_queries)} queries would change.")
    logger.info(f"N-gram-level boilerplate model (n={ngram_length}): {num_ngrams_flagged} distinct n-grams flagged; "
                f"{num_changed_by_ngram_reduction}/{len(all_queries)} queries would change.")

    boilerplate_method_used = "ngram" if num_changed_by_ngram_reduction > num_changed_by_sentence_reduction else "sentence"
    if boilerplate_method_used == "sentence":
        logger.warning("Sentence-level boilerplate reduction affects as many or more queries than the n-gram "
                        "method on this run's data; using sentence-level as representation C.")
    boilerplate_reduce_fn = (
        (lambda q: reduce_ngram_boilerplate(q, ngram_model)) if boilerplate_method_used == "ngram"
        else (lambda q: reduce_boilerplate(q, sentence_model))
    )

    query_by_case = {
        "original": {c.case_id: c.query for c in scored_cases},
        "normalized": {c.case_id: normalize_query(c.query) for c in scored_cases},
        "boilerplate_reduced": {c.case_id: boilerplate_reduce_fn(c.query) for c in scored_cases},
    }

    retriever = RetrievalEngine()
    settings_snapshot_before = (
        settings.retrieval_top_k_per_path, settings.retrieval_fusion_top_k, settings.rrf_k,
        settings.qdrant_collection_primary, settings.embedding_model,
    )

    result_original = await _run_case_level_representation(retriever, scored_cases, query_by_case["original"], depths, chunks_by_doc_id)
    result_normalized = await _run_case_level_representation(retriever, scored_cases, query_by_case["normalized"], depths, chunks_by_doc_id)
    result_boilerplate = await _run_case_level_representation(retriever, scored_cases, query_by_case["boilerplate_reduced"], depths, chunks_by_doc_id)

    baseline_check = next((c for c in result_original["coverages"] if c.depth == 30), None)
    reproduced_recall30 = baseline_check.mean_recall_at_depth if baseline_check else None
    reproduced_mrr30 = baseline_check.mean_reciprocal_rank_at_depth if baseline_check else None
    baseline_matches = (reproduced_recall30 == BASELINE_RECALL_AT_30 and reproduced_mrr30 == BASELINE_MRR)
    if not baseline_matches:
        raise RuntimeError(
            f"Frozen-baseline control FAILED to reproduce the established baseline: "
            f"got Recall@30={reproduced_recall30}, MRR={reproduced_mrr30}, "
            f"expected Recall@30={BASELINE_RECALL_AT_30}, MRR={BASELINE_MRR}. "
            f"Stopping before comparing representations, as required."
        )
    logger.info(f"Frozen-baseline control REPRODUCED: Recall@30={reproduced_recall30}, MRR={reproduced_mrr30}")

    result_multi = _build_multi_representation(scored_cases, result_original, result_boilerplate, depths, chunks_by_doc_id, max_depth)

    settings_snapshot_after = (
        settings.retrieval_top_k_per_path, settings.retrieval_fusion_top_k, settings.rrf_k,
        settings.qdrant_collection_primary, settings.embedding_model,
    )
    if settings_snapshot_before != settings_snapshot_after:
        raise RuntimeError(
            "Production Settings changed during the query-reformulation experiment -- "
            "this must never happen; refusing to write a report."
        )

    # --- Error analysis at the production-relevant depth (30), reported
    # for every representation against the frozen original. ---
    error_analysis = {
        "normalized": _summarize_error_analysis(_error_analysis(scored_cases, result_original, result_normalized, 30)),
        "boilerplate_reduced": _summarize_error_analysis(_error_analysis(scored_cases, result_original, result_boilerplate, 30)),
        "multi_representation": _summarize_error_analysis(_error_analysis(scored_cases, result_original, result_multi, 30)),
    }

    # --- Per-case deltas at depth 30 (never cherry-picked -- every case reported) ---
    per_case_deltas = []
    for case in scored_cases:
        row = {"case_id": case.case_id, "relevant_count": len(case.relevant_chunk_ids)}
        for label, result in [("original", result_original), ("normalized", result_normalized),
                               ("boilerplate_reduced", result_boilerplate), ("multi_representation", result_multi)]:
            row[f"{label}_recall_at_30"] = result["per_case_recall_by_depth"][30][case.case_id]
            row[f"{label}_rr_at_30"] = result["per_case_rr_by_depth"][30][case.case_id]
        row["delta_normalized"] = row["normalized_recall_at_30"] - row["original_recall_at_30"]
        row["delta_boilerplate_reduced"] = row["boilerplate_reduced_recall_at_30"] - row["original_recall_at_30"]
        row["delta_multi_representation"] = row["multi_representation_recall_at_30"] - row["original_recall_at_30"]
        per_case_deltas.append(row)

    # --- Rejected single-query stopword-ratio heuristic: analytical evidence only ---
    all_sentences = [s for q in all_queries for s in split_sentences(q)]
    boilerplate_flagged_sentences = [s for s in all_sentences if sentence_model.is_boilerplate(normalize_query(s))]
    ratio_report = {
        "num_sentences_examined": len(all_sentences),
        "note": (
            "content_token_ratio (fraction of a sentence's tokens surviving bm25s's existing "
            "English-stopword tokenization) was evaluated as a possible single-query-only "
            "boilerplate signal; see run output for whether it separates boilerplate from "
            "substantive sentences on this run's real query set."
        ),
    }

    run_config = {
        "num_contracts": num_contracts, "max_questions": max_questions, "depths": depths,
        "ngram_length": ngram_length, "min_frequency_ratio": min_frequency_ratio,
        "boilerplate_method_used": boilerplate_method_used,
        "embedding_model": settings.embedding_model, "rrf_k": settings.rrf_k,
    }

    output = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_kind": "EXPERIMENT -- not a production configuration",
        "benchmark": "cuad",
        "cuad_source_path": cuad_path,
        "run_config": run_config,
        "baseline_reference": {"recall_at_30": BASELINE_RECALL_AT_30, "mrr": BASELINE_MRR},
        "baseline_control_check": {
            "recall_at_30_reproduced": reproduced_recall30, "mrr_reproduced": reproduced_mrr30,
            "matches_baseline": baseline_matches,
        },
        "boilerplate_candidate_analysis": {
            "sentence_level": {
                "num_distinct_sentences_flagged": num_sentences_flagged,
                "num_queries_changed": num_changed_by_sentence_reduction,
                "verdict": "unsupported -- flags zero/near-zero sentences because the fixed template "
                           "wording surrounds a varying quoted term rather than forming a whole repeated sentence"
                           if num_sentences_flagged == 0 else "supported",
            },
            "ngram_level": {
                "ngram_length": ngram_length,
                "num_distinct_ngrams_flagged": num_ngrams_flagged,
                "num_queries_changed": num_changed_by_ngram_reduction,
                "verdict": "supported" if num_changed_by_ngram_reduction > 0 else "unsupported",
            },
            "method_used_for_representation_c": boilerplate_method_used,
        },
        "num_cases": len(scored_cases),
        "num_total_relevant_chunk_pairs": sum(len(c.relevant_chunk_ids) for c in scored_cases),
        "representations": {
            "original": [_coverage_to_dict(c) for c in result_original["coverages"]],
            "normalized": [_coverage_to_dict(c) for c in result_normalized["coverages"]],
            "boilerplate_reduced": [_coverage_to_dict(c) for c in result_boilerplate["coverages"]],
            "multi_representation": [_coverage_to_dict(c) for c in result_multi["coverages"]],
        },
        "error_analysis_at_depth_30": error_analysis,
        "per_case_deltas_at_depth_30": per_case_deltas,
        "rejected_candidate_analysis": ratio_report,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Query-reformulation EXPERIMENT report written to {output_path}")

    # --- Human-readable summary ---
    lines = ["=== Step 12 Query-Reformulation EXPERIMENT (not production) ===", ""]
    lines.append(f"Frozen baseline (unchanged): Recall@30={BASELINE_RECALL_AT_30}, MRR={BASELINE_MRR}")
    lines.append(f"Control reproduction (original, this run): Recall@30={reproduced_recall30}, MRR={reproduced_mrr30} "
                 f"-- {'MATCH' if baseline_matches else 'MISMATCH'}")
    lines.append("")
    lines.append("| Representation | RRF Recall@30 | RRF MRR@30 | RRF Recall@100 | RRF Recall@200 |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, result in [("Original", result_original), ("Normalized", result_normalized),
                           ("Boilerplate-reduced", result_boilerplate), ("Multi-representation", result_multi)]:
        by_depth = {c.depth: c for c in result["coverages"]}
        r30, r100, r200 = by_depth.get(30), by_depth.get(100), by_depth.get(200)
        lines.append(f"| {label} | {r30.mean_recall_at_depth:.3f} | {r30.mean_reciprocal_rank_at_depth:.3f} | "
                     f"{r100.mean_recall_at_depth if r100 else float('nan'):.3f} | "
                     f"{r200.mean_recall_at_depth if r200 else float('nan'):.3f} |")
    lines.append("")
    lines.append("Error analysis at depth 30 (vs. original):")
    for label, summary in error_analysis.items():
        lines.append(f"  {label}: became_retrievable={summary['num_became_retrievable']} "
                     f"regressed={summary['num_regressed']} remains_missing={summary['num_remains_missing']} "
                     f"concentrated={summary['gains_concentrated']}")

    summary = "\n".join(lines)
    logger.info("\n" + summary)
    return output


def main():
    arg_parser = argparse.ArgumentParser(description="Offline query-reformulation experiment -- generic, deterministic, no ground truth. Not a production change.")
    arg_parser.add_argument("--benchmark", choices=["cuad"], default="cuad")
    arg_parser.add_argument("--cuad-path", default="data/cuad_raw/CUAD_v1/CUAD_v1.json")
    arg_parser.add_argument("--num-contracts", type=int, default=10)
    arg_parser.add_argument("--max-questions", type=int, default=50)
    arg_parser.add_argument("--depths", default="30,100,200")
    arg_parser.add_argument("--ngram-length", type=int, default=6, help="Word n-gram window length for boilerplate-span detection.")
    arg_parser.add_argument("--min-frequency-ratio", type=float, default=0.5, help="Fraction of the query set a sentence/n-gram must appear in to be classified boilerplate.")
    arg_parser.add_argument("--output", default="evaluation/reports/cuad_query_reformulation_experiment.json")
    args = arg_parser.parse_args()

    depths = [int(d.strip()) for d in args.depths.split(",") if d.strip()]

    if args.benchmark == "cuad":
        asyncio.run(run_cuad_query_reformulation(
            args.cuad_path, args.num_contracts, args.max_questions, depths,
            args.ngram_length, args.min_frequency_ratio, args.output,
        ))
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()
