"""
CLI: case-level ORACLE MRR diagnostic (plan P0).  oracle_query_diagnostic.

Question: on our strict chunk-level labels, how high can case-level MRR go
when the query is the answer itself, and what blocks it when it stops short?
Measurement only -- the oracle query is ground truth and never a production
query; results are not a baseline or an improvement.

Representations (all evaluated CASE-LEVEL against each case's FULL relevant
set, the same aggregation as the frozen baseline):
  existing            case.query. A CONTROL: at the baseline depth it must
                      reproduce the frozen baseline artifact exactly or the
                      run aborts.
  oracle_joined       ONE query per case: all of the case's answer texts
                      joined in their given order.
  oracle_best_chunk   for each relevant chunk, a query from the answer text(s)
                      matched to that chunk; the case takes its BEST result
                      over those queries (an upper bound, not a realistic
                      system).
Also reported: the per-(case, chunk) singleton oracle MRR (the Step 11
aggregation) as a cross-check, bootstrap 95% CIs for every case-level mean,
and -- at the first depth -- what sits ABOVE the first relevant chunk when it
is not ranked first (see oracle_case_level.py).

Uses only retrieve_with_trace(); a settings snapshot guard raises if
production settings change.

    python -m lexis.evaluation.run_oracle_case_level --depths 30,100
"""
import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from lexis.config import settings
from lexis.evaluation.dataset.cuad_loader import CUADAdapter, CUADLoader, select_contracts
from lexis.evaluation.dataset.mapping import build_answer_derived_query, deterministic_document_id
from lexis.evaluation.metrics import reciprocal_rank
from lexis.evaluation.oracle_case_level import (
    CaseOracleAnalysis, analyze_ranked_case, best_of, call_until_healthy, summarize_blockers,
)
from lexis.evaluation.provenance import build_provenance
from lexis.evaluation.run_eval import chunk_document_text
from lexis.evaluation.stats import bootstrap_ci
from lexis.indexing.schema import Chunk
from lexis.ingestion.chunker import SemanticChunker
from lexis.ingestion.embedder import BGEM3Embedder
from lexis.ingestion.parser import LexisParser
from lexis.registry.layered_config import load_yaml
from lexis.retrieval.hybrid_retriever import RetrievalEngine

DIAGNOSTIC_TYPE = "oracle_query_diagnostic"


def _ranked(trace) -> List[dict]:
    return [{"id": c["id"], "doc_id": (c.get("payload") or {}).get("doc_id"), "text": c.get("text", "")}
            for c in trace.final_chunks]


def _settings_fingerprint():
    return (settings.retrieval_top_k_per_path, settings.retrieval_fusion_top_k, settings.rrf_k,
            settings.qdrant_collection_primary, settings.embedding_model)


def _ci(values: List[float], n_resamples: int, alpha: float, seed: int) -> dict:
    ci = bootstrap_ci(values, n_resamples, alpha, seed)
    return {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "n": ci.n}


async def run(args) -> dict:
    cfg = load_yaml(Path(args.config_dir) / "defaults.yaml")
    frozen = cfg["frozen_baseline"]
    threshold = cfg["diagnostics"]["answer_overlap_threshold"]
    ev = cfg["eval"]
    depths = sorted({int(d) for d in args.depths.split(",")}) if args.depths else sorted(ev["depths"])
    first_depth = depths[0]

    baseline = json.loads(Path(frozen["report"]).read_text(encoding="utf-8"))
    raw = CUADLoader().load(args.cuad_path)
    contracts = select_contracts(raw, frozen["contracts"])

    parser, embedder = LexisParser(), BGEM3Embedder()
    chunker = SemanticChunker(embedder=embedder)
    chunks_by_doc: Dict[str, List[Chunk]] = {}
    chunk_by_id: Dict[str, Chunk] = {}
    for contract in contracts:
        doc_id = deterministic_document_id(contract["title"], prefix="cuad")
        chunks = chunk_document_text(parser, chunker, contract["paragraphs"][0]["context"], doc_id)
        chunks_by_doc[doc_id] = chunks
        chunk_by_id.update({c.chunk_id: c for c in chunks})

    cases, _ = CUADAdapter().build_cases(raw, chunks_by_doc, contracts, frozen["questions"])
    cases = [c for c in cases if c.has_chunk_level_ground_truth()]
    print(f"cases: {len(cases)}, relevant chunk pairs: {sum(len(c.relevant_chunk_ids) for c in cases)}")

    retriever = RetrievalEngine()
    before = _settings_fingerprint()

    call_stats = {"calls": 0, "retried_calls": 0, "extra_attempts": 0, "unrecovered_calls": 0}
    max_retries = cfg["diagnostics"]["max_call_retries"]

    async def trace_for(query: str, depth: int):
        outcome = await call_until_healthy(
            lambda: retriever.retrieve_with_trace(query, top_k_per_path=depth, top_n_rrf=depth),
            lambda t: bool(t.dense_candidates) and bool(t.bm25_candidates),
            max_retries,
        )
        call_stats["calls"] += 1
        call_stats["extra_attempts"] += outcome.attempts - 1
        call_stats["retried_calls"] += 1 if outcome.attempts > 1 else 0
        call_stats["unrecovered_calls"] += 0 if outcome.healthy else 1
        return outcome.result

    # analyses[rep][depth][case_id] -> CaseOracleAnalysis
    analyses: Dict[str, Dict[int, Dict[str, CaseOracleAnalysis]]] = {
        r: {d: {} for d in depths} for r in ("existing", "oracle_joined", "oracle_best_chunk")}
    singleton_rr: Dict[int, List[float]] = {d: [] for d in depths}

    for case in cases:
        answers = list(case.metadata.get("answer_texts", []))
        relevant, doc_ids = list(case.relevant_chunk_ids), list(case.relevant_doc_ids)
        joined_query = " ".join(answers)
        chunk_queries = []
        for cid in sorted(relevant):
            q = build_answer_derived_query(answers, chunk_by_id[cid])
            if q is not None:
                chunk_queries.append((cid, q))

        for depth in depths:
            def analyze(trace):
                return analyze_ranked_case(_ranked(trace), relevant, doc_ids, answers, threshold, depth)

            analyses["existing"][depth][case.case_id] = analyze(await trace_for(case.query, depth))
            analyses["oracle_joined"][depth][case.case_id] = analyze(await trace_for(joined_query, depth))
            per_chunk = []
            for cid, q in chunk_queries:
                trace = await trace_for(q, depth)
                per_chunk.append(analyze(trace))
                singleton_rr[depth].append(reciprocal_rank([cid], [c["id"] for c in trace.final_chunks]))
            analyses["oracle_best_chunk"][depth][case.case_id] = best_of(per_chunk)

    if _settings_fingerprint() != before:
        raise RuntimeError("production Settings changed during the oracle run; refusing to write a report")

    # ---- control: existing must reproduce the frozen baseline exactly ----
    ex = analyses["existing"][first_depth]
    ctrl_recall = sum(a.recall for a in ex.values()) / len(ex)
    ctrl_mrr = sum(a.reciprocal_rank for a in ex.values()) / len(ex)
    tol = float(args.control_tolerance) if args.control_tolerance is not None else 0.0
    if first_depth == baseline["run_config"]["top_k"] and (
            abs(ctrl_recall - baseline["mean_recall_at_k"]) > tol or abs(ctrl_mrr - baseline["mean_reciprocal_rank"]) > tol):
        raise RuntimeError(f"CONTROL FAILED: got recall={ctrl_recall!r}, mrr={ctrl_mrr!r}; "
                           f"baseline is {baseline['mean_recall_at_k']!r}, {baseline['mean_reciprocal_rank']!r}")

    n_res, alpha, seed = ev["bootstrap_resamples"], ev["alpha"], ev["seed"]
    summary = {}
    for rep, by_depth in analyses.items():
        summary[rep] = {}
        for depth, by_case in by_depth.items():
            vals = list(by_case.values())
            summary[rep][str(depth)] = {
                "recall": _ci([a.recall for a in vals], n_res, alpha, seed),
                "mrr": _ci([a.reciprocal_rank for a in vals], n_res, alpha, seed),
                "share_first_relevant_at_rank_1": sum(a.top1_is_relevant for a in vals) / len(vals),
            }
    singleton = {str(d): _ci(v, n_res, alpha, seed) for d, v in singleton_rr.items() if v}
    blockers = {rep: summarize_blockers(analyses[rep][first_depth].values()) for rep in analyses}

    per_case = [{
        "case_id": c.case_id, "relevant_count": len(c.relevant_chunk_ids),
        **{f"{rep}_d{depth}": {"rr": analyses[rep][depth][c.case_id].reciprocal_rank,
                               "recall": analyses[rep][depth][c.case_id].recall,
                               "first_relevant_rank": analyses[rep][depth][c.case_id].first_relevant_rank,
                               "blocker": analyses[rep][depth][c.case_id].blocker}
           for rep in analyses for depth in depths},
    } for c in cases]

    out = {
        "diagnostic_type": DIAGNOSTIC_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "depths": depths, "answer_overlap_threshold": threshold,
        "n_cases": len(cases),
        "retrieval_call_health": call_stats,
        "control": {"recall": ctrl_recall, "mrr": ctrl_mrr,
                    "baseline_recall": baseline["mean_recall_at_k"], "baseline_mrr": baseline["mean_reciprocal_rank"]},
        "case_level": summary,
        "singleton_oracle_mrr": singleton,
        "top1_blockers_at_first_depth": {"depth": first_depth, **blockers},
        "per_case": per_case,
        "provenance": build_provenance(config={"depths": depths, "frozen_baseline": frozen, "threshold": threshold},
                                       data_paths=[args.cuad_path, frozen["report"]], packages=["numpy"]),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nretrieval call health: {call_stats}")
    print(f"[{DIAGNOSTIC_TYPE}] control (existing @ {first_depth}): recall={ctrl_recall:.4f} mrr={ctrl_mrr:.4f}")
    print(f"{'representation':<20}{'depth':>6}  {'Recall (95% CI)':<26}{'MRR (95% CI)':<26}top1-relevant")
    for rep in summary:
        for depth in depths:
            s = summary[rep][str(depth)]
            r, m = s["recall"], s["mrr"]
            print(f"{rep:<20}{depth:>6}  {r['mean']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]      "
                  f"{m['mean']:.3f} [{m['lo']:.3f},{m['hi']:.3f}]      {s['share_first_relevant_at_rank_1']:.2f}")
    print("singleton (case,chunk) oracle MRR:", {d: round(v["mean"], 3) for d, v in singleton.items()})
    print(f"\nwhat blocks rank 1 (depth {first_depth}):")
    for rep, b in blockers.items():
        print(f"  {rep}: {b}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Case-level oracle MRR diagnostic (measurement only).")
    ap.add_argument("--cuad-path", default="data/cuad_raw/CUAD_v1/CUAD_v1.json")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--depths", default=None, help="comma-separated; default from config eval.depths")
    ap.add_argument("--control-tolerance", default=None, help="abs tolerance for the baseline control (default exact)")
    ap.add_argument("--output", default="evaluation/reports/cuad_oracle_case_level.json")
    asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
