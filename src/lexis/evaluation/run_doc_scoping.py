"""
CLI: document-scoping experiment (plan R6).  Measurement only.

The oracle diagnostic showed 89% of real-query rank-1 misses are chunks from
a DIFFERENT contract. How much would restricting retrieval to one document
recover, and how far can a realistic router get?

Representations (case-level, each case's full relevant set, as the baseline):
  existing        unscoped retrieval. CONTROL: must reproduce the frozen
                  baseline artifact exactly or the run aborts.
  scoped_all      scope = every indexed document. CONTROL: must reproduce
                  `existing` per case (proves the scoping code adds no
                  effect of its own) or the run aborts.
  oracle_doc      scope = the case's true document. An UPPER BOUND on what a
                  router could deliver (uses ground truth; never a product).
  router_vote     scope = the document chosen by a rank-weighted vote over the
                  unscoped result list (doc_routing.vote_document). Uses NO
                  ground truth. The realistic first router.
Reported: Recall/MRR with bootstrap CIs, paired comparisons vs existing (Holm
corrected), router document accuracy, and what still blocks rank 1.

Also a product framing, kept separate: oracle_doc is what a USER-SELECTED
document filter delivers; router_vote is what AUTOMATIC routing delivers.
Retrieval calls are guarded against silent dense/BM25 timeouts (retried,
counted in the artifact). Only evaluation-layer code runs; RetrievalEngine
is not modified and a settings guard raises if production settings change.
"""
import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from lexis.config import settings
from lexis.evaluation.doc_routing import vote_document
from lexis.evaluation.frozen_scope import load_frozen_scope
from lexis.evaluation.oracle_case_level import CaseOracleAnalysis, analyze_ranked_case, call_until_healthy, summarize_blockers
from lexis.evaluation.provenance import build_provenance
from lexis.evaluation.scoped_retrieval import retrieve_scoped
from lexis.evaluation.stats import bootstrap_ci, holm_correction, paired_comparison
from lexis.registry.layered_config import load_yaml
from lexis.retrieval.hybrid_retriever import RetrievalEngine

REPS = ("existing", "scoped_all", "oracle_doc", "router_vote")


def _ranked(trace) -> List[dict]:
    return [{"id": c["id"], "doc_id": (c.get("payload") or {}).get("doc_id"), "text": c.get("text", "")}
            for c in trace.final_chunks]


def _fingerprint():
    return (settings.retrieval_top_k_per_path, settings.retrieval_fusion_top_k, settings.rrf_k,
            settings.qdrant_collection_primary, settings.embedding_model)


def _ci(values, n, alpha, seed):
    ci = bootstrap_ci(values, n, alpha, seed)
    return {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "n": ci.n}


async def run(args) -> dict:
    cfg = load_yaml(Path(args.config_dir) / "defaults.yaml")
    frozen, ev = cfg["frozen_baseline"], cfg["eval"]
    threshold = cfg["diagnostics"]["answer_overlap_threshold"]
    max_retries = cfg["diagnostics"]["max_call_retries"]
    depths = sorted({int(d) for d in args.depths.split(",")}) if args.depths else sorted(ev["depths"])
    first_depth = depths[0]

    baseline = json.loads(Path(frozen["report"]).read_text(encoding="utf-8"))
    cases, chunks_by_doc, _ = load_frozen_scope(frozen, args.cuad_path)
    all_doc_ids = sorted(chunks_by_doc)
    print(f"cases: {len(cases)}, documents: {len(all_doc_ids)}")

    engine = RetrievalEngine()
    before = _fingerprint()
    health = {"calls": 0, "retried_calls": 0, "extra_attempts": 0, "unrecovered_calls": 0}

    async def guarded(make_call):
        out = await call_until_healthy(make_call, lambda t: bool(t.dense_candidates) and bool(t.bm25_candidates), max_retries)
        health["calls"] += 1
        health["extra_attempts"] += out.attempts - 1
        health["retried_calls"] += 1 if out.attempts > 1 else 0
        health["unrecovered_calls"] += 0 if out.healthy else 1
        return out.result

    analyses: Dict[str, Dict[int, Dict[str, CaseOracleAnalysis]]] = {r: {d: {} for d in depths} for r in REPS}
    router_correct: Dict[int, List[float]] = {d: [] for d in depths}
    unscoped_ranked: Dict[int, Dict[str, list]] = {d: {} for d in depths}

    for case in cases:
        answers = list(case.metadata.get("answer_texts", []))
        relevant, doc_ids = list(case.relevant_chunk_ids), list(case.relevant_doc_ids)
        for depth in depths:
            def analyze(trace):
                return analyze_ranked_case(_ranked(trace), relevant, doc_ids, answers, threshold, depth)

            async def scoped(scope):
                return await guarded(lambda: retrieve_scoped(engine, case.query, scope, depth, depth))

            base_trace = await guarded(lambda: engine.retrieve_with_trace(case.query, top_k_per_path=depth, top_n_rrf=depth))
            unscoped_ranked[depth][case.case_id] = _ranked(base_trace)
            analyses["existing"][depth][case.case_id] = analyze(base_trace)
            analyses["scoped_all"][depth][case.case_id] = analyze(await scoped(all_doc_ids))
            analyses["oracle_doc"][depth][case.case_id] = analyze(await scoped(doc_ids))

            routed = vote_document(unscoped_ranked[depth][case.case_id], settings.rrf_k)
            router_correct[depth].append(1.0 if routed in doc_ids else 0.0)
            analyses["router_vote"][depth][case.case_id] = analyze(await scoped([routed] if routed else all_doc_ids))

    if _fingerprint() != before:
        raise RuntimeError("production Settings changed during the run; refusing to write a report")

    # ---- controls ----
    ex = analyses["existing"][first_depth]
    recall0 = sum(a.recall for a in ex.values()) / len(ex)
    mrr0 = sum(a.reciprocal_rank for a in ex.values()) / len(ex)
    if first_depth == baseline["run_config"]["top_k"] and (
            recall0 != baseline["mean_recall_at_k"] or mrr0 != baseline["mean_reciprocal_rank"]):
        raise RuntimeError(f"BASELINE CONTROL FAILED: recall={recall0!r} mrr={mrr0!r}")
    mismatched = [cid for d in depths for cid, a in analyses["existing"][d].items()
                  if (a.recall, a.reciprocal_rank) != (analyses["scoped_all"][d][cid].recall, analyses["scoped_all"][d][cid].reciprocal_rank)]
    if mismatched:
        raise RuntimeError(f"SCOPING CONTROL FAILED: scoped_all differs from existing for {len(mismatched)} (case,depth) pairs")

    n, alpha, seed = ev["bootstrap_resamples"], ev["alpha"], ev["seed"]
    summary, paired_rows = {}, []
    for rep in REPS:
        summary[rep] = {}
        for d in depths:
            vals = list(analyses[rep][d].values())
            summary[rep][str(d)] = {
                "recall": _ci([a.recall for a in vals], n, alpha, seed),
                "mrr": _ci([a.reciprocal_rank for a in vals], n, alpha, seed),
                "share_rank1_relevant": sum(a.top1_is_relevant for a in vals) / len(vals),
            }
    for rep in ("oracle_doc", "router_vote"):
        for d in depths:
            for metric in ("recall", "reciprocal_rank"):
                a = [getattr(analyses[rep][d][c.case_id], metric) for c in cases]
                b = [getattr(analyses["existing"][d][c.case_id], metric) for c in cases]
                r = paired_comparison(a, b, n, alpha, seed)
                paired_rows.append({"rep": rep, "depth": d, "metric": metric, **vars(r)})
    for row, p in zip(paired_rows, holm_correction([r["p_value"] for r in paired_rows])):
        row["p_value_holm"] = p
        row["supported_holm"] = row["ci_lo"] > 0 and p < alpha

    router_acc = {str(d): _ci(v, n, alpha, seed) for d, v in router_correct.items()}
    blockers = {rep: summarize_blockers(analyses[rep][first_depth].values()) for rep in REPS}

    out = {
        "diagnostic_type": "document_scoping_experiment",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "depths": depths, "n_cases": len(cases), "n_documents": len(all_doc_ids),
        "retrieval_call_health": health,
        "controls": {"baseline_recall": recall0, "baseline_mrr": mrr0, "scoped_all_matches_existing": True},
        "case_level": summary, "paired_vs_existing": paired_rows,
        "router_document_accuracy": router_acc, "top1_blockers_at_first_depth": {"depth": first_depth, **blockers},
        "per_case": [{"case_id": c.case_id, **{f"{rep}_d{d}": {"rr": analyses[rep][d][c.case_id].reciprocal_rank,
                                                                "recall": analyses[rep][d][c.case_id].recall}
                                               for rep in REPS for d in depths}} for c in cases],
        "provenance": build_provenance(config={"depths": depths, "frozen_baseline": frozen},
                                       data_paths=[args.cuad_path, frozen["report"]], packages=["numpy"]),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\ncall health: {health}")
    print(f"controls OK: baseline reproduced (recall={recall0:.4f}, mrr={mrr0:.4f}); scoped_all == existing")
    print(f"{'representation':<14}{'depth':>6}  {'Recall (95% CI)':<26}{'MRR (95% CI)':<26}rank1-relevant")
    for rep in REPS:
        for d in depths:
            s = summary[rep][str(d)]
            r, m = s["recall"], s["mrr"]
            print(f"{rep:<14}{d:>6}  {r['mean']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]      "
                  f"{m['mean']:.3f} [{m['lo']:.3f},{m['hi']:.3f}]      {s['share_rank1_relevant']:.2f}")
    print("router document accuracy:", {d: round(v["mean"], 3) for d, v in router_acc.items()})
    print("paired vs existing (Holm):")
    for r in paired_rows:
        print(f"  {r['rep']:<12} d={r['depth']:<4} {r['metric']:<16} diff={r['mean_diff']:+.3f} "
              f"CI[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] holm_p={r['p_value_holm']:.4f} supported={r['supported_holm']}")
    print(f"blockers at depth {first_depth}:")
    for rep in REPS:
        print(f"  {rep}: {blockers[rep]}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Document-scoping experiment (measurement only).")
    ap.add_argument("--cuad-path", default="data/cuad_raw/CUAD_v1/CUAD_v1.json")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--depths", default=None)
    ap.add_argument("--output", default="evaluation/reports/cuad_doc_scoping.json")
    asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
