"""
CLI: confidence intervals + paired significance for any per-case results
table (plan P0). Benchmark-agnostic: it only needs a JSON file containing a
list of row dicts, a baseline column, and one or more alternative columns of
per-query scores.

Example (Step 12 artifact):
    python -m lexis.evaluation.run_stats_report \\
        --input evaluation/reports/cuad_query_reformulation_experiment.json \\
        --rows-key per_case_deltas_at_depth_30 \\
        --baseline-col original_recall_at_30 \\
        --alt-cols normalized_recall_at_30,boilerplate_reduced_recall_at_30,multi_representation_recall_at_30 \\
        --output evaluation/reports/stats_step12_recall30.json
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

from lexis.evaluation.provenance import build_provenance
from lexis.evaluation.stats import bootstrap_ci, claim_supported, holm_correction, paired_comparison


def build_stats_report(rows: List[dict], baseline_col: str, alt_cols: List[str],
                       n_resamples: int, alpha: float, seed: int) -> Dict:
    missing = [c for c in [baseline_col, *alt_cols] if any(c not in r for r in rows)]
    if missing:
        raise KeyError(f"columns missing from some rows: {missing}")
    base = [float(r[baseline_col]) for r in rows]
    report = {
        "n_queries": len(rows),
        "baseline": {"column": baseline_col, **vars(bootstrap_ci(base, n_resamples, alpha, seed))},
        "comparisons": [],
    }
    results = [(c, paired_comparison([float(r[c]) for r in rows], base, n_resamples, alpha, seed)) for c in alt_cols]
    adjusted = holm_correction([r.p_value for _, r in results])
    for (col, res), p_adj in zip(results, adjusted):
        report["comparisons"].append({
            "column": col, **vars(res), "p_value_holm": p_adj,
            "claim_supported_uncorrected": claim_supported(res),
            "claim_supported_holm": res.ci_lo > 0 and p_adj < alpha,
            "alt_ci": vars(bootstrap_ci([float(r[col]) for r in rows], n_resamples, alpha, seed)),
        })
    return report


def main():
    ap = argparse.ArgumentParser(description="CIs and paired tests over a per-case results table.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--rows-key", required=True, help="JSON key holding the list of per-case row dicts")
    ap.add_argument("--baseline-col", required=True)
    ap.add_argument("--alt-cols", required=True, help="comma-separated columns to compare against the baseline")
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = data[args.rows_key]
    alt_cols = [c.strip() for c in args.alt_cols.split(",") if c.strip()]
    report = build_stats_report(rows, args.baseline_col, alt_cols, args.n_resamples, args.alpha, args.seed)
    report["provenance"] = build_provenance(
        config=vars(args), data_paths=[args.input], packages=["numpy"],
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    b = report["baseline"]
    print(f"baseline {b['column']}: mean={b['mean']:.4f}  95% CI [{b['lo']:.4f}, {b['hi']:.4f}]  n={report['n_queries']}")
    for c in report["comparisons"]:
        print(f"{c['column']}: diff={c['mean_diff']:+.4f} CI [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] "
              f"p={c['p_value']:.4f} (holm {c['p_value_holm']:.4f}) improved/regressed/tied="
              f"{c['n_improved']}/{c['n_regressed']}/{c['n_tied']} supported(holm)={c['claim_supported_holm']}")


if __name__ == "__main__":
    main()
