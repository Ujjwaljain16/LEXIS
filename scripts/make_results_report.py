"""Regenerates the README's benchmark figure and per-category stats from the
frozen held-out result (evaluation/reports/cuad_doc_scoped_test.json).

Pure post-processing of already-frozen per-case results: no retrieval, no
models, no network -- safe to run on a small machine. The headline numbers
are the frozen ones; everything here is DESCRIPTIVE breakdown, never used to
tune anything (the test split is held out).

    python scripts/make_results_report.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lexis.evaluation.stats import bootstrap_ci  # noqa: E402

FROZEN = Path("evaluation/reports/cuad_doc_scoped_test.json")
OUT_FIG = Path("figures/cuad_test_per_category.png")
OUT_JSON = Path("evaluation/reports/cuad_test_breakdown.json")

# CUAD categories whose answer is a document-header fact (title, date, parties)
# rather than a clause -- a post-hoc, descriptive grouping, see README.
FRONT_MATTER = {"Document Name", "Agreement Date", "Effective Date", "Parties"}

BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated pair (scripts/validate_palette.js: ALL PASS)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def ci_dict(values):
    ci = bootstrap_ci(values, n_resamples=10000, alpha=0.05, seed=0)
    return {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "n": ci.n}


def main():
    data = json.loads(FROZEN.read_text(encoding="utf-8"))
    cases = data["per_case"]
    by_cat = defaultdict(list)
    for c in cases:
        by_cat[c["case_id"].rsplit("__", 1)[-1]].append(c)

    rows = []
    for cat, cs in by_cat.items():
        rows.append({
            "category": cat, "n": len(cs), "front_matter": cat in FRONT_MATTER,
            "recall_at_30": sum(x["recall_at_k"] for x in cs) / len(cs),
            "mrr": sum(x["reciprocal_rank"] for x in cs) / len(cs),
        })
    rows.sort(key=lambda r: r["mrr"])

    clause = [c for c in cases if c["case_id"].rsplit("__", 1)[-1] not in FRONT_MATTER]
    front = [c for c in cases if c["case_id"].rsplit("__", 1)[-1] in FRONT_MATTER]
    summary = {
        "headline_frozen": {
            "recall_at_30": ci_dict([c["recall_at_k"] for c in cases]),
            "mrr": ci_dict([c["reciprocal_rank"] for c in cases]),
        },
        "clause_categories": {
            "recall_at_30": ci_dict([c["recall_at_k"] for c in clause]),
            "mrr": ci_dict([c["reciprocal_rank"] for c in clause]),
        },
        "front_matter_categories": {
            "recall_at_30": ci_dict([c["recall_at_k"] for c in front]),
            "mrr": ci_dict([c["reciprocal_rank"] for c in front]),
        },
        "macro_over_categories": {
            "recall_at_30": sum(r["recall_at_30"] for r in rows) / len(rows),
            "mrr": sum(r["mrr"] for r in rows) / len(rows),
            "n_categories": len(rows),
        },
        "per_category": rows,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, (ax_m, ax_r) = plt.subplots(1, 2, figsize=(11.5, 11), facecolor=SURFACE, sharey=True,
                                     gridspec_kw={"wspace": 0.06})
    ys = list(range(len(rows)))
    colors = [ORANGE if r["front_matter"] else BLUE for r in rows]
    for ax, key, title in ((ax_m, "mrr", "MRR"), (ax_r, "recall_at_30", "Recall@30")):
        ax.set_facecolor(SURFACE)
        ax.barh(ys, [r[key] for r in rows], height=0.62, color=colors, zorder=2)
        for y, r in zip(ys, rows):
            ax.text(r[key] + 0.012, y, f"{r[key]:.2f}", va="center", ha="left", fontsize=7.5, color=INK)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="both", length=0, colors=MUTED)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    ax_m.set_yticks(ys)
    ax_m.set_yticklabels([f"{r['category']}  (n={r['n']})" for r in rows], fontsize=8.5, color=INK)

    h = summary["headline_frozen"]
    fig.suptitle(
        "LEXIS on held-out CUAD: retrieval quality by clause type",
        fontsize=12.5, color=INK, x=0.02, ha="left", y=0.992, fontweight="bold",
    )
    fig.text(
        0.02, 0.962,
        f"164 contracts, 1,964 questions | frozen headline: Recall@30 {h['recall_at_30']['mean']:.3f}, "
        f"MRR {h['mrr']['mean']:.3f} | document-scoped protocol",
        fontsize=9.5, color=MUTED, ha="left",
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(handles, ["Clause categories", "Front-matter facts (title / date / parties)"],
               loc="upper left", bbox_to_anchor=(0.02, 0.945), ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.subplots_adjust(left=0.27, right=0.985, top=0.905, bottom=0.04)
    OUT_FIG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160, facecolor=SURFACE)
    print(json.dumps({k: summary[k] for k in ("headline_frozen", "clause_categories",
                                                "front_matter_categories", "macro_over_categories")}, indent=2))


if __name__ == "__main__":
    main()
