"""
Independent re-verification of the frozen held-out result, with no models and no network.

Everything the README headlines is recomputed from the committed per-case results, and the
artifact is cross-checked against the split manifest it claims to have been run on:

  - the stored means equal the means recomputed from per_case
  - the stored bootstrap CIs equal CIs recomputed with the recorded seed/resamples
  - the contracts the run used are exactly the manifest's test split, disjoint from dev
  - the manifest on disk is the one the run recorded (SHA-256, line endings normalised)
  - the README quotes the same numbers

    python -m lexis.evaluation.verify_frozen          # exit 1 on any mismatch
"""
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from lexis.config import settings
from lexis.evaluation.stats import bootstrap_ci


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify(root: Path) -> List[Check]:
    report = json.loads((root / "evaluation/reports/cuad_doc_scoped_test.json").read_text(encoding="utf-8"))
    manifest_path = root / "evaluation/splits/cuad_split_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    readme = (root / "README.md").read_text(encoding="utf-8")
    cases = report["per_case"]
    checks: List[Check] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(Check(name, bool(ok), detail))

    add("report is the frozen test baseline", report["status"] == "frozen_test_baseline"
        and report["run_config"]["split_name"] == "test" and report["protocol"] == "doc_scoped")
    add("case count matches", len(cases) == report["num_cases_scored"], f"{len(cases)} vs {report['num_cases_scored']}")

    recall = [c["recall_at_k"] for c in cases]
    mrr = [c["reciprocal_rank"] for c in cases]
    for label, values, stored_mean, key in (("Recall@30", recall, report["mean_recall_at_k"], "recall_at_30"),
                                            ("MRR", mrr, report["mean_reciprocal_rank"], "mrr")):
        recomputed = sum(values) / len(values)
        add(f"{label} mean recomputes", abs(recomputed - stored_mean) < settings.verify_mean_tolerance, f"{recomputed:.6f} vs {stored_mean:.6f}")
        stored_ci = report["bootstrap_ci"][key]
        ci = bootstrap_ci(values, n_resamples=stored_ci["n_resamples"], alpha=stored_ci["alpha"], seed=stored_ci["seed"])
        add(f"{label} bootstrap CI recomputes",
            abs(ci.lo - stored_ci["lo"]) < settings.verify_ci_tolerance and abs(ci.hi - stored_ci["hi"]) < settings.verify_ci_tolerance,
            f"[{ci.lo:.4f}, {ci.hi:.4f}] vs [{stored_ci['lo']:.4f}, {stored_ci['hi']:.4f}]")
        add(f"README quotes {label}", (f"{stored_mean:.3f}" in readme and f"[{stored_ci['lo']:.3f}, {stored_ci['hi']:.3f}]" in readme))

    test_titles = sorted(e["title"] for e in manifest["splits"]["test"])
    dev_titles = {e["title"] for e in manifest["splits"]["dev"]}
    used = sorted(report["run_config"]["contract_titles_used"])
    add("run used exactly the manifest's test contracts", used == test_titles, f"{len(used)} used vs {len(test_titles)}")
    add("dev and test contracts are disjoint", not (set(test_titles) & dev_titles))
    add("test contract count matches manifest", len(test_titles) == manifest["counts"]["test"]["contracts"])

    recorded = report["provenance"]["data_sha256"].get("evaluation/splits/cuad_split_v1.json")
    add("manifest on disk is the one the run recorded", recorded == _sha256_lf(manifest_path),
        "recorded " + str(recorded)[:12] + " vs disk " + _sha256_lf(manifest_path)[:12])
    prov = report["provenance"]
    add("provenance is complete", all(prov.get(k) for k in ("git_sha", "config_hash", "data_sha256", "package_versions")))
    add("no data files were missing at run time", not prov["data_missing"])
    return checks


def main() -> int:
    checks = verify(Path(__file__).resolve().parents[3])
    for c in checks:
        print(f"{'PASS' if c.ok else 'FAIL'}  {c.name}" + (f"  ({c.detail})" if c.detail and not c.ok else ""))
    failed = [c for c in checks if not c.ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
