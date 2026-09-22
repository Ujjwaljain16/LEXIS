"""
CLI: evaluate a phase gate against a results file. Exit status is CI-ready.

    python -m lexis.evaluation.run_gate_check --gate config/gates/p1_recall.yaml --results results.json

Exit 0 = pass (or a failed *provisional* gate, which is reported but
advisory); exit 1 = a frozen gate failed, or --enforce-provisional was given
and a provisional gate failed.
"""
import argparse
import json
from pathlib import Path

from lexis.evaluation.gates import evaluate_gate, load_gate
from lexis.registry.layered_config import load_yaml


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a phase gate against results.")
    ap.add_argument("--gate", required=True)
    ap.add_argument("--results", required=True, help="JSON with 'metrics' and 'paired' sections")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--enforce-provisional", action="store_true")
    args = ap.parse_args(argv)

    alpha = load_yaml(Path(args.config_dir) / "defaults.yaml")["eval"]["alpha"]
    gate = load_gate(Path(args.gate))
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    report = evaluate_gate(gate, results, alpha)

    print(f"gate {report.phase} (split={report.split}, status={report.status}): {'PASS' if report.passed else 'FAIL'}")
    for r in report.results:
        print(f"  [{'x' if r.passed else ' '}] {r.id}: {r.detail}")
    if report.provisional and not report.passed:
        print("  note: gate is provisional (thresholds not yet frozen) - advisory unless --enforce-provisional")
    return 1 if report.blocks(args.enforce_provisional) else 0


if __name__ == "__main__":
    raise SystemExit(main())
