"""
Per-phase quality gates (plan defect D7).

plan.md applied ONE final-target dict at every phase, so CI would have failed
for the first ten weeks. Gates are instead DATA (config/gates/<phase>.yaml):
each phase declares its own machine-checkable criteria, the split they are
judged on, and whether they are still `provisional` (thresholds not yet
re-set from the re-baselined test split) or `frozen`.

Criterion types (all statistics-aware -- none compares bare point estimates
when an interval is available):
  absolute_min   metric[use] >= threshold          (use = mean | lo | hi | width)
  absolute_max   metric[use] <= threshold
  relative_gain  paired vs baseline: CI excludes 0 on the positive side,
                 p < alpha, and mean_diff >= min_effect. Uses the Holm-
                 adjusted p-value when the results provide one.
  no_regression  paired vs baseline: CI lower bound > -tolerance
                 (non-inferiority: "not significantly worse")

Results shape (built from run_stats_report.py output by
results_from_stats_report, or by any evaluator):
  {"metrics": {name: {"mean","lo","hi"}},
   "paired":  {name: {"mean_diff","ci_lo","ci_hi","p_value"[, "p_value_holm"]}}}
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

CriterionType = Literal["absolute_min", "absolute_max", "relative_gain", "no_regression"]
Use = Literal["mean", "lo", "hi", "width"]


class Criterion(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    metric: str = Field(min_length=1)
    type: CriterionType
    threshold: Optional[float] = None
    use: Use = "mean"
    min_effect: float = Field(default=0.0, ge=0.0)
    tolerance: float = Field(default=0.0, ge=0.0)
    description: str = ""

    @model_validator(mode="after")
    def _threshold_rules(self) -> "Criterion":
        needs = self.type in ("absolute_min", "absolute_max")
        if needs and self.threshold is None:
            raise ValueError(f"criterion {self.id!r} of type {self.type} requires a threshold")
        if not needs and self.threshold is not None:
            raise ValueError(f"criterion {self.id!r} of type {self.type} must not set a threshold")
        return self


class Gate(BaseModel):
    phase: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = ""
    split: Literal["dev", "test"]
    status: Literal["provisional", "frozen"]
    criteria: List[Criterion] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "Gate":
        ids = [c.id for c in self.criteria]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate criterion ids: {dupes}")
        return self


@dataclass(frozen=True)
class CriterionResult:
    id: str
    metric: str
    type: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateReport:
    phase: str
    split: str
    status: str
    results: List[CriterionResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def provisional(self) -> bool:
        return self.status == "provisional"

    def blocks(self, enforce_provisional: bool = False) -> bool:
        """Whether this report should fail CI: a failed frozen gate always
        does; a failed provisional gate only when explicitly enforced."""
        return (not self.passed) and (enforce_provisional or not self.provisional)


def load_gate(path: Path) -> Gate:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    gate = Gate(**data)
    if gate.phase != Path(path).stem:
        raise ValueError(f"{path}: phase {gate.phase!r} must equal the file name {Path(path).stem!r}")
    return gate


def discover_gates(directory: Path) -> Dict[str, Gate]:
    return {g.phase: g for g in (load_gate(p) for p in sorted(Path(directory).glob("*.yaml")))}


def _point(entry: Mapping[str, float], use: str) -> Optional[float]:
    if use == "width":
        if "lo" not in entry or "hi" not in entry:
            return None
        return entry["hi"] - entry["lo"]
    return entry.get(use)


def _check(c: Criterion, results: Mapping[str, Any], alpha: float) -> CriterionResult:
    def res(passed: bool, detail: str) -> CriterionResult:
        return CriterionResult(c.id, c.metric, c.type, passed, detail)

    if c.type in ("absolute_min", "absolute_max"):
        entry = results.get("metrics", {}).get(c.metric)
        value = _point(entry, c.use) if entry else None
        if value is None:
            return res(False, f"missing metric {c.metric!r} ({c.use})")
        ok = value >= c.threshold if c.type == "absolute_min" else value <= c.threshold
        op = ">=" if c.type == "absolute_min" else "<="
        return res(ok, f"{c.metric}[{c.use}]={value:.4f} {op} {c.threshold} -> {'ok' if ok else 'FAIL'}")

    entry = results.get("paired", {}).get(c.metric)
    if not entry or "ci_lo" not in entry:
        return res(False, f"missing paired comparison for {c.metric!r}")
    if c.type == "relative_gain":
        p = entry.get("p_value_holm", entry.get("p_value"))
        if p is None:
            return res(False, f"paired comparison for {c.metric!r} has no p-value")
        ok = entry["ci_lo"] > 0 and p < alpha and entry.get("mean_diff", 0.0) >= c.min_effect
        return res(ok, f"diff={entry.get('mean_diff', float('nan')):+.4f} CI_lo={entry['ci_lo']:+.4f} "
                       f"p={p:.4f} (alpha {alpha}, min_effect {c.min_effect}) -> {'ok' if ok else 'FAIL'}")
    ok = entry["ci_lo"] > -c.tolerance
    return res(ok, f"CI_lo={entry['ci_lo']:+.4f} > -{c.tolerance} -> {'ok' if ok else 'FAIL'}")


def evaluate_gate(gate: Gate, results: Mapping[str, Any], alpha: float) -> GateReport:
    return GateReport(gate.phase, gate.split, gate.status, [_check(c, results, alpha) for c in gate.criteria])


def results_from_stats_report(report: Mapping[str, Any], metric: str, column: str) -> Dict[str, Any]:
    """Adapt evaluation/run_stats_report.py output for one alternative column
    into the gate results shape, naming it `metric`."""
    comp = next((c for c in report["comparisons"] if c["column"] == column), None)
    if comp is None:
        raise KeyError(f"column {column!r} not in report comparisons")
    alt = comp["alt_ci"]
    paired = {k: comp[k] for k in ("mean_diff", "ci_lo", "ci_hi", "p_value") if k in comp}
    if "p_value_holm" in comp:
        paired["p_value_holm"] = comp["p_value_holm"]
    return {"metrics": {metric: {"mean": alt["mean"], "lo": alt["lo"], "hi": alt["hi"]}}, "paired": {metric: paired}}


def merge_results(*parts: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"metrics": {}, "paired": {}}
    for part in parts:
        out["metrics"].update(part.get("metrics", {}))
        out["paired"].update(part.get("paired", {}))
    return out
