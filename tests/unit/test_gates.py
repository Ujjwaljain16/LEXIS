from pathlib import Path

import pytest
from pydantic import ValidationError

from lexis.evaluation.gates import (
    Criterion,
    Gate,
    discover_gates,
    evaluate_gate,
    load_gate,
    merge_results,
    results_from_stats_report,
)

ALPHA = 0.05
ROOT = Path(__file__).resolve().parents[2]


def gate(*criteria, status="frozen"):
    return Gate(phase="p_test", split="test", status=status, criteria=list(criteria))


def crit(**kw):
    base = dict(id="c1", metric="m", type="absolute_min", threshold=0.5)
    base.update(kw)
    return Criterion(**base)


def metrics(mean=0.6, lo=0.55, hi=0.65):
    return {"metrics": {"m": {"mean": mean, "lo": lo, "hi": hi}}}


def paired(mean_diff=0.05, ci_lo=0.01, ci_hi=0.09, p=0.01, **extra):
    return {"paired": {"m": {"mean_diff": mean_diff, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_value": p, **extra}}}


def run(criterion, results, status="frozen"):
    return evaluate_gate(gate(criterion, status=status), results, ALPHA)


# --- absolute criteria ---

def test_absolute_min_passes_and_fails_on_the_chosen_point():
    assert run(crit(use="mean", threshold=0.6), metrics(mean=0.6)).passed
    assert not run(crit(use="lo", threshold=0.6), metrics(lo=0.55)).passed  # conservative: CI lower bound


def test_absolute_max_uses_upper_bound():
    c = crit(type="absolute_max", use="hi", threshold=0.17)
    assert run(c, metrics(hi=0.15)).passed
    assert not run(c, metrics(mean=0.10, hi=0.20)).passed


def test_width_criterion():
    c = crit(type="absolute_max", use="width", threshold=0.06)
    assert run(c, metrics(lo=0.50, hi=0.55)).passed
    assert not run(c, metrics(lo=0.40, hi=0.60)).passed


def test_missing_metric_fails_with_a_clear_reason():
    r = run(crit(), {"metrics": {}})
    assert not r.passed and "missing metric" in r.results[0].detail


def test_width_without_interval_fails_rather_than_passes():
    assert not run(crit(type="absolute_max", use="width", threshold=1.0), {"metrics": {"m": {"mean": 0.5}}}).passed


# --- relative_gain ---

def rg(**kw):
    return crit(type="relative_gain", threshold=None, **kw)


def test_relative_gain_requires_ci_and_p_and_effect():
    assert run(rg(), paired()).passed
    assert not run(rg(), paired(ci_lo=-0.01)).passed          # CI spans zero
    assert not run(rg(), paired(p=0.2)).passed                  # not significant
    assert not run(rg(min_effect=0.1), paired(mean_diff=0.05)).passed  # too small


def test_relative_gain_prefers_holm_adjusted_p_when_present():
    assert not run(rg(), paired(p=0.01, p_value_holm=0.2)).passed
    assert run(rg(), paired(p=0.2, p_value_holm=0.01)).passed


def test_relative_gain_missing_comparison_fails():
    assert not run(rg(), {"paired": {}}).passed


# --- no_regression ---

def nr(**kw):
    return crit(type="no_regression", threshold=None, **kw)


def test_no_regression_is_non_inferiority_with_tolerance():
    assert run(nr(tolerance=0.02), paired(ci_lo=-0.01)).passed
    assert not run(nr(tolerance=0.02), paired(ci_lo=-0.05)).passed
    assert not run(nr(), paired(ci_lo=-0.001)).passed  # zero tolerance: any significant drop fails


# --- schema validation ---

def test_absolute_criterion_requires_threshold():
    with pytest.raises(ValidationError):
        Criterion(id="c", metric="m", type="absolute_min")


def test_relative_criterion_forbids_threshold():
    with pytest.raises(ValidationError):
        Criterion(id="c", metric="m", type="relative_gain", threshold=0.1)


def test_duplicate_criterion_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        gate(crit(), crit())


def test_gate_needs_at_least_one_criterion():
    with pytest.raises(ValidationError):
        Gate(phase="p", split="test", status="frozen", criteria=[])


# --- provisional vs frozen enforcement (defect D7) ---

def test_failed_frozen_gate_blocks_but_failed_provisional_is_advisory():
    failing = metrics(mean=0.1)
    frozen = run(crit(threshold=0.9), failing, status="frozen")
    provisional = run(crit(threshold=0.9), failing, status="provisional")
    assert frozen.blocks() and not provisional.blocks()
    assert provisional.blocks(enforce_provisional=True)


def test_passing_gate_never_blocks():
    assert not run(crit(threshold=0.1), metrics(), status="frozen").blocks()


# --- adapting stats-report output ---

def stats_report():
    return {"comparisons": [{
        "column": "alt", "mean_diff": 0.05, "ci_lo": 0.01, "ci_hi": 0.09, "p_value": 0.01, "p_value_holm": 0.02,
        "alt_ci": {"mean": 0.6, "lo": 0.55, "hi": 0.65},
    }]}


def test_results_from_stats_report_maps_metric_and_paired_fields():
    r = results_from_stats_report(stats_report(), "m", "alt")
    assert r["metrics"]["m"] == {"mean": 0.6, "lo": 0.55, "hi": 0.65}
    assert r["paired"]["m"]["p_value_holm"] == 0.02
    assert evaluate_gate(gate(rg(id="gain"), crit(id="floor", use="lo", threshold=0.5)), r, ALPHA).passed


def test_results_from_stats_report_unknown_column():
    with pytest.raises(KeyError):
        results_from_stats_report(stats_report(), "m", "nope")


def test_merge_results_combines_sections():
    a = {"metrics": {"x": {"mean": 1}}, "paired": {}}
    b = {"metrics": {"y": {"mean": 2}}, "paired": {"y": {"ci_lo": 0}}}
    merged = merge_results(a, b)
    assert set(merged["metrics"]) == {"x", "y"} and "y" in merged["paired"]


# --- the shipped gate files ---

GATES = discover_gates(ROOT / "config" / "gates")


def test_all_phases_have_a_gate():
    assert set(GATES) == {"p0_foundations", "p1_recall", "p2_ranking", "p3_trust",
                          "p4_jurisdictions", "p5_production", "p6_release"}


@pytest.mark.parametrize("phase", sorted(GATES))
def test_shipped_gates_are_judged_on_the_frozen_test_split(phase):
    assert GATES[phase].split == "test"


def test_no_shipped_gate_is_frozen_before_targets_are_reset_from_data():
    assert all(g.status == "provisional" for g in GATES.values())


def test_load_gate_phase_must_match_filename(tmp_path):
    p = tmp_path / "wrong_name.yaml"
    p.write_text("phase: other\nsplit: test\nstatus: frozen\ncriteria:\n  - {id: a, metric: m, type: relative_gain}\n")
    with pytest.raises(ValueError, match="file name"):
        load_gate(p)


def test_p1_gate_demonstrates_the_step12_verdict_end_to_end():
    """Uses the real (small-n) Step 12 numbers: the paired stats did not
    support the gain, so the P1 relative-gain criterion must FAIL."""
    step12 = {"comparisons": [{
        "column": "boilerplate", "mean_diff": 0.0567, "ci_lo": 0.0067, "ci_hi": 0.12,
        "p_value": 0.0625, "p_value_holm": 0.1875, "alt_ci": {"mean": 0.6147, "lo": 0.48, "hi": 0.74},
    }]}
    results = results_from_stats_report(step12, "recall@30", "boilerplate")
    report = evaluate_gate(GATES["p1_recall"], results, ALPHA)
    by_id = {r.id: r for r in report.results}
    assert not by_id["recall30_gain"].passed
