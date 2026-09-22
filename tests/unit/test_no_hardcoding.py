"""
Enforces the zero-hardcoding ratchet (quality/hardcode_baseline.json) and
unit-tests the lint that produces it. Uses generic synthetic snippets, never
project code, for the detection tests.
"""
import json
from pathlib import Path

import pytest

from lexis.quality.hardcode_lint import (
    compare_to_baseline,
    load_rules,
    scan_source,
    scan_tree,
    summarize,
    total,
    write_baseline,
)

REPO = Path(__file__).resolve().parents[2]
RULES = load_rules(REPO / "quality" / "hardcode_rules.json")


def rules_of(v):
    return sorted(x.rule for x in v)


# --- enforcement ---

def test_no_new_hardcoding_beyond_the_ratchet_baseline():
    current = summarize(scan_tree(REPO, RULES))
    baseline = json.loads((REPO / "quality" / "hardcode_baseline.json").read_text())
    problems = compare_to_baseline(current, baseline)
    assert not problems, "New hardcoded values (move them to config/packs/registry):\n" + "\n".join(problems)


def test_baseline_is_not_stale_high():
    """When violations are removed the baseline should be lowered so the
    ratchet keeps biting: current total may not be far below the baseline."""
    current = total(summarize(scan_tree(REPO, RULES)))
    baseline = total(json.loads((REPO / "quality" / "hardcode_baseline.json").read_text()))
    assert current >= baseline - 5, "Lower quality/hardcode_baseline.json (run with --update-baseline)"


# --- detection ---

def test_flags_numeric_default_and_call_keyword():
    src = "def f(x, top_k=15):\n    return g(x, timeout=2.5)\n"
    assert rules_of(scan_source(src, "m.py", RULES)) == ["numeric:call_keyword", "numeric:function_default"]


def test_flags_comparison_and_module_constant():
    src = "LIMIT = 250\ndef f(v):\n    return v > 0.35\n"
    assert rules_of(scan_source(src, "m.py", RULES)) == ["numeric:comparison", "numeric:module_constant"]


def test_allowed_numbers_are_not_flagged():
    src = "def f(x=0, y=1):\n    return g(a=-1, b=2.0) if x > 1 else 0\n"
    assert scan_source(src, "m.py", RULES) == []


def test_negative_numbers_are_evaluated_with_sign():
    src = "def f(x=-1, y=-7):\n    pass\n"
    vals = [v.value for v in scan_source(src, "m.py", RULES)]
    assert vals == ["-7"]


def test_flags_url_model_and_repo_id_strings():
    src = 'A = "https://example.org/x"\nB = "some-org/some-model"\nC = "text-embedding-bge-large"\n'
    assert rules_of(scan_source(src, "m.py", RULES)) == ["string:hf_repo_id", "string:model_identifier", "string:url_literal"]


def test_docstrings_are_not_flagged():
    src = '"""Uses gemini and https://x.org/y"""\ndef f():\n    """Calls gpt-4 via http://a.b/c"""\n'
    assert scan_source(src, "m.py", RULES) == []


def test_bools_and_plain_strings_are_not_flagged():
    src = 'def f(flag=True, name="hello world"):\n    return g(verbose=False)\n'
    assert scan_source(src, "m.py", RULES) == []


def test_syntax_error_source_is_skipped_not_fatal():
    assert scan_source("def broken(:\n", "m.py", RULES) == []


def test_new_pattern_in_rules_requires_no_code_change():
    rules = dict(RULES, string_patterns={"custom": "acme-corp"})
    assert rules_of(scan_source('X = "acme-corp-thing"\n', "m.py", rules)) == ["string:custom"]


# --- ratchet mechanics ---

def test_compare_flags_increase_and_new_file_only():
    base = {"a.py": {"numeric:call_keyword": 2}}
    assert compare_to_baseline({"a.py": {"numeric:call_keyword": 2}}, base) == []
    assert compare_to_baseline({"a.py": {"numeric:call_keyword": 1}}, base) == []
    assert len(compare_to_baseline({"a.py": {"numeric:call_keyword": 3}}, base)) == 1
    assert len(compare_to_baseline({"b.py": {"string:url_literal": 1}}, base)) == 1


def test_write_baseline_refuses_to_raise_counts(tmp_path):
    path = tmp_path / "b.json"
    existing = {"a.py": {"numeric:call_keyword": 1}}
    with pytest.raises(ValueError):
        write_baseline({"a.py": {"numeric:call_keyword": 2}}, path, existing)
    write_baseline({"a.py": {"numeric:call_keyword": 2}}, path, existing, allow_increase=True)
    assert json.loads(path.read_text()) == {"a.py": {"numeric:call_keyword": 2}}


def test_write_baseline_allows_lowering(tmp_path):
    path = tmp_path / "b.json"
    write_baseline({"a.py": {"numeric:call_keyword": 0}}, path, {"a.py": {"numeric:call_keyword": 3}})
    assert path.exists()
