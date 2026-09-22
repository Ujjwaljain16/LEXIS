import copy

import pytest

from lexis.registry.layered_config import (
    apply_overrides,
    deep_merge,
    diff_configs,
    load_yaml,
    parse_override,
    resolve_config,
)


def test_deep_merge_merges_mappings_and_replaces_lists():
    base = {"a": {"x": 1, "y": 2}, "l": [1, 2, 3]}
    out = deep_merge(base, {"a": {"y": 9, "z": 3}, "l": [7]})
    assert out == {"a": {"x": 1, "y": 9, "z": 3}, "l": [7]}


def test_deep_merge_does_not_mutate_inputs():
    base, override = {"a": {"x": [1]}}, {"a": {"x": [2]}}
    b0, o0 = copy.deepcopy(base), copy.deepcopy(override)
    out = deep_merge(base, override)
    out["a"]["x"].append(99)
    assert base == b0 and override == o0


def test_deep_merge_scalar_replaces_mapping_and_vice_versa():
    assert deep_merge({"a": {"x": 1}}, {"a": 5}) == {"a": 5}
    assert deep_merge({"a": 5}, {"a": {"x": 1}}) == {"a": {"x": 1}}


@pytest.mark.parametrize("text,expected", [
    ("a=1", {"a": 1}),
    ("a=1.5", {"a": 1.5}),
    ("a=true", {"a": True}),
    ("a=null", {"a": None}),
    ("a=[1,2]", {"a": [1, 2]}),
    ("a=hello", {"a": "hello"}),
    ('a="quoted"', {"a": "quoted"}),
    ("a.b.c=3", {"a": {"b": {"c": 3}}}),
    ("a=x=y", {"a": "x=y"}),
])
def test_parse_override_types_and_nesting(text, expected):
    assert parse_override(text) == expected


@pytest.mark.parametrize("bad", ["novalue", "=1", "a..b=1", ".a=1"])
def test_parse_override_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_override(bad)


def test_overrides_apply_in_order_last_wins():
    assert apply_overrides({"a": 1}, ["a=2", "a=3"]) == {"a": 3}


def test_precedence_defaults_then_pack_then_experiment_then_cli():
    out = resolve_config(
        [{"k": "default", "only_default": 1}, {"k": "pack"}, {"k": "experiment", "e": 1}],
        overrides=["k=cli"],
    )
    assert out == {"k": "cli", "only_default": 1, "e": 1}


def test_resolve_without_overrides_uses_last_layer():
    assert resolve_config([{"k": 1}, {"k": 2}]) == {"k": 2}


def test_diff_configs_reports_changed_added_removed_paths():
    a = {"x": 1, "n": {"p": 1, "q": 2}, "gone": 1}
    b = {"x": 2, "n": {"p": 1, "q": 3}, "new": 1}
    assert diff_configs(a, b) == ["gone", "n.q", "new", "x"]


def test_diff_configs_identical_is_empty():
    assert diff_configs({"a": {"b": 1}}, {"a": {"b": 1}}) == []


def test_load_yaml_rejects_non_mapping_and_handles_empty(tmp_path):
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n")
    (tmp_path / "empty.yaml").write_text("")
    with pytest.raises(ValueError):
        load_yaml(tmp_path / "list.yaml")
    assert load_yaml(tmp_path / "empty.yaml") == {}
