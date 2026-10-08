"""
Tests for the CUAD split glue using a tiny SYNTHETIC dataset in CUAD's
SQuAD-style shape (no real CUAD text), plus integrity checks on the frozen
manifest that ships in evaluation/splits/.
"""
import json
from pathlib import Path

import pytest

from lexis.evaluation.dataset.cuad_split import build_cuad_split, contract_group_id, contracts_for_split
from lexis.evaluation.splits import assert_disjoint
from lexis.registry.layered_config import load_yaml

FR = {"dev": 0.7, "test": 0.3}
ROOT = Path(__file__).resolve().parents[2]


def make_raw(n=40):
    def qa(i, impossible):
        return {"id": f"q{i}", "question": "q", "is_impossible": impossible,
                "answers": [] if impossible else [{"text": "a", "answer_start": 0}]}
    data = []
    for i in range(n):
        data.append({"title": f"Contract {i:03d}", "paragraphs": [{
            "context": "text", "qas": [qa(0, False), qa(1, False), qa(2, True)],
        }]})
    return {"data": data}


def test_split_is_contract_disjoint_and_complete():
    m = build_cuad_split(make_raw(), FR, "salt")
    dev = {e["title"] for e in m["splits"]["dev"]}
    test = {e["title"] for e in m["splits"]["test"]}
    assert not dev & test and len(dev | test) == 40


def test_split_is_deterministic():
    assert build_cuad_split(make_raw(), FR, "s") == build_cuad_split(make_raw(), FR, "s")


def test_different_salt_changes_the_split():
    a = build_cuad_split(make_raw(), FR, "s1")["splits"]
    b = build_cuad_split(make_raw(), FR, "s2")["splits"]
    assert a != b


def test_counts_only_answerable_questions():
    m = build_cuad_split(make_raw(), FR, "s")
    total = sum(c["answerable_questions"] for c in m["counts"].values())
    assert total == 40 * 2  # the impossible question is excluded
    for name, entries in m["splits"].items():
        assert m["counts"][name]["contracts"] == len(entries)


def test_pinned_titles_are_forced_into_dev():
    raw = make_raw(60)
    natural = build_cuad_split(raw, FR, "s")
    victims = [e["title"] for e in natural["splits"]["test"][:4]]
    pinned = build_cuad_split(raw, FR, "s", pinned_dev_titles=victims)
    dev = {e["title"] for e in pinned["splits"]["dev"]}
    assert set(victims) <= dev
    assert pinned["pinned"] == {"dev": sorted(victims)}


def test_pinned_title_not_in_dataset_raises():
    with pytest.raises(ValueError, match="not present"):
        build_cuad_split(make_raw(), FR, "s", pinned_dev_titles=["No Such Contract"])


def test_duplicate_titles_rejected():
    raw = make_raw(3)
    raw["data"][1]["title"] = raw["data"][0]["title"]
    with pytest.raises(ValueError, match="unique"):
        build_cuad_split(raw, FR, "s")


def test_contracts_for_split_returns_the_right_contracts_sorted():
    raw = make_raw()
    m = build_cuad_split(raw, FR, "s")
    got = contracts_for_split(raw, m, "test")
    assert [c["title"] for c in got] == sorted(e["title"] for e in m["splits"]["test"])


def test_contracts_for_split_detects_a_changed_dataset():
    raw = make_raw()
    m = build_cuad_split(raw, FR, "s")
    raw["data"] = [c for c in raw["data"] if c["title"] != m["splits"]["test"][0]["title"]]
    with pytest.raises(ValueError, match="missing"):
        contracts_for_split(raw, m, "test")


def test_contracts_for_split_detects_doc_id_drift():
    raw = make_raw()
    m = build_cuad_split(raw, FR, "s")
    m["splits"]["test"][0]["doc_id"] = "cuad-0000000000000000"
    with pytest.raises(ValueError, match="out of sync"):
        contracts_for_split(raw, m, "test")


def test_unknown_split_name():
    raw = make_raw()
    with pytest.raises(KeyError):
        contracts_for_split(raw, build_cuad_split(raw, FR, "s"), "holdout")


# --- the frozen manifest that ships with the repo ---

MANIFEST_PATH = ROOT / "evaluation" / "splits" / "cuad_split_v1.json"


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_frozen_manifest_is_disjoint_and_internally_consistent(manifest):
    assert_disjoint({k: [e["doc_id"] for e in v] for k, v in manifest["splits"].items()})
    for name, entries in manifest["splits"].items():
        assert manifest["counts"][name]["contracts"] == len(entries)
        assert manifest["counts"][name]["answerable_questions"] == sum(e["num_answerable_questions"] for e in entries)
        assert all(e["doc_id"] == contract_group_id(e["title"]) for e in entries)


def test_frozen_manifest_pins_the_baseline_contracts_to_dev(manifest):
    dev_titles = {e["title"] for e in manifest["splits"]["dev"]}
    assert len(manifest["pinned"]["dev"]) >= 10
    assert set(manifest["pinned"]["dev"]) <= dev_titles


def test_frozen_manifest_matches_config_policy(manifest):
    cfg = load_yaml(ROOT / "config" / "defaults.yaml")["eval"]["split"]
    assert manifest["salt"] == cfg["salt"] and manifest["fractions"] == cfg["fractions"]


def test_test_split_is_large_enough_for_tight_intervals(manifest):
    # SE of a proportion ~ 0.5/sqrt(n); with >= 1000 questions the 95% half-width is < 0.032
    assert manifest["counts"]["test"]["answerable_questions"] >= 1000
