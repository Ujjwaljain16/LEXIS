"""
Contract tests over the REAL shipped config (config/models.yaml, config/packs/*.yaml,
config/defaults.yaml) plus the resolver. These are the tests that make the
"add a jurisdiction by adding data" claim checkable, and that keep licence
obligations from being forgotten.
"""
import json
from pathlib import Path

import pytest

from lexis.evaluation.provenance import config_hash
from lexis.packs.citations import extract_citations
from lexis.packs.loader import PackError, discover_packs, validate_against_registry
from lexis.registry.layered_config import load_yaml
from lexis.registry.models import LicenseError, ModelRegistry
from lexis.registry.resolve import resolve_experiment

CONFIG = Path(__file__).resolve().parents[2] / "config"
REGISTRY = ModelRegistry.from_file(CONFIG / "models.yaml")
PACKS = discover_packs(CONFIG / "packs")


def test_shipped_packs_cover_the_three_v1_jurisdictions():
    assert {"us_contracts", "india", "uk_eu_statutes"} <= set(PACKS)


@pytest.mark.parametrize("name", sorted(PACKS))
def test_every_shipped_pack_validates_against_the_registry(name):
    validate_against_registry(PACKS[name], REGISTRY, commercial=False)


@pytest.mark.parametrize("name", sorted(PACKS))
def test_every_shipped_pack_has_citation_grammars_and_compliance_notes(name):
    assert PACKS[name].citation_grammars, f"{name} defines no citation grammars"
    assert PACKS[name].compliance, f"{name} records no licence/compliance obligations"


def test_registry_models_all_have_a_recorded_license():
    assert all(REGISTRY.get(k).license for k in REGISTRY.keys())


def test_noncommercial_models_are_never_marked_commercial_ok():
    for key in REGISTRY.keys():
        spec = REGISTRY.get(key)
        if "NC" in spec.license.upper().split("-"):
            assert not spec.commercial_ok, f"{key} has a non-commercial licence but commercial_ok=true"


def test_datasets_with_nc_licences_are_flagged_in_the_pack_that_uses_them():
    india = PACKS["india"]
    flagged = {c.source for c in india.compliance if c.blocks_commercial_use}
    assert {"IL-TUR", "AILQA"} <= flagged


def test_india_pack_extracts_all_three_citation_styles():
    text = "See AIR 1973 SC 1461, (1973) 4 SCC 225 and 2023 INSC 1."
    assert [m.grammar for m in extract_citations(text, PACKS["india"])] == ["air", "scc", "neutral_insc"]


def test_uk_eu_pack_extracts_neutral_ecli_and_celex():
    text = "[2020] UKSC 1 ; ECLI:EU:C:2014:317 ; Regulation 32016R0679."
    assert [m.grammar for m in extract_citations(text, PACKS["uk_eu_statutes"])] == ["uk_neutral", "ecli", "celex"]


def test_us_pack_extracts_reports_and_code_citations():
    text = "Brown, 347 U.S. 483; see 15 U.S.C. § 78j."
    assert [m.grammar for m in extract_citations(text, PACKS["us_contracts"])] == ["us_reports", "us_code"]


def test_commercial_profile_currently_blocks_unverified_models():
    """Documents a real to-do: bge-m3 and the NLI model licences are unverified."""
    with pytest.raises(PackError):
        validate_against_registry(PACKS["us_contracts"], REGISTRY, commercial=True)


# --- resolver ---

def test_defaults_mirror_the_frozen_baseline_retrieval_settings():
    d = load_yaml(CONFIG / "defaults.yaml")
    assert d["retrieval"]["rrf_k"] == 61 and d["retrieval"]["top_k_per_path"] == 15
    assert all(v is False for v in d["flags"].values()), "every ladder rung must default OFF (baseline behaviour)"


def test_resolve_is_deterministic_and_hash_changes_with_overrides():
    a = resolve_experiment(CONFIG, "india")
    b = resolve_experiment(CONFIG, "india")
    c = resolve_experiment(CONFIG, "india", overrides=["flags.hype=true"])
    assert config_hash(a) == config_hash(b) != config_hash(c)


def test_resolve_attaches_resolved_model_specs():
    r = resolve_experiment(CONFIG, "us_contracts")
    assert r["resolved_models"]["embedder"]["dim"] == 1024
    assert r["resolved_models"]["reranker"]["kind"] == "reranker"


def test_resolve_revalidates_pack_after_overrides():
    with pytest.raises(PackError):
        resolve_experiment(CONFIG, "india", overrides=['pack.models.reranker="not-a-model"'])


def test_resolve_commercial_blocks_noncommercial_override():
    with pytest.raises(PackError, match="non-commercial"):
        resolve_experiment(CONFIG, "india", overrides=['pack.models.reranker="jina-reranker-v3"'], commercial=True)


def test_resolve_layers_experiment_file_between_pack_and_cli(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text("retrieval:\n  rrf_k: 45\nflags:\n  rerank: true\n")
    r = resolve_experiment(CONFIG, "us_contracts", experiment_files=[exp], overrides=["retrieval.rrf_k=99"])
    assert r["retrieval"]["rrf_k"] == 99 and r["flags"]["rerank"] is True


def test_resolved_config_is_json_serializable():
    json.dumps(resolve_experiment(CONFIG, "uk_eu_statutes"))
