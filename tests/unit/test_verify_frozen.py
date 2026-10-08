"""The committed frozen result must keep verifying, tampering must be detected, and the generated
compliance register must match its sources."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from lexis.evaluation.verify_frozen import verify

ROOT = Path(__file__).resolve().parents[2]


def failed(checks):
    return [c.name for c in checks if not c.ok]


def test_committed_frozen_result_verifies():
    assert failed(verify(ROOT)) == []


@pytest.fixture
def sandbox(tmp_path):
    for rel in ("evaluation/reports/cuad_doc_scoped_test.json", "evaluation/splits/cuad_split_v1.json", "README.md"):
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dest)
    return tmp_path


def edit_report(root, fn):
    path = root / "evaluation/reports/cuad_doc_scoped_test.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    fn(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_tampered_case_scores_are_detected(sandbox):
    def tamper(d):
        d["per_case"][0]["recall_at_k"] = 1.0 - d["per_case"][0]["recall_at_k"]
    edit_report(sandbox, tamper)
    assert "Recall@30 mean recomputes" in failed(verify(sandbox))


def test_headline_edit_without_rerun_is_detected(sandbox):
    edit_report(sandbox, lambda d: d.__setitem__("mean_reciprocal_rank", 0.9))
    assert "MRR mean recomputes" in failed(verify(sandbox))


def test_readme_drift_is_detected(sandbox):
    readme = sandbox / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8").replace("0.853", "0.953"), encoding="utf-8")
    assert "README quotes Recall@30" in failed(verify(sandbox))


def test_manifest_swap_is_detected(sandbox):
    manifest = sandbox / "evaluation/splits/cuad_split_v1.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["splits"]["test"] = data["splits"]["test"][:-1]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    names = failed(verify(sandbox))
    assert "manifest on disk is the one the run recorded" in names
    assert "run used exactly the manifest's test contracts" in names


def test_compliance_register_matches_its_sources():
    spec = importlib.util.spec_from_file_location("render_compliance", ROOT / "scripts" / "render_compliance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    committed = (ROOT / "docs" / "COMPLIANCE.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed == module.render(), "docs/COMPLIANCE.md is stale; run: python scripts/render_compliance.py"
