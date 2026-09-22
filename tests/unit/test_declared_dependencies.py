"""
Every third-party import used anywhere in src/lexis must be declared in
pyproject.toml's [project] dependencies -- otherwise the package works in
whatever environment happens to already have the stray package installed
(as this repo's own local venv did, twice: python-docx and asyncpg were
both used unconditionally but undeclared, so a fresh install -- e.g. Colab
-- broke with ModuleNotFoundError while the local venv looked fine).

This maps import name -> distribution (pip package) name using
importlib.metadata.packages_distributions() against what is ALREADY
installed locally, then checks that at least one providing distribution
for each import is declared. This cannot prove a distribution's declared
version range actually ships that import name (only a real fresh install
can prove that end-to-end) -- but it does catch the exact class of bug
that bit this project twice, cheaply, on every test run.

OPTIONAL_IMPORTS lists imports that are deliberately soft (wrapped in
try/except and never required for the module to load), so they are not
held to the same rule.
"""
import ast
import sys
import tomllib
from pathlib import Path
from importlib.metadata import packages_distributions

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "lexis"

# name -> why it's exempt from the "must be a hard dependency" rule
OPTIONAL_IMPORTS = {
    "torch": "evaluation/run_eval.py::device_info() imports it inside a try/except as a "
             "best-effort provenance fact; also guaranteed transitively by sentence-transformers.",
    "kagglehub": "evaluation/generate_dataset.py imports it lazily inside the one function that "
                 "needs it, with a clear RuntimeError if missing -- not needed by the live pipeline.",
}


def _declared_package_names():
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    names = set()
    for dep in data["project"]["dependencies"]:
        # strip extras ("pkg[extra]"), version specifiers, and markers -- keep just the name
        name = dep
        for sep in ("[", "<", ">", "=", "!", ";", " "):
            name = name.split(sep, 1)[0]
        names.add(name.strip().lower())
    return names


def _used_top_level_imports():
    used = set()
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    used.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                used.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names)
    return {u for u in used if u not in stdlib and u != "lexis"}


def test_optional_imports_list_has_no_stale_entries():
    """If an OPTIONAL_IMPORTS entry is no longer actually imported anywhere,
    remove it -- a stale exemption hides nothing but is worth cleaning up."""
    used = _used_top_level_imports()
    stale = set(OPTIONAL_IMPORTS) - used
    assert not stale, f"OPTIONAL_IMPORTS lists imports no longer used anywhere: {stale}"


def test_every_hard_import_has_a_declared_dependency():
    used = _used_top_level_imports() - set(OPTIONAL_IMPORTS)
    declared = _declared_package_names()
    dist_map = packages_distributions()

    unmapped = sorted(u for u in used if u not in dist_map)
    gaps = []
    for name in sorted(used):
        if name not in dist_map:
            continue
        providers = [d.lower() for d in dist_map[name]]
        if not any(p in declared for p in providers):
            gaps.append(f"{name!r} (installed via {dist_map[name]}, not declared in pyproject.toml)")

    assert not gaps, (
        "Undeclared hard dependencies -- these import fine here only because the package "
        "happens to already be installed locally; a fresh install (Colab, CI, a new machine) "
        "will hit ModuleNotFoundError:\n  " + "\n  ".join(gaps)
    )
    assert not unmapped, (
        "Imports with no known local distribution mapping -- verify by hand whether they need "
        "a pyproject.toml entry, then add to OPTIONAL_IMPORTS with a reason if genuinely optional:\n  "
        + "\n  ".join(unmapped)
    )


@pytest.mark.parametrize("name,reason", sorted(OPTIONAL_IMPORTS.items()))
def test_optional_import_is_actually_guarded(name, reason):
    """Every OPTIONAL_IMPORTS entry must cite a real file where the import
    appears inside a try/except (or is otherwise not import-time-required),
    not just be an unverified claim."""
    assert reason, f"{name} has no documented reason for its exemption"
    hits = [p for p in SRC_ROOT.rglob("*.py") if f"import {name}" in p.read_text(encoding="utf-8")]
    assert hits, f"{name} is in OPTIONAL_IMPORTS but is no longer imported anywhere: {reason}"
