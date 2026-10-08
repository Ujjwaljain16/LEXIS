"""
Every module in the package must import. Two deep-mode modules once
referenced types that were never defined, so they crashed on import and
nothing noticed because nothing imported them. This test makes an
un-importable module a build failure.

Modules are discovered by walking the source tree, not pkgutil: the
`lexis` package is a namespace package (no __init__.py), which pkgutil's
walk_packages silently skips below the top level.
"""
import importlib
from pathlib import Path

import pytest

import lexis

SRC_ROOT = Path(next(iter(lexis.__path__))).parent


def _discover():
    names = []
    for path in sorted((SRC_ROOT / "lexis").rglob("*.py")):
        if path.name == "__init__.py" or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(SRC_ROOT).with_suffix("")
        names.append(".".join(rel.parts))
    return names


MODULES = _discover()


def test_discovery_finds_the_whole_tree():
    assert len(MODULES) > 50
    assert "lexis.retrieval.hybrid_retriever" in MODULES


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
