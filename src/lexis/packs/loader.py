"""
Pack loading, discovery, and cross-validation against the model registry.
"""
from pathlib import Path
from typing import Dict, List

import yaml

from lexis.packs.schema import JurisdictionPack
from lexis.registry.models import LicenseError, ModelRegistry


class PackError(Exception):
    """One or more problems with a pack; message lists all of them."""


def load_pack(path: Path) -> JurisdictionPack:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PackError(f"{path}: top level must be a mapping")
    pack = JurisdictionPack(**data)
    if pack.name != path.stem:
        raise PackError(f"{path}: pack name {pack.name!r} must equal the file name {path.stem!r}")
    return pack


def discover_packs(directory: Path) -> Dict[str, JurisdictionPack]:
    packs: Dict[str, JurisdictionPack] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        pack = load_pack(path)
        packs[pack.name] = pack
    return packs


_KINDS = ("embedder", "reranker", "nli", "llm")


def validate_against_registry(pack: JurisdictionPack, registry: ModelRegistry, commercial: bool = False) -> None:
    """Every model key a pack references must exist, be of the right kind,
    and (in a commercial profile) be licence-permitted. Reports ALL problems."""
    problems: List[str] = []
    for kind in _KINDS:
        key = getattr(pack.models, kind)
        if key is None:
            continue
        try:
            registry.resolve(key, kind, commercial=commercial)
        except (KeyError, TypeError, LicenseError) as e:
            problems.append(f"{pack.name}.models.{kind}: {e}")
    if problems:
        raise PackError("\n".join(problems))
