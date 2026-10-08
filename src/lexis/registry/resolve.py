"""
Resolve a full, validated experiment configuration.

    python -m lexis.registry.resolve --pack india --set flags.hype=true --set retrieval.rrf_k=30

Steps: load defaults -> nest the chosen jurisdiction pack under "pack" ->
merge experiment files -> apply CLI overrides -> RE-VALIDATE the pack from
the merged result (so an override like pack.models.reranker=... is checked
against the model registry and licence profile) -> attach the resolved model
specs -> hash. The returned mapping is what an evaluation artifact stores.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from lexis.evaluation.provenance import config_hash, redact_secrets
from lexis.packs.loader import load_pack, validate_against_registry
from lexis.packs.schema import JurisdictionPack
from lexis.registry.layered_config import apply_overrides, deep_merge, load_yaml
from lexis.registry.models import ModelRegistry


def resolve_experiment(
    config_dir: Path,
    pack_name: str,
    experiment_files: Iterable[Path] = (),
    overrides: Iterable[str] = (),
    commercial: bool = False,
) -> Dict[str, Any]:
    config_dir = Path(config_dir)
    registry = ModelRegistry.from_file(config_dir / "models.yaml")
    pack = load_pack(config_dir / "packs" / f"{pack_name}.yaml")

    merged: Dict[str, Any] = load_yaml(config_dir / "defaults.yaml")
    merged = deep_merge(merged, {"pack": pack.model_dump()})
    for path in experiment_files:
        merged = deep_merge(merged, load_yaml(Path(path)))
    merged = apply_overrides(merged, overrides)

    final_pack = JurisdictionPack(**merged["pack"])
    validate_against_registry(final_pack, registry, commercial=commercial)

    merged["pack"] = final_pack.model_dump()
    merged["profile"] = {"commercial": commercial}
    merged["resolved_models"] = {
        kind: registry.get(key).model_dump()
        for kind in ("embedder", "reranker", "nli", "llm")
        if (key := getattr(final_pack.models, kind)) is not None
    }
    return merged


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Resolve and validate a layered experiment config.")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--experiment", action="append", default=[], help="experiment YAML (repeatable)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="dotted.path=json_value (repeatable)")
    ap.add_argument("--commercial", action="store_true", help="refuse non-commercial / unverified-licence models")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    resolved = resolve_experiment(Path(args.config_dir), args.pack, [Path(p) for p in args.experiment],
                                  args.overrides, args.commercial)
    text = json.dumps(redact_secrets(resolved), indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    print(f"config_hash: {config_hash(resolved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
