"""
CLI: build (or verify) the frozen CUAD dev/test split manifest.

    python -m lexis.evaluation.run_make_cuad_split \\
        --cuad-path data/cuad_raw/CUAD_v1/CUAD_v1.json \\
        --pin-dev-first-n 10 --output evaluation/splits/cuad_split_v1.json

Salt and fractions come from config/defaults.yaml (eval.split) unless
overridden. --pin-dev-first-n pins the first N contracts by title -- exactly
the contracts the frozen baseline (select_contracts) used -- to dev.
--check re-derives the split and fails if it differs from the file on disk,
which is how CI proves the frozen test set has not moved.
"""
import argparse
import json
from pathlib import Path

from lexis.evaluation.dataset.cuad_loader import CUADLoader, select_contracts
from lexis.evaluation.dataset.cuad_split import build_cuad_split
from lexis.evaluation.provenance import build_provenance
from lexis.registry.layered_config import load_yaml


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build/verify the frozen CUAD dev/test split.")
    ap.add_argument("--cuad-path", required=True)
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--pin-dev-first-n", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--check", action="store_true", help="fail if the split on disk differs from a fresh derivation")
    args = ap.parse_args(argv)

    split_cfg = load_yaml(Path(args.config_dir) / "defaults.yaml")["eval"]["split"]
    raw = CUADLoader().load(args.cuad_path)
    pinned = [c["title"] for c in select_contracts(raw, args.pin_dev_first_n)] if args.pin_dev_first_n > 0 else []
    manifest = build_cuad_split(raw, split_cfg["fractions"], split_cfg["salt"], pinned_dev_titles=pinned)

    out = Path(args.output)
    if args.check:
        on_disk = json.loads(out.read_text(encoding="utf-8"))
        on_disk.pop("provenance", None)
        if on_disk != manifest:
            print("SPLIT MISMATCH: derived split differs from the frozen manifest")
            return 1
        print("split manifest verified: identical to a fresh derivation")
        return 0

    manifest["provenance"] = build_provenance(
        config={"split": split_cfg, "pin_dev_first_n": args.pin_dev_first_n},
        data_paths=[args.cuad_path],
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, c in manifest["counts"].items():
        print(f"{name}: {c['contracts']} contracts, {c['answerable_questions']} answerable questions")
    print(f"pinned to dev (already analysed): {len(pinned)} contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
