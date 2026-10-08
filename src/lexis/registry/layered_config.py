"""
Layered experiment configuration (plan section 3.1, extension point 5).

Precedence, lowest to highest:
    defaults  ->  jurisdiction pack  ->  experiment file  ->  CLI overrides

Everything is a plain mapping; layers are deep-merged (mappings merge key by
key, every other value -- including lists -- is replaced wholesale, so an
experiment can fully redefine a list without surprising append semantics).
The resolved result is what gets hashed and stored by
evaluation/provenance.py, so an artifact always records the exact
configuration that produced it.

CLI overrides use dotted paths with JSON-typed values:
    retrieval.rrf_k=61   models.reranker="bge-reranker-v2-m3"   flags.hype=true
A bare (non-JSON) value is kept as a string.
"""
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def parse_override(text: str) -> Dict[str, Any]:
    """'a.b.c=VALUE' -> {'a': {'b': {'c': VALUE}}} with VALUE JSON-decoded when possible."""
    if "=" not in text:
        raise ValueError(f"override {text!r} must look like dotted.path=value")
    path, raw = text.split("=", 1)
    keys = [k for k in path.strip().split(".")]
    if not path.strip() or any(not k for k in keys):
        raise ValueError(f"override {text!r} has an empty path segment")
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    nested: Any = value
    for key in reversed(keys):
        nested = {key: nested}
    return nested


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    out = dict(config)
    for text in overrides:
        out = deep_merge(out, parse_override(text))
    return out


def resolve_config(layers: Iterable[Mapping[str, Any]], overrides: Iterable[str] = ()) -> Dict[str, Any]:
    """Merge layers in order (later wins), then apply CLI overrides last."""
    merged: Dict[str, Any] = {}
    for layer in layers:
        merged = deep_merge(merged, layer)
    return apply_overrides(merged, overrides)


def diff_configs(a: Mapping[str, Any], b: Mapping[str, Any], prefix: str = "") -> List[str]:
    """Dotted paths whose values differ -- handy for logging exactly what an
    ablation changed relative to its baseline."""
    paths: List[str] = []
    for key in sorted(set(a) | set(b)):
        here = f"{prefix}{key}"
        if key not in a or key not in b:
            paths.append(here)
        elif isinstance(a[key], Mapping) and isinstance(b[key], Mapping):
            paths.extend(diff_configs(a[key], b[key], prefix=f"{here}."))
        elif a[key] != b[key]:
            paths.append(here)
    return paths
