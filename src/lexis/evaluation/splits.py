"""
Deterministic, group-disjoint dev/test splitting (plan P0).

The unit of splitting is a GROUP (a contract, a judgment, a statute -- any
document-level id), never an individual query or chunk, so no document
contributes to both dev and test. Two strategies, chosen by need:

- assign_split(): pure function of (group_id, salt). Stable under additions
  -- once a group lands in "test" it stays there forever, no matter what
  else is added later. Proportions are only approximate for small sets.
  Use this to protect a frozen test set.
- split_groups_exact(): rank groups by hash and cut at exact proportions.
  Proportions are exact for the given set, but adding groups can shift
  boundary members. Use this for one-off experiments on small corpora.

Nothing here knows about any benchmark; the salt and fractions come from
config so the split policy is data, not code.
"""
import hashlib
from typing import Dict, Iterable, List, Mapping, Optional


def _validate(fractions: Mapping[str, float]) -> None:
    if not fractions:
        raise ValueError("fractions must not be empty")
    if any(f <= 0 for f in fractions.values()):
        raise ValueError("every split fraction must be > 0")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")


def _unit_interval(group_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}|{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(group_id: str, fractions: Mapping[str, float], salt: str = "") -> str:
    _validate(fractions)
    u = _unit_interval(group_id, salt)
    cumulative = 0.0
    last = None
    for name, frac in fractions.items():
        cumulative += frac
        last = name
        if u < cumulative:
            return name
    return last


def split_groups(group_ids: Iterable[str], fractions: Mapping[str, float], salt: str = "",
                 pinned: Optional[Mapping[str, Iterable[str]]] = None) -> Dict[str, List[str]]:
    """Threshold-hash split (stable under additions).

    `pinned` forces specific groups into a named split regardless of their
    hash -- used for groups that were already analysed while building the
    system (they can never be held-out test data). Pinned ids must exist in
    group_ids, name a real split, and appear in only one split; anything else
    raises so a typo cannot silently unpin a contaminated group."""
    _validate(fractions)
    ids = sorted(set(group_ids))
    forced: Dict[str, str] = {}
    for split_name, members in (pinned or {}).items():
        if split_name not in fractions:
            raise ValueError(f"pinned split {split_name!r} is not one of {list(fractions)}")
        for gid in members:
            if gid not in ids:
                raise ValueError(f"pinned group {gid!r} is not among the provided groups")
            if gid in forced and forced[gid] != split_name:
                raise ValueError(f"group {gid!r} is pinned to both {forced[gid]!r} and {split_name!r}")
            forced[gid] = split_name
    out: Dict[str, List[str]] = {name: [] for name in fractions}
    for gid in ids:
        out[forced.get(gid) or assign_split(gid, fractions, salt)].append(gid)
    return out


def split_groups_exact(group_ids: Iterable[str], fractions: Mapping[str, float], salt: str = "") -> Dict[str, List[str]]:
    """Rank-by-hash split with exact proportions (largest-remainder rounding)."""
    _validate(fractions)
    ids = sorted(set(group_ids), key=lambda g: (_unit_interval(g, salt), g))
    n = len(ids)
    raw = {name: frac * n for name, frac in fractions.items()}
    counts = {name: int(v) for name, v in raw.items()}
    remainder = n - sum(counts.values())
    for name in sorted(raw, key=lambda k: (raw[k] - counts[k]), reverse=True)[:remainder]:
        counts[name] += 1
    out: Dict[str, List[str]] = {}
    start = 0
    for name in fractions:
        out[name] = sorted(ids[start:start + counts[name]])
        start += counts[name]
    return out


def assert_disjoint(splits: Mapping[str, Iterable[str]]) -> None:
    seen: Dict[str, str] = {}
    for name, members in splits.items():
        for m in members:
            if m in seen:
                raise AssertionError(f"group {m!r} appears in both {seen[m]!r} and {name!r}")
            seen[m] = name
