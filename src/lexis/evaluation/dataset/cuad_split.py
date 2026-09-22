"""
CUAD dev/test split (benchmark-adapter glue over the generic splits module).

The split is contract-disjoint (a contract never contributes questions to
both dev and test) and is frozen as a MANIFEST file so it can be reviewed,
hashed, and never silently changes. Contracts already analysed while the
system was being built (the frozen baseline's contracts) are PINNED to dev:
data that has informed design decisions cannot double as held-out test data.

Nothing here decides the salt, fractions, or which contracts to pin -- those
are inputs (config / CLI) so the split policy stays data, not code.
"""
from typing import Any, Dict, Iterable, List, Mapping

from lexis.evaluation.dataset.cuad_loader import _answerable_questions
from lexis.evaluation.dataset.mapping import deterministic_document_id
from lexis.evaluation.splits import assert_disjoint, split_groups

BENCHMARK_PREFIX = "cuad"
MANIFEST_VERSION = 1


def contract_group_id(title: str) -> str:
    return deterministic_document_id(title, prefix=BENCHMARK_PREFIX)


def build_cuad_split(
    raw: Mapping[str, Any],
    fractions: Mapping[str, float],
    salt: str,
    pinned_dev_titles: Iterable[str] = (),
    pinned_split: str = "dev",
) -> Dict[str, Any]:
    contracts = list(raw["data"])
    titles = [c["title"] for c in contracts]
    if len(set(titles)) != len(titles):
        raise ValueError("contract titles must be unique to be used as split group keys")

    by_gid = {contract_group_id(c["title"]): c for c in contracts}
    title_to_gid = {c["title"]: contract_group_id(c["title"]) for c in contracts}
    pinned_titles = sorted(set(pinned_dev_titles))
    unknown = [t for t in pinned_titles if t not in title_to_gid]
    if unknown:
        raise ValueError(f"pinned titles not present in the dataset: {unknown}")

    splits = split_groups(
        by_gid, fractions, salt,
        pinned={pinned_split: [title_to_gid[t] for t in pinned_titles]} if pinned_titles else None,
    )
    assert_disjoint(splits)

    def entry(gid: str) -> Dict[str, Any]:
        contract = by_gid[gid]
        return {"title": contract["title"], "doc_id": gid,
                "num_answerable_questions": len(_answerable_questions(contract))}

    manifest_splits = {
        name: sorted((entry(g) for g in gids), key=lambda e: e["title"]) for name, gids in splits.items()
    }
    return {
        "version": MANIFEST_VERSION,
        "benchmark": BENCHMARK_PREFIX,
        "salt": salt,
        "fractions": dict(fractions),
        "pinned": {pinned_split: pinned_titles},
        "counts": {
            name: {"contracts": len(entries), "answerable_questions": sum(e["num_answerable_questions"] for e in entries)}
            for name, entries in manifest_splits.items()
        },
        "splits": manifest_splits,
    }


def contracts_for_split(raw: Mapping[str, Any], manifest: Mapping[str, Any], split: str) -> List[Dict[str, Any]]:
    """The raw contract dicts belonging to `split`, sorted by title (the same
    ordering select_contracts uses). Verifies every manifest entry still
    resolves to the same doc_id, so a changed/renamed dataset fails loudly
    instead of silently evaluating on different data."""
    if split not in manifest["splits"]:
        raise KeyError(f"unknown split {split!r}; manifest has {sorted(manifest['splits'])}")
    by_title = {c["title"]: c for c in raw["data"]}
    out = []
    for entry in manifest["splits"][split]:
        title = entry["title"]
        if title not in by_title:
            raise ValueError(f"manifest contract {title!r} is missing from the dataset")
        if contract_group_id(title) != entry["doc_id"]:
            raise ValueError(f"doc_id mismatch for {title!r}: dataset/manifest are out of sync")
        out.append(by_title[title])
    return sorted(out, key=lambda c: c["title"])
