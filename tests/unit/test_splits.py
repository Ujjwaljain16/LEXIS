import pytest

from lexis.evaluation.splits import assert_disjoint, assign_split, split_groups, split_groups_exact

FR = {"dev": 0.6, "test": 0.4}


def test_assign_split_is_deterministic():
    assert assign_split("doc-1", FR, salt="s") == assign_split("doc-1", FR, salt="s")


def test_salt_changes_assignment_for_some_groups():
    ids = [f"doc-{i}" for i in range(200)]
    a = [assign_split(i, FR, "a") for i in ids]
    b = [assign_split(i, FR, "b") for i in ids]
    assert a != b


def test_split_is_group_disjoint_and_complete():
    ids = [f"g{i}" for i in range(100)]
    s = split_groups(ids, FR, "x")
    assert_disjoint(s)
    assert sorted(sum(s.values(), [])) == sorted(ids)


def test_threshold_split_proportions_approximately_hold_on_large_sets():
    ids = [f"g{i}" for i in range(2000)]
    s = split_groups(ids, FR, "x")
    assert abs(len(s["test"]) / 2000 - 0.4) < 0.05


def test_threshold_split_is_stable_under_additions():
    base = [f"g{i}" for i in range(50)]
    before = split_groups(base, FR, "x")
    after = split_groups(base + [f"new{i}" for i in range(50)], FR, "x")
    for name in FR:
        assert set(before[name]) <= set(after[name])  # nobody moved between splits


def test_split_ignores_input_order_and_duplicates():
    ids = [f"g{i}" for i in range(30)]
    assert split_groups(ids, FR, "x") == split_groups(list(reversed(ids)) + ids[:5], FR, "x")


def test_exact_split_has_exact_counts():
    ids = [f"g{i}" for i in range(10)]
    s = split_groups_exact(ids, {"dev": 0.7, "test": 0.3}, "x")
    assert len(s["dev"]) == 7 and len(s["test"]) == 3
    assert_disjoint(s)


def test_exact_split_three_way_counts_sum_to_n():
    ids = [f"g{i}" for i in range(11)]
    s = split_groups_exact(ids, {"train": 0.5, "dev": 0.25, "test": 0.25}, "x")
    assert sum(len(v) for v in s.values()) == 11


@pytest.mark.parametrize("bad", [{}, {"a": 0.5}, {"a": 1.2, "b": -0.2}, {"a": 0.0, "b": 1.0}])
def test_invalid_fractions_rejected(bad):
    with pytest.raises(ValueError):
        assign_split("x", bad)


def test_pinned_groups_are_forced_into_their_split_regardless_of_hash():
    ids = [f"g{i}" for i in range(200)]
    natural = split_groups(ids, FR, "x")
    victims = natural["test"][:5]                       # would naturally land in test
    pinned = split_groups(ids, FR, "x", pinned={"dev": victims})
    assert all(v in pinned["dev"] and v not in pinned["test"] for v in victims)
    assert_disjoint(pinned)
    assert sorted(sum(pinned.values(), [])) == sorted(ids)


def test_pinning_leaves_unpinned_assignments_unchanged():
    ids = [f"g{i}" for i in range(100)]
    natural = split_groups(ids, FR, "x")
    pinned = split_groups(ids, FR, "x", pinned={"dev": natural["test"][:3]})
    for gid in ids:
        if gid not in natural["test"][:3]:
            assert (gid in pinned["dev"]) == (gid in natural["dev"])


def test_pinned_unknown_group_raises():
    with pytest.raises(ValueError, match="not among"):
        split_groups(["a", "b"], FR, "x", pinned={"dev": ["zzz"]})


def test_pinned_unknown_split_raises():
    with pytest.raises(ValueError, match="not one of"):
        split_groups(["a"], FR, "x", pinned={"holdout": ["a"]})


def test_group_pinned_to_two_splits_raises():
    with pytest.raises(ValueError, match="both"):
        split_groups(["a"], FR, "x", pinned={"dev": ["a"], "test": ["a"]})


def test_assert_disjoint_detects_overlap():
    with pytest.raises(AssertionError):
        assert_disjoint({"dev": ["a", "b"], "test": ["b"]})
