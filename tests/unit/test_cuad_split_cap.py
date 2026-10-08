"""
cap_split_contracts: lets a dev ablation be large enough to resolve a small effect (50 questions give
a +/-0.11 MRR interval) without ingesting all 346 dev contracts, while making it impossible to
accidentally evaluate a partial held-out test split.
"""
import pytest

from lexis.evaluation.dataset.cuad_split import cap_split_contracts

CONTRACTS = [{"title": f"c{i}"} for i in range(10)]


def test_none_means_no_cap_on_any_split():
    assert cap_split_contracts(CONTRACTS, None, "dev") == CONTRACTS
    assert cap_split_contracts(CONTRACTS, None, "test") == CONTRACTS


def test_dev_takes_the_first_n_in_order():
    assert cap_split_contracts(CONTRACTS, 3, "dev") == CONTRACTS[:3]


def test_cap_larger_than_the_split_returns_everything():
    assert cap_split_contracts(CONTRACTS, 99, "dev") == CONTRACTS


def test_cap_is_deterministic():
    assert cap_split_contracts(CONTRACTS, 4, "dev") == cap_split_contracts(CONTRACTS, 4, "dev")


def test_test_split_can_never_be_capped():
    with pytest.raises(ValueError, match="dev split"):
        cap_split_contracts(CONTRACTS, 3, "test")


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_cap_rejected(bad):
    with pytest.raises(ValueError, match="positive"):
        cap_split_contracts(CONTRACTS, bad, "dev")


def test_offset_selects_a_disjoint_slice_in_split_order():
    contracts = [{"title": f"c{i}"} for i in range(10)]
    first = cap_split_contracts(contracts, 4, "dev", offset=0)
    second = cap_split_contracts(contracts, 4, "dev", offset=4)
    assert [c["title"] for c in first] == ["c0", "c1", "c2", "c3"]
    assert [c["title"] for c in second] == ["c4", "c5", "c6", "c7"]
    assert cap_split_contracts(contracts, 4, "dev", offset=8) == contracts[8:10]   # runs off the end safely


def test_offset_rules():
    contracts = [{"title": f"c{i}"} for i in range(5)]
    with pytest.raises(ValueError, match="requires --max-contracts"):
        cap_split_contracts(contracts, None, "dev", offset=2)
    with pytest.raises(ValueError, match="non-negative"):
        cap_split_contracts(contracts, 2, "dev", offset=-1)
    with pytest.raises(ValueError, match="dev split"):
        cap_split_contracts(contracts, 2, "test", offset=1)
