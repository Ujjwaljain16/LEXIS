import math

import numpy as np
import pytest

from lexis.evaluation.stats import (
    bootstrap_ci,
    claim_supported,
    holm_correction,
    paired_comparison,
    paired_permutation_pvalue,
)


def test_bootstrap_ci_contains_mean_and_is_ordered():
    ci = bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], n_resamples=2000, seed=1)
    assert ci.lo <= ci.mean <= ci.hi
    assert ci.n == 8


def test_bootstrap_ci_is_deterministic_for_a_seed():
    vals = list(np.linspace(0, 1, 30))
    assert bootstrap_ci(vals, seed=7) == bootstrap_ci(vals, seed=7)


def test_bootstrap_ci_differs_across_seeds_but_stays_close():
    vals = list(np.linspace(0, 1, 60))
    a, b = bootstrap_ci(vals, seed=1), bootstrap_ci(vals, seed=2)
    assert a.lo != b.lo
    assert abs(a.lo - b.lo) < 0.05


def test_bootstrap_ci_constant_values_collapse():
    ci = bootstrap_ci([0.5] * 20, n_resamples=500, seed=0)
    assert ci.lo == ci.hi == ci.mean == 0.5


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(0)
    small = rng.random(20)
    large = rng.random(2000)
    w_small = bootstrap_ci(small, seed=0).hi - bootstrap_ci(small, seed=0).lo
    w_large = bootstrap_ci(large, seed=0).hi - bootstrap_ci(large, seed=0).lo
    assert w_large < w_small


@pytest.mark.parametrize("bad", [[], [float("nan"), 1.0]])
def test_bootstrap_ci_rejects_empty_or_nonfinite(bad):
    with pytest.raises(ValueError):
        bootstrap_ci(bad)


def test_bootstrap_ci_rejects_bad_alpha():
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], alpha=1.5)


def test_paired_identical_systems_no_difference():
    a = [0.1, 0.5, 0.9, 0.3]
    r = paired_comparison(a, a, n_resamples=500, seed=0)
    assert r.mean_diff == 0 and r.n_tied == 4 and r.p_value == 1.0
    assert math.isnan(r.cohens_dz)
    assert not claim_supported(r)


def test_paired_clear_win_is_supported():
    b = np.linspace(0.1, 0.5, 40)
    a = b + 0.2
    r = paired_comparison(a, b, n_resamples=2000, seed=0)
    assert r.mean_diff == pytest.approx(0.2)
    assert r.ci_lo > 0 and r.p_value < 0.05 and r.n_improved == 40 and r.n_regressed == 0
    assert claim_supported(r)


def test_paired_noise_is_not_supported():
    rng = np.random.default_rng(3)
    b = rng.random(50)
    a = b + rng.normal(0, 0.3, 50)
    r = paired_comparison(a, b, n_resamples=2000, seed=0)
    assert not claim_supported(r) or r.ci_lo > 0  # never supported with a CI that spans 0
    if r.ci_lo <= 0:
        assert not claim_supported(r)


def test_paired_requires_equal_length():
    with pytest.raises(ValueError):
        paired_comparison([1, 2, 3], [1, 2])


def test_paired_counts_regressions():
    r = paired_comparison([1, 0, 1, 1], [0, 1, 1, 0], n_resamples=200, seed=0)
    assert (r.n_improved, r.n_regressed, r.n_tied) == (2, 1, 1)


def test_permutation_exact_small_n_matches_hand_calc():
    # 3 non-zero diffs of equal sign: only the all-same-sign flips (2 of 8) are as extreme
    p, exact = paired_permutation_pvalue([1.0, 1.0, 1.0])
    assert exact and p == pytest.approx(2 / 8)


def test_permutation_monte_carlo_for_large_n_is_deterministic():
    d = list(np.linspace(-1, 1.5, 40))
    a = paired_permutation_pvalue(d, n_resamples=2000, seed=5)
    b = paired_permutation_pvalue(d, n_resamples=2000, seed=5)
    assert a == b and a[1] is False


def test_permutation_all_zero_diffs():
    assert paired_permutation_pvalue([0.0, 0.0]) == (1.0, True)


def test_holm_monotone_and_bounded():
    adj = holm_correction([0.01, 0.04, 0.03, 0.5])
    assert all(0 <= x <= 1 for x in adj)
    assert adj[0] == pytest.approx(0.04)
    assert adj[3] == pytest.approx(0.5)
    assert all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03, 0.5]))


def test_holm_single_value_unchanged():
    assert holm_correction([0.2]) == [0.2]
