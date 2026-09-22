"""
Statistical toolkit for retrieval/generation evaluation (plan P0).

Pure numpy, deterministic given a seed, no dependency on any benchmark,
model, or retrieval component. Every "A beats B" claim in this project
should be produced by paired_comparison() over per-query scores so it
carries a confidence interval and a significance test instead of a bare
delta.

Design notes:
- Resampling is over QUERIES (the unit of independence), never over chunks.
- All functions take an explicit seed; the seed is part of the result so an
  artifact records exactly how its interval was produced.
- Bootstrap resampling is batched so memory stays bounded for large n.
"""
from dataclasses import dataclass
from itertools import product
from typing import List, Sequence

import numpy as np

_BATCH = 1000


@dataclass(frozen=True)
class CI:
    mean: float
    lo: float
    hi: float
    n: int
    alpha: float
    n_resamples: int
    seed: int


@dataclass(frozen=True)
class PairedResult:
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float          # mean(a - b)
    ci_lo: float              # bootstrap CI on mean_diff
    ci_hi: float
    p_value: float            # two-sided paired permutation (sign-flip) test
    cohens_dz: float          # mean_diff / std(diff); nan if std == 0
    n_improved: int           # a > b
    n_regressed: int          # a < b
    n_tied: int
    alpha: float
    n_resamples: int
    seed: int
    exact: bool               # True if the permutation test was exhaustive


def _as_array(values: Sequence[float], name: str = "values") -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _bootstrap_means(arr: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = arr.size
    out = np.empty(n_resamples, dtype=float)
    done = 0
    while done < n_resamples:
        b = min(_BATCH, n_resamples - done)
        idx = rng.integers(0, n, size=(b, n))
        out[done:done + b] = arr[idx].mean(axis=1)
        done += b
    return out


def bootstrap_ci(values: Sequence[float], n_resamples: int = 10000, alpha: float = 0.05, seed: int = 0) -> CI:
    """Percentile bootstrap CI for the mean of per-query scores."""
    arr = _as_array(values)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    means = _bootstrap_means(arr, n_resamples, seed)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(mean=float(arr.mean()), lo=float(lo), hi=float(hi), n=int(arr.size),
              alpha=alpha, n_resamples=n_resamples, seed=seed)


def paired_permutation_pvalue(diffs: Sequence[float], n_resamples: int = 10000, seed: int = 0,
                              max_exact_n: int = 16) -> tuple:
    """Two-sided sign-flip permutation test on paired differences.
    Exhaustive (exact) when the number of non-zero differences <= max_exact_n,
    otherwise Monte Carlo with add-one smoothing. Returns (p_value, exact)."""
    d = _as_array(diffs, "diffs")
    nz = d[d != 0]
    if nz.size == 0:
        return 1.0, True
    observed = abs(nz.mean())
    tol = 1e-12
    if nz.size <= max_exact_n:
        signs = np.array(list(product((-1.0, 1.0), repeat=nz.size)))
        stats = np.abs((signs * nz).mean(axis=1))
        return float((stats >= observed - tol).mean()), True
    rng = np.random.default_rng(seed)
    count = 0
    done = 0
    while done < n_resamples:
        b = min(_BATCH, n_resamples - done)
        signs = rng.choice((-1.0, 1.0), size=(b, nz.size))
        stats = np.abs((signs * nz).mean(axis=1))
        count += int((stats >= observed - tol).sum())
        done += b
    return (count + 1) / (n_resamples + 1), False


def paired_comparison(a: Sequence[float], b: Sequence[float], n_resamples: int = 10000,
                      alpha: float = 0.05, seed: int = 0) -> PairedResult:
    """Compare system A to system B on the SAME queries (paired).
    a[i] and b[i] must be the per-query scores for query i."""
    arr_a, arr_b = _as_array(a, "a"), _as_array(b, "b")
    if arr_a.size != arr_b.size:
        raise ValueError("a and b must have the same length (paired by query)")
    diffs = arr_a - arr_b
    ci = bootstrap_ci(diffs, n_resamples=n_resamples, alpha=alpha, seed=seed)
    p, exact = paired_permutation_pvalue(diffs, n_resamples=n_resamples, seed=seed)
    sd = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
    dz = float(diffs.mean() / sd) if sd > 0 else float("nan")
    return PairedResult(
        n=int(diffs.size), mean_a=float(arr_a.mean()), mean_b=float(arr_b.mean()),
        mean_diff=float(diffs.mean()), ci_lo=ci.lo, ci_hi=ci.hi, p_value=p, cohens_dz=dz,
        n_improved=int((diffs > 0).sum()), n_regressed=int((diffs < 0).sum()),
        n_tied=int((diffs == 0).sum()), alpha=alpha, n_resamples=n_resamples, seed=seed, exact=exact,
    )


def holm_correction(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (controls family-wise error across
    the many ablation comparisons a project like this runs)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def claim_supported(result: PairedResult) -> bool:
    """A gain is claimable only if the paired CI excludes zero on the
    positive side AND the permutation test rejects at the same alpha."""
    return result.ci_lo > 0 and result.p_value < result.alpha
