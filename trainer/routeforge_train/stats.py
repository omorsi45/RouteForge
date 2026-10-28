"""Paired bootstrap, exact tests and intervals for the offline evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import scipy.stats

from routeforge_train.metrics import StrategyOutcome, summarize

N_RESAMPLES = 10_000
SEED = 20261026
TOST_MARGIN = 0.02
_SIGNIFICANCE = 0.05


def bootstrap_totals(
    strata: np.ndarray,
    values: np.ndarray,
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    chunk: int = 500,
) -> np.ndarray:
    """Column sums of `values` over `n_resamples` bootstrap resamples, shape (n_resamples, K).

    Each resample draws rows with replacement inside every stratum, so stratum sizes are
    preserved exactly. The RNG call order fixes reproducibility.
    """
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"values must be 2-D (rows, columns), got shape {values.shape}")
    if len(strata) != values.shape[0]:
        raise ValueError(f"strata has {len(strata)} rows, values has {values.shape[0]}")
    if values.shape[0] == 0:
        raise ValueError("cannot bootstrap no rows")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")
    if chunk < 1:
        raise ValueError(f"chunk must be at least 1, got {chunk}")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in sorted(set(strata.tolist()))]
    out = np.empty((n_resamples, values.shape[1]))
    for start in range(0, n_resamples, chunk):
        m = min(chunk, n_resamples - start)
        tot = np.zeros((m, values.shape[1]))
        for idx in groups:
            counts = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=m)
            tot += counts @ values[idx]
        out[start : start + m] = tot
    return out


def _check_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval ignoring NaN samples; (nan, nan) when none are finite."""
    _check_level(level)
    finite = np.asarray(samples, dtype=np.float64)
    finite = finite[~np.isnan(finite)]
    if finite.size == 0:
        return math.nan, math.nan
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(finite, [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_p_value(diff: np.ndarray) -> float:
    """Two-sided bootstrap p-value of a difference, +1 corrected; NaN samples are skipped."""
    d = np.asarray(diff, dtype=np.float64)
    d = d[~np.isnan(d)]
    if d.size == 0:
        raise ValueError("no finite bootstrap samples")
    at_most_zero = int(np.count_nonzero(d <= 0))
    at_least_zero = int(np.count_nonzero(d >= 0))
    return float(min(1.0, 2 * min(at_most_zero + 1, at_least_zero + 1) / (d.size + 1)))


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise quotient, NaN where the denominator is zero."""
    out = np.full(np.shape(numerator), np.nan)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def _finite_or_none(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def _interval(estimate: float | None, samples: np.ndarray, *, with_p: bool) -> dict:
    """Estimate, percentile CI and optionally p-value; undefined parts are None."""
    low, high = percentile_ci(samples)
    out: dict = {
        "estimate": estimate,
        "ci_low": _finite_or_none(low),
        "ci_high": _finite_or_none(high),
    }
    if with_p:
        has_samples = bool(np.isfinite(samples).any())
        out["p_value"] = bootstrap_p_value(samples) if has_samples else None
    return out


def compare(
    a: StrategyOutcome, b: StrategyOutcome, totals_a: np.ndarray, totals_b: np.ndarray
) -> dict:
    """Paired comparison of `a` against `b`, with diff = a - b.

    Each `totals_*` is the (B, 2) slice of `bootstrap_totals` holding the [success, cost]
    columns of that strategy, drawn with the same resamples. Resamples with no successes leave
    cost per success undefined; they are counted in `degenerate_resamples` and left out of the
    cost-per-success intervals.
    """
    n = len(a.success)
    if len(b.success) != n:
        raise ValueError(f"a has {n} rows, b has {len(b.success)}; they must have the same length")
    totals_a = np.asarray(totals_a, dtype=np.float64)
    totals_b = np.asarray(totals_b, dtype=np.float64)
    if totals_a.ndim != 2 or totals_a.shape[1] != 2 or totals_a.shape != totals_b.shape:
        raise ValueError(
            f"totals must both have shape (B, 2), got {totals_a.shape} and {totals_b.shape}"
        )
    sum_a, sum_b = summarize(a), summarize(b)
    cps_a = _ratio(totals_a[:, 1], totals_a[:, 0])
    cps_b = _ratio(totals_b[:, 1], totals_b[:, 0])
    accuracy_samples = (totals_a[:, 0] - totals_b[:, 0]) / n
    cps_samples = cps_a - cps_b
    reduction_samples = 1.0 - _ratio(cps_a, cps_b)

    point_a, point_b = sum_a["cost_per_success_usd"], sum_b["cost_per_success_usd"]
    defined = point_a is not None and point_b is not None
    return {
        "accuracy_diff": _interval(
            sum_a["accuracy"] - sum_b["accuracy"], accuracy_samples, with_p=True
        ),
        "cost_per_success_diff": _interval(
            point_a - point_b if defined else None, cps_samples, with_p=True
        ),
        "relative_cps_reduction": _interval(
            1.0 - point_a / point_b if defined and point_b != 0 else None,
            reduction_samples,
            with_p=False,
        ),
        "degenerate_resamples": int(np.count_nonzero(np.isnan(cps_a) | np.isnan(cps_b))),
    }


def mcnemar_exact(a_success: np.ndarray, b_success: np.ndarray) -> dict:
    """Exact two-sided McNemar test; `b` counts prompts only `a` solved, `c` only `b` solved."""
    a = np.asarray(a_success, dtype=bool)
    b = np.asarray(b_success, dtype=bool)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("a_success and b_success must be 1-D arrays of the same length")
    only_a = int(np.count_nonzero(a & ~b))
    only_b = int(np.count_nonzero(~a & b))
    discordant = only_a + only_b
    if discordant == 0:
        p_value = 1.0
    else:
        tail = float(scipy.stats.binom.cdf(min(only_a, only_b), discordant, 0.5))
        p_value = min(1.0, 2 * tail)
    return {"b": only_a, "c": only_b, "p_value": p_value}


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the original order."""
    ps = [float(p) for p in p_values]
    for p in ps:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value {p!r} is outside [0, 1]")
    m = len(ps)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: (ps[i], i))):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def tost(diff_samples: np.ndarray, estimate: float, margin: float = TOST_MARGIN) -> dict:
    """Two one-sided tests of a bootstrap difference against +/- `margin`.

    `non_inferior` rejects diff <= -margin; `equivalent` also rejects diff >= margin.
    """
    d = np.asarray(diff_samples, dtype=np.float64)
    if d.size == 0:
        raise ValueError("tost needs bootstrap samples, got no samples")
    if not np.isfinite(d).all():
        raise ValueError("tost needs finite bootstrap samples")
    p_lower = (int(np.count_nonzero(d <= -margin)) + 1) / (d.size + 1)
    p_upper = (int(np.count_nonzero(d >= margin)) + 1) / (d.size + 1)
    p_value = max(p_lower, p_upper)
    low, high = np.percentile(d, [5, 95])
    return {
        "estimate": float(estimate),
        "margin": float(margin),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_value": p_value,
        "equivalent": p_value <= _SIGNIFICANCE,
        "non_inferior": p_lower <= _SIGNIFICANCE,
        "ci90": [float(low), float(high)],
    }


def clopper_pearson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Two-sided exact binomial interval for k events in n trials."""
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n >= 1, got k={k}, n={n}")
    _check_level(level)
    tail = (1 - level) / 2
    lower = 0.0 if k == 0 else float(scipy.stats.beta.ppf(tail, k, n - k + 1))
    upper = 1.0 if k == n else float(scipy.stats.beta.ppf(1 - tail, k + 1, n - k))
    return lower, upper
