"""Learn-then-Test certification of the lambda threshold (spec 8.5).

Fixed-sequence testing from lambda = 1.00 downward: reject H0 "regret rate > alpha" while
P(Binomial(n, alpha) <= k_regret) <= delta, and certify the last rejected lambda.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.stats

from routeforge_train.routing import regret_mask, route_quality

LAMBDA_GRID = tuple((100 - i) / 100 for i in range(101))
SENTINEL_LAMBDA = 1.01
DELTA = 0.05
PROFILES = (("strict", 0.02), ("balanced", 0.05), ("economy", 0.10))


@dataclass(frozen=True)
class ProfileCert:
    name: str
    alpha: float
    delta: float
    lambda_: float
    cert_n: int
    cert_regret: int
    regret_ucb: float

    @property
    def sentinel(self) -> bool:
        return self.lambda_ == SENTINEL_LAMBDA

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "delta": self.delta,
            "lambda": self.lambda_,
            "cert_n": self.cert_n,
            "cert_regret": self.cert_regret,
            "regret_ucb": self.regret_ucb,
        }

    @classmethod
    def from_json(cls, d: dict) -> ProfileCert:
        return cls(
            name=d["name"],
            alpha=d["alpha"],
            delta=d["delta"],
            lambda_=d["lambda"],
            cert_n=int(d["cert_n"]),
            cert_regret=int(d["cert_regret"]),
            regret_ucb=d["regret_ucb"],
        )


def regret_counts(
    lcb: np.ndarray,
    exp_cost: np.ndarray,
    success: np.ndarray,
    strongest: int,
    grid: Sequence[float] = LAMBDA_GRID,
) -> list[int]:
    """Regret events of the quality rule at each lambda of the grid."""
    counts = []
    for lam in grid:
        chosen, _ = route_quality(lcb, exp_cost, strongest, lam)
        counts.append(int(regret_mask(chosen, success, strongest).sum()))
    return counts


def ltt_lambda(
    n: int, counts: Sequence[int], alpha: float, delta: float = DELTA
) -> tuple[float, int]:
    """Certified lambda and its regret count; (SENTINEL_LAMBDA, 0) when 1.00 is not rejected."""
    certified, k_at = None, 0
    for lam, k in zip(LAMBDA_GRID, counts, strict=True):
        if scipy.stats.binom.cdf(k, n, alpha) <= delta:
            certified, k_at = lam, k
        else:
            break
    return (SENTINEL_LAMBDA, 0) if certified is None else (certified, k_at)


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound for k events in n trials."""
    if k == n:
        return 1.0
    return float(scipy.stats.beta.ppf(confidence, k + 1, n - k))


def certify_profiles(
    lcb: np.ndarray, exp_cost: np.ndarray, success: np.ndarray, strongest: int
) -> tuple[ProfileCert, ...]:
    n = int(lcb.shape[0])
    counts = regret_counts(lcb, exp_cost, success, strongest)
    certs = []
    for name, alpha in PROFILES:
        lam, k = ltt_lambda(n, counts, alpha)
        certs.append(ProfileCert(name, alpha, DELTA, lam, n, k, clopper_pearson_upper(k, n)))
    return tuple(certs)
