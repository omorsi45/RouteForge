"""Per-strategy summaries and calibration metrics for the offline evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

ECE_BINS = 15


@dataclass(frozen=True, eq=False)
class StrategyOutcome:
    """Per-prompt outcome of one routing strategy over the evaluated rows.

    The arrays are floats so that share-matched random, an average over many seeds, can carry
    fractional successes, costs and regrets.
    """

    name: str
    success: np.ndarray
    cost: np.ndarray
    regret: np.ndarray
    shares: dict[str, float]
    escalated: np.ndarray | None = None
    chosen: np.ndarray | None = None


def _mean(values: np.ndarray, n: int) -> float:
    return math.fsum(np.asarray(values, dtype=np.float64).tolist()) / n


def summarize(o: StrategyOutcome) -> dict:
    """Accuracy, cost, cost per success, regret and escalation of one strategy.

    `cost_per_success_usd` is None when nothing succeeded, so the dict stays valid JSON.
    """
    n = len(o.success)
    lengths = {len(o.success), len(o.cost), len(o.regret)}
    if o.escalated is not None:
        lengths.add(len(o.escalated))
    if len(lengths) != 1:
        raise ValueError("success, cost, regret and escalated must have the same length")
    if n == 0:
        raise ValueError("cannot summarize no rows")
    successes = math.fsum(np.asarray(o.success, dtype=np.float64).tolist())
    total = math.fsum(np.asarray(o.cost, dtype=np.float64).tolist())
    regrets = math.fsum(np.asarray(o.regret, dtype=np.float64).tolist())
    return {
        "n": n,
        "accuracy": successes / n,
        "successes": successes,
        "total_cost_usd": total,
        "cost_per_success_usd": total / successes if successes > 0 else None,
        "regret_count": regrets,
        "regret_rate": regrets / n,
        "escalation_rate": None if o.escalated is None else _mean(o.escalated, n),
        "shares": dict(o.shares),
    }


def _pair(success: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(success, dtype=np.float64)
    scores = np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or y.shape != scores.shape:
        raise ValueError(f"success {y.shape} and scores {scores.shape} must be matching 1-D arrays")
    if y.size == 0:
        raise ValueError("cannot score no rows")
    return y, scores


def auroc(success: np.ndarray, score: np.ndarray) -> float | None:
    """Area under the ROC curve; None when `success` has a single class."""
    y = np.asarray(success).astype(bool)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def brier(success: np.ndarray, p: np.ndarray) -> float:
    y, scores = _pair(success, p)
    return float(np.mean((scores - y) ** 2))


def reliability(success: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> list[dict]:
    """Equal-mass bins in score order; empty bins are skipped."""
    y, scores = _pair(success, p)
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    out = []
    for idx in np.array_split(np.argsort(scores, kind="stable"), bins):
        if idx.size == 0:
            continue
        out.append(
            {
                "n": int(idx.size),
                "mean_p": float(scores[idx].mean()),
                "frac_success": float(y[idx].mean()),
            }
        )
    return out


def ece(success: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error over equal-mass bins."""
    n = len(np.asarray(success))
    return math.fsum(
        b["n"] / n * abs(b["frac_success"] - b["mean_p"]) for b in reliability(success, p, bins)
    )
