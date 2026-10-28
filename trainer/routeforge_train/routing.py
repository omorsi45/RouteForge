"""Offline routing rules: the quality decision rule and the reference strategies.

The Go gateway reimplements `route_quality` (`internal/router/engine.go`), so tie handling is
fixed here: candidates are ordered by expected cost with ties kept in pool order, and the first
with lcb >= lambda wins.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from routeforge_train.prices import PriceTable
from routeforge_train.rules import KeywordRules

REASON_PASSING = "cheapest_passing"
REASON_ESCALATED = "escalated_no_model_passed"


def estimated_prompt_tokens(prompt_bytes: np.ndarray) -> np.ndarray:
    return (np.asarray(prompt_bytes, dtype=np.int64) + 3) // 4


def expected_cost_usd(
    prompt_bytes: np.ndarray,
    pool: Sequence[str],
    prices: PriceTable,
    expected_output_tokens: Mapping[str, float],
) -> np.ndarray:
    """(n, M) expected USD cost: (est_prompt_tokens * input + expected_output * output) / 1e6."""
    tokens = estimated_prompt_tokens(prompt_bytes).astype(np.float64)
    cost = np.empty((len(tokens), len(pool)), dtype=np.float64)
    for m, name in enumerate(pool):
        price = prices.price(name)
        cost[:, m] = (
            tokens * price.input_per_mtok + expected_output_tokens[name] * price.output_per_mtok
        ) / 1e6
    return cost


def route_quality(
    lcb: np.ndarray, exp_cost: np.ndarray, strongest: int, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cheapest model whose lcb >= lam, else the strongest. Returns (chosen, escalated)."""
    order = np.argsort(exp_cost, axis=1, kind="stable")
    passing = np.take_along_axis(lcb, order, axis=1) >= lam
    any_pass = passing.any(axis=1)
    first = np.argmax(passing, axis=1)
    picked = order[np.arange(len(order)), first]
    chosen = np.where(any_pass, picked, strongest)
    return chosen.astype(np.int64), ~any_pass


def route_cheapest(exp_cost: np.ndarray) -> np.ndarray:
    return np.argmin(exp_cost, axis=1)


def route_static(
    texts_or_families: Sequence[str],
    rules: KeywordRules,
    pool: Sequence[str],
    *,
    classify: bool,
) -> np.ndarray:
    """Per-family model from the keyword rules; classify the text, or take the given family."""
    index = {name: i for i, name in enumerate(pool)}
    families = [rules.classify(t) if classify else t for t in texts_or_families]
    return np.array([index[rules.family_model[f]] for f in families], dtype=np.int64)


def regret_mask(chosen: np.ndarray, success: np.ndarray, strongest: int) -> np.ndarray:
    """Regret event: the routed model fails and the strongest model succeeds."""
    routed_ok = success[np.arange(len(chosen)), chosen]
    return ~routed_ok & success[:, strongest]


def strongest_index(success: np.ndarray, cost: np.ndarray) -> int:
    """Highest mean success; ties by lower mean recorded cost, then pool order."""
    wins = success.sum(axis=0)
    mean_cost = cost.mean(axis=0)
    best = int(wins.max())
    candidates = [m for m in range(success.shape[1]) if int(wins[m]) == best]
    return min(candidates, key=lambda m: (mean_cost[m], m))
