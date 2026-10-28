"""Keyword-rule baseline: unigram keywords per family, cheapest adequate model.

Evaluated offline only, never served; Go reproduces `classify` from the exported rules.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from routeforge_train.features import tokenize

KEYWORD_MIN_DF = 20
KEYWORD_MIN_PRECISION = 0.9
KEYWORD_MAX = 25


def _token_set(text: str) -> set[str]:
    return {t.decode("ascii") for t in tokenize(text.encode("utf-8"))}


@dataclass(frozen=True)
class KeywordRules:
    family_keywords: dict[str, tuple[str, ...]]
    family_model: dict[str, str]
    family_order: tuple[str, ...]
    default_family: str

    def classify(self, text: str) -> str:
        tokens = _token_set(text)
        best_family = self.default_family
        best_count = 0
        for family in self.family_order:
            count = sum(1 for k in self.family_keywords[family] if k in tokens)
            if count > best_count:
                best_family, best_count = family, count
        return best_family

    def to_json(self) -> dict:
        return {
            "family_keywords": {f: list(k) for f, k in self.family_keywords.items()},
            "family_model": dict(self.family_model),
            "family_order": list(self.family_order),
            "default_family": self.default_family,
        }

    @classmethod
    def from_json(cls, d: dict) -> KeywordRules:
        return cls(
            family_keywords={f: tuple(k) for f, k in d["family_keywords"].items()},
            family_model=dict(d["family_model"]),
            family_order=tuple(d["family_order"]),
            default_family=d["default_family"],
        )


def _family_model(
    rows: np.ndarray, success: np.ndarray, exp_cost: np.ndarray, pool: Sequence[str]
) -> str:
    n_f = len(rows)
    wins = success[rows].sum(axis=0).astype(np.int64)
    max_k = int(wins.max())
    mean_cost = exp_cost[rows].mean(axis=0)
    qualifiers = [m for m in range(len(pool)) if 100 * int(wins[m]) >= 100 * max_k - 2 * n_f]
    return pool[min(qualifiers, key=lambda m: (mean_cost[m], m))]


def fit_keyword_rules(
    texts: Sequence[str],
    families: Sequence[str],
    success: np.ndarray,
    exp_cost: np.ndarray,
    pool: Sequence[str],
) -> KeywordRules:
    if not (len(texts) == len(families) == success.shape[0] == exp_cost.shape[0]):
        raise ValueError("texts, families, success and exp_cost must have the same length")
    if len(pool) != success.shape[1] or len(pool) != exp_cost.shape[1]:
        raise ValueError("success and exp_cost must have one column per pool model")
    df: Counter[str] = Counter()
    df_family: dict[str, Counter[str]] = {}
    counts: Counter[str] = Counter()
    for text, family in zip(texts, families, strict=True):
        tokens = _token_set(text)
        df.update(tokens)
        df_family.setdefault(family, Counter()).update(tokens)
        counts[family] += 1
    order = tuple(sorted(counts, key=lambda f: (-counts[f], f)))
    family_keywords: dict[str, tuple[str, ...]] = {}
    family_model: dict[str, str] = {}
    families_arr = np.array(list(families))
    for family in order:
        eligible = [
            t
            for t, n in df_family[family].items()
            if df[t] >= KEYWORD_MIN_DF and n / df[t] >= KEYWORD_MIN_PRECISION
        ]
        eligible.sort(key=lambda t: (-df_family[family][t], t))
        family_keywords[family] = tuple(eligible[:KEYWORD_MAX])
        rows = np.flatnonzero(families_arr == family)
        family_model[family] = _family_model(rows, success, exp_cost, pool)
    return KeywordRules(family_keywords, family_model, order, order[0])
