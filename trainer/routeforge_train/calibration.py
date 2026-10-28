"""Isotonic-binned calibration with Wilson lower bounds.

Serving maps a raw score to a block by binary search with no interpolation; Go reimplements
`Calibration.lookup`, so every comparison here is exact (integers or `Fraction`). The method is
explained in docs/routing.md, "Calibration and lower bounds".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from sklearn.linear_model import LogisticRegression

MIN_BLOCK = 40
WILSON_Z = 1.645
_PLATT_EPS = 1e-6


def wilson_lcb(k: int, n: int, z: float = WILSON_Z) -> float:
    """One-sided Wilson lower bound for k successes in n trials, clamped to [0, 1]."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, n], got k={k}, n={n}")
    if k == 0:
        return 0.0
    p = k / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return min(1.0, max(0.0, (centre - spread) / (1 + z2 / n)))


@dataclass(frozen=True)
class Block:
    lo: float
    hi: float
    n: int
    k: int
    p: float
    lcb: float


@dataclass(frozen=True)
class Calibration:
    blocks: tuple[Block, ...]

    def lookup(self, raw: np.ndarray) -> np.ndarray:
        """Index of the last block with lo <= raw; scores below the first lo map to block 0."""
        los = np.array([b.lo for b in self.blocks], dtype=np.float64)
        idx = np.searchsorted(los, np.asarray(raw, dtype=np.float64), side="right") - 1
        return np.clip(idx, 0, None)

    def p_of(self, raw: np.ndarray) -> np.ndarray:
        return np.array([b.p for b in self.blocks], dtype=np.float64)[self.lookup(raw)]

    def lcb_of(self, raw: np.ndarray) -> np.ndarray:
        return np.array([b.lcb for b in self.blocks], dtype=np.float64)[self.lookup(raw)]

    def to_json(self) -> dict:
        return {
            "blocks": [
                {"lo": b.lo, "hi": b.hi, "n": b.n, "k": b.k, "p": b.p, "lcb": b.lcb}
                for b in self.blocks
            ]
        }

    @classmethod
    def from_json(cls, d: dict) -> Calibration:
        return cls(
            tuple(
                Block(float(b["lo"]), float(b["hi"]), int(b["n"]), int(b["k"]), b["p"], b["lcb"])
                for b in d["blocks"]
            )
        )


def _merge_target(blocks: list[list], i: int) -> int:
    """Neighbour to merge block i with: the one with the closer p, ties to the left."""
    if i == 0:
        return 1
    if i == len(blocks) - 1:
        return i - 1
    p = Fraction(blocks[i][2], blocks[i][1])
    dl = abs(Fraction(blocks[i - 1][2], blocks[i - 1][1]) - p)
    dr = abs(Fraction(blocks[i + 1][2], blocks[i + 1][1]) - p)
    return i - 1 if dl <= dr else i + 1


def fit_calibration(
    raw: np.ndarray, success: np.ndarray, min_block: int = MIN_BLOCK
) -> Calibration:
    """Weighted PAV over distinct raw scores, then merge undersized blocks."""
    order = np.argsort(raw, kind="stable")
    xs, ys = raw[order], success[order].astype(np.int64)
    uniq, start = np.unique(xs, return_index=True)  # identical raw scores never split
    n_g = np.diff(np.append(start, len(xs)))
    k_g = np.add.reduceat(ys, start)
    blocks: list[list] = []  # [lo, n, k]
    for lo, n, k in zip(uniq, n_g, k_g, strict=True):
        blocks.append([float(lo), int(n), int(k)])
        while len(blocks) > 1 and blocks[-2][2] * blocks[-1][1] >= blocks[-1][2] * blocks[-2][1]:
            _, n2, k2 = blocks.pop()
            blocks[-1][1] += n2
            blocks[-1][2] += k2
    while len(blocks) > 1 and min(b[1] for b in blocks) < min_block:
        i = min(range(len(blocks)), key=lambda j: (blocks[j][1], j))
        j = _merge_target(blocks, i)
        a, b = min(i, j), max(i, j)
        blocks[a] = [blocks[a][0], blocks[a][1] + blocks[b][1], blocks[a][2] + blocks[b][2]]
        del blocks[b]
    los = [0.0] + [b[0] for b in blocks[1:]]
    his = [*los[1:], 1.0]
    return Calibration(
        tuple(
            Block(lo, hi, n, k, k / n, wilson_lcb(k, n))
            for lo, hi, (_, n, k) in zip(los, his, blocks, strict=True)
        )
    )


def _logit(raw: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(raw, dtype=np.float64), _PLATT_EPS, 1 - _PLATT_EPS)
    return np.log(clipped / (1 - clipped))


@dataclass(frozen=True)
class Platt:
    a: float
    b: float

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * _logit(raw) + self.b)))


def fit_platt(raw: np.ndarray, success: np.ndarray) -> Platt:
    """Platt scaling on the logit of the raw score; offline comparison only."""
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(_logit(raw).reshape(-1, 1), np.asarray(success).astype(np.int64))
    return Platt(float(model.coef_[0, 0]), float(model.intercept_[0]))
