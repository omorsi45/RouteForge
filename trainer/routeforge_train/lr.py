"""Per-model logistic regression predictor (spec 8.2).

Fitting happens on the compact design matrix (vocabulary columns plus the eight dense
columns); serving scores the hashed block through the retained sparse weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from routeforge_train.features import DENSE_NAMES

C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
CV_FOLDS = 5
SEED = 20261026
WEIGHT_SIG_DIGITS = 6
MAX_ITER = 5000


def round_sig(values: np.ndarray, digits: int = WEIGHT_SIG_DIGITS) -> np.ndarray:
    """Round each element to `digits` significant digits via its decimal text form."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    out = np.array([float(f"{v:.{digits - 1}e}") for v in flat], dtype=np.float64)
    return out.reshape(np.shape(values))


@dataclass(frozen=True)
class LRModel:
    intercept: float
    sparse_indices: np.ndarray
    sparse_weights: np.ndarray
    dense_weights: np.ndarray

    def raw(self, hashed: csr_matrix, dense_z: np.ndarray) -> np.ndarray:
        z = dense_z @ self.dense_weights + self.intercept
        if len(self.sparse_indices):
            z = hashed[:, self.sparse_indices] @ self.sparse_weights + z
        return expit(z)

    def to_json(self) -> dict:
        return {
            "intercept": float(self.intercept),
            "sparse_indices": [int(i) for i in self.sparse_indices],
            "sparse_weights": [float(w) for w in self.sparse_weights],
            "dense_weights": [float(w) for w in self.dense_weights],
        }

    @classmethod
    def from_json(cls, d: dict) -> LRModel:
        return cls(
            intercept=float(d["intercept"]),
            sparse_indices=np.array(d["sparse_indices"], dtype=np.int64),
            sparse_weights=np.array(d["sparse_weights"], dtype=np.float64),
            dense_weights=np.array(d["dense_weights"], dtype=np.float64),
        )


def _classifier(c: float) -> LogisticRegression:
    return LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER)


def choose_c(
    X: csr_matrix,
    y: np.ndarray,
    grid: tuple[float, ...] = C_GRID,
    folds: int = CV_FOLDS,
    seed: int = SEED,
) -> tuple[float, dict[float, float]]:
    """Mean held-out log loss per C; the minimum wins and ties go to the smaller C."""
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(X, y))
    losses: dict[float, float] = {}
    for c in grid:
        fold_losses = []
        for train, held in splits:
            clf = _classifier(c).fit(X[train], y[train])
            fold_losses.append(log_loss(y[held], clf.predict_proba(X[held])[:, 1], labels=[0, 1]))
        losses[c] = float(np.mean(fold_losses))
    best = min(grid, key=lambda g: (losses[g], g))
    return best, losses


def fit_lr(X: csr_matrix, y: np.ndarray, vocab: np.ndarray, c: float) -> LRModel:
    if len(np.unique(y)) < 2:
        raise ValueError("training labels have a single class")
    n_dense = len(DENSE_NAMES)
    if X.shape[1] != len(vocab) + n_dense:
        raise ValueError(f"design matrix has {X.shape[1]} columns, expected {len(vocab) + n_dense}")
    clf = _classifier(c).fit(X, y)
    coef = clf.coef_[0]
    sparse_weights = round_sig(coef[: len(vocab)])
    keep = sparse_weights != 0.0
    return LRModel(
        intercept=float(round_sig(clf.intercept_)[0]),
        sparse_indices=np.asarray(vocab, dtype=np.int64)[keep],
        sparse_weights=sparse_weights[keep],
        dense_weights=round_sig(coef[len(vocab) :]),
    )
