"""Per-model gradient boosting predictor (spec 8.2).

Trees are exported as flat arrays over global feature ids. Serving compares
`float64(float32(x)) <= threshold`, because scikit-learn evaluates trees on float32 inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.ensemble import GradientBoostingClassifier

from routeforge_train.features import HASH_DIM, global_feature

GBM_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "random_state": 20261026,
}


@dataclass(frozen=True)
class Tree:
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray


@dataclass(frozen=True)
class GBMModel:
    init: float
    learning_rate: float
    trees: tuple[Tree, ...]

    def raw(self, hashed: csr_matrix, dense_z: np.ndarray) -> np.ndarray:
        n = hashed.shape[0]
        used = sorted({int(f) for t in self.trees for f in t.feature if f >= 0})
        hashed_used = [f for f in used if f < HASH_DIM]
        columns: dict[int, np.ndarray] = {}
        if hashed_used:
            block = hashed[:, hashed_used].toarray()
            columns.update({f: block[:, j] for j, f in enumerate(hashed_used)})
        for f in used:
            if f >= HASH_DIM:
                columns[f] = dense_z[:, f - HASH_DIM]
        # float32 first, then back to float64 for the comparison, as scikit-learn does.
        x32 = {f: c.astype(np.float32).astype(np.float64) for f, c in columns.items()}
        out = np.full(n, self.init)
        for t in self.trees:
            node = np.zeros(n, dtype=np.int64)
            while True:
                active = np.flatnonzero(t.left[node] != -1)
                if active.size == 0:
                    break
                at = node[active]
                feats = t.feature[at]
                go_left = np.empty(active.size, dtype=bool)
                for f in np.unique(feats):
                    sel = feats == f
                    go_left[sel] = x32[int(f)][active[sel]] <= t.threshold[at[sel]]
                node[active] = np.where(go_left, t.left[at], t.right[at])
            out = out + self.learning_rate * t.value[node]
        return expit(out)

    def to_json(self) -> dict:
        return {
            "init": float(self.init),
            "learning_rate": float(self.learning_rate),
            "feature_dtype": "float32",
            "trees": [
                {
                    "feature": [int(v) for v in t.feature],
                    "threshold": [float(v) for v in t.threshold],
                    "left": [int(v) for v in t.left],
                    "right": [int(v) for v in t.right],
                    "value": [float(v) for v in t.value],
                }
                for t in self.trees
            ],
        }

    @classmethod
    def from_json(cls, d: dict) -> GBMModel:
        trees = tuple(
            Tree(
                feature=np.array(t["feature"], dtype=np.int64),
                threshold=np.array(t["threshold"], dtype=np.float64),
                left=np.array(t["left"], dtype=np.int64),
                right=np.array(t["right"], dtype=np.int64),
                value=np.array(t["value"], dtype=np.float64),
            )
            for t in d["trees"]
        )
        return cls(init=float(d["init"]), learning_rate=float(d["learning_rate"]), trees=trees)


def _export_tree(tree_, vocab: np.ndarray) -> Tree:
    left = np.asarray(tree_.children_left, dtype=np.int64)
    right = np.asarray(tree_.children_right, dtype=np.int64)
    leaf = left == -1
    feature = np.array(
        [
            -1 if is_leaf else global_feature(int(f), vocab)
            for f, is_leaf in zip(tree_.feature, leaf, strict=True)
        ],
        dtype=np.int64,
    )
    threshold = np.where(leaf, 0.0, np.asarray(tree_.threshold, dtype=np.float64))
    left = np.where(leaf, -1, left)
    right = np.where(leaf, -1, right)
    return Tree(feature, threshold, left, right, np.asarray(tree_.value[:, 0, 0], dtype=np.float64))


def fit_gbm(X: csr_matrix, y: np.ndarray, vocab: np.ndarray) -> GBMModel:
    if len(np.unique(y)) < 2:
        raise ValueError("training labels have a single class")
    clf = GradientBoostingClassifier(**GBM_PARAMS).fit(X, y)
    p1 = float(clf.init_.class_prior_[1])
    trees = tuple(
        _export_tree(clf.estimators_[t, 0].tree_, vocab) for t in range(len(clf.estimators_))
    )
    return GBMModel(
        init=math.log(p1 / (1 - p1)), learning_rate=float(clf.learning_rate), trees=trees
    )
