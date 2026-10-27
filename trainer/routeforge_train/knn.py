"""Per-model k-nearest-neighbor predictor (spec 8.2). Evaluated offline only, never served."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

K = 25


@dataclass(frozen=True)
class KNNModel:
    train_hashed: csr_matrix
    train_success: np.ndarray
    k: int

    def raw(self, hashed: csr_matrix) -> np.ndarray:
        """Shape (n, M): mean success of each model over the k nearest train rows."""
        nn = NearestNeighbors(n_neighbors=self.k, metric="cosine", algorithm="brute")
        nn.fit(self.train_hashed)
        neighbors = nn.kneighbors(hashed, return_distance=False)
        return self.train_success.astype(np.float64)[neighbors].mean(axis=1)


def fit_knn(hashed: csr_matrix, success: np.ndarray, k: int = K) -> KNNModel:
    n = hashed.shape[0]
    if success.shape[0] != n:
        raise ValueError(f"success has {success.shape[0]} rows, hashed has {n}")
    if k < 1 or k > n:
        raise ValueError(f"k={k} must be between 1 and the {n} train rows")
    return KNNModel(train_hashed=hashed, train_success=success, k=k)
