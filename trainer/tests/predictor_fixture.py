"""Tiny synthetic classification problem shared by the predictor tests. No real data."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from routeforge_train.features import (
    DenseScaler,
    design_matrix,
    featurize,
    select_vocab,
)


@dataclass(frozen=True)
class Problem:
    texts: list[str]
    hashed: sparse.csr_matrix
    dense_z: np.ndarray
    vocab: np.ndarray
    design: sparse.csr_matrix
    y: np.ndarray


def make_problem(n: int = 300, seed: int = 7) -> Problem:
    rng = np.random.default_rng(seed)
    words = [f"w{i}" for i in range(40)]
    texts = []
    has_signal = np.zeros(n, dtype=bool)
    for i in range(n):
        picked = rng.choice(len(words), size=int(rng.integers(5, 12)), replace=False)
        texts.append(" ".join(words[j] for j in picked))
        has_signal[i] = bool({0, 1} & {int(j) for j in picked})
    feats = featurize(texts)
    dense_z = DenseScaler.fit(feats.dense_raw).transform(feats.dense_raw)
    dense_z = dense_z + rng.normal(size=dense_z.shape)
    score = has_signal.astype(float) + 0.5 * dense_z[:, 0] + rng.normal(scale=0.3, size=n)
    y = (score > 0.7).astype(np.int64)
    vocab = select_vocab(feats.hashed, min_df=2)
    design = design_matrix(feats.hashed, dense_z, vocab)
    return Problem(texts, feats.hashed, dense_z, vocab, design, y)
