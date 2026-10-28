import json

import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from routeforge_train.lr import (
    C_GRID,
    CV_FOLDS,
    SEED,
    WEIGHT_SIG_DIGITS,
    LRModel,
    choose_c,
    fit_lr,
    round_sig,
)
from tests.predictor_fixture import make_problem


def test_constants():
    assert C_GRID == (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    assert CV_FOLDS == 5
    assert SEED == 20261026
    assert WEIGHT_SIG_DIGITS == 6


def test_round_sig_six_digits():
    got = round_sig(np.array([0.123456789, -1234567.0, 0.0, 9.9999996e-5]))
    assert got.tolist() == [0.123457, -1234570.0, 0.0, 0.0001]


def test_choose_c_returns_grid_and_argmin():
    rng = np.random.default_rng(3)
    x = sparse.csr_matrix(rng.normal(size=(200, 10)))
    y = (x[:, 0].toarray()[:, 0] + rng.normal(scale=1.5, size=200) > 0).astype(np.int64)
    c, losses = choose_c(x, y)
    assert tuple(losses) == C_GRID
    assert c == min(C_GRID, key=lambda g: (losses[g], g))
    assert (c, losses) == choose_c(x, y)


def test_choose_c_ties_go_to_smaller_c():
    x = sparse.csr_matrix(np.zeros((40, 3)))
    y = np.array([0, 1] * 20)
    c, losses = choose_c(x, y, grid=(0.5, 0.1, 2.0))
    assert len(set(losses.values())) == 1
    assert c == 0.1


def test_fit_lr_weights_rounded_and_split():
    p = make_problem()
    m = fit_lr(p.design, p.y, p.vocab, c=1.0)
    assert len(m.dense_weights) == 8
    assert m.intercept == float(f"{m.intercept:.5e}")
    for w in (m.sparse_weights, m.dense_weights):
        assert np.array_equal(w, round_sig(w))
    assert len(m.sparse_indices) == len(m.sparse_weights)
    assert np.all(np.diff(m.sparse_indices) > 0)
    assert set(m.sparse_indices.tolist()) <= set(p.vocab.tolist())
    assert np.all(m.sparse_weights != 0.0)


def test_lr_raw_matches_sklearn_within_rounding():
    p = make_problem()
    m = fit_lr(p.design, p.y, p.vocab, c=1.0)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000).fit(p.design, p.y)
    want = clf.predict_proba(p.design)[:, 1]
    got = m.raw(p.hashed, p.dense_z)
    assert got.shape == want.shape
    assert np.max(np.abs(got - want)) < 1e-4


def test_lr_json_round_trip_exact():
    p = make_problem()
    m = fit_lr(p.design, p.y, p.vocab, c=1.0)
    again = LRModel.from_json(json.loads(json.dumps(m.to_json())))
    assert np.array_equal(again.raw(p.hashed, p.dense_z), m.raw(p.hashed, p.dense_z))
    assert set(m.to_json()) == {"intercept", "sparse_indices", "sparse_weights", "dense_weights"}
    assert isinstance(m.to_json()["sparse_indices"], list)


def test_lr_raw_without_sparse_weights():
    p = make_problem(n=20)
    m = LRModel(0.5, np.array([], dtype=np.int64), np.array([]), np.zeros(8))
    got = m.raw(p.hashed, p.dense_z)
    assert np.allclose(got, 1 / (1 + np.exp(-0.5)))


def test_fit_lr_is_deterministic():
    p = make_problem()
    a = fit_lr(p.design, p.y, p.vocab, c=0.3).to_json()
    b = fit_lr(p.design, p.y, p.vocab, c=0.3).to_json()
    assert a == b
    assert choose_c(p.design, p.y) == choose_c(p.design, p.y)


def test_fit_lr_rejects_single_class_labels():
    p = make_problem()
    with pytest.raises(ValueError, match="single class"):
        fit_lr(p.design, np.ones(len(p.y), dtype=np.int64), p.vocab, c=1.0)
