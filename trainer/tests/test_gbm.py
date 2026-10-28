import json

import numpy as np
import pytest
from scipy import sparse
from scipy.special import expit
from sklearn.ensemble import GradientBoostingClassifier

from routeforge_train.features import HASH_DIM
from routeforge_train.gbm import GBM_PARAMS, GBMModel, Tree, fit_gbm
from tests.predictor_fixture import make_problem


def test_gbm_uses_spec_hyperparameters():
    assert GBM_PARAMS == {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 20261026,
    }


def test_gbm_model_matches_sklearn_predict_proba():
    p = make_problem()
    m = fit_gbm(p.design, p.y, p.vocab)
    clf = GradientBoostingClassifier(**GBM_PARAMS).fit(p.design, p.y)
    want = clf.predict_proba(p.design)[:, 1]
    got = m.raw(p.hashed, p.dense_z)
    assert len(m.trees) == 200
    assert np.max(np.abs(got - want)) <= 1e-12


def test_gbm_threshold_compares_float32_values():
    tree = Tree(
        feature=np.array([HASH_DIM, -1, -1]),
        threshold=np.array([0.1, 0.0, 0.0]),
        left=np.array([1, -1, -1]),
        right=np.array([2, -1, -1]),
        value=np.array([0.0, -1.0, 1.0]),
    )
    m = GBMModel(init=0.0, learning_rate=1.0, trees=(tree,))
    hashed = sparse.csr_matrix((1, HASH_DIM))
    got = m.raw(hashed, np.array([[0.1]]))
    assert got.tolist() == [float(expit(1.0))]


def test_gbm_leaves_and_global_features():
    p = make_problem()
    m = fit_gbm(p.design, p.y, p.vocab)
    vocab = set(p.vocab.tolist())
    for t in m.trees:
        leaf = t.left == -1
        assert np.array_equal(leaf, t.right == -1)
        assert np.all(t.feature[leaf] == -1)
        assert np.all(t.threshold[leaf] == 0.0)
        for f in t.feature[~leaf].tolist():
            assert f in vocab or HASH_DIM <= f < HASH_DIM + 8


def test_gbm_json_round_trip_exact():
    p = make_problem()
    m = fit_gbm(p.design, p.y, p.vocab)
    doc = m.to_json()
    assert doc["feature_dtype"] == "float32"
    assert set(doc["trees"][0]) == {"feature", "threshold", "left", "right", "value"}
    again = GBMModel.from_json(json.loads(json.dumps(doc)))
    assert np.array_equal(again.raw(p.hashed, p.dense_z), m.raw(p.hashed, p.dense_z))


def test_fit_gbm_is_deterministic():
    p = make_problem()
    assert fit_gbm(p.design, p.y, p.vocab).to_json() == fit_gbm(p.design, p.y, p.vocab).to_json()


def test_fit_gbm_rejects_design_width_that_does_not_match_the_vocabulary():
    p = make_problem()
    with pytest.raises(ValueError, match="design matrix"):
        fit_gbm(p.design, p.y, p.vocab[:-1])
    with pytest.raises(ValueError, match="design matrix"):
        fit_gbm(p.design[:, :-1], p.y, p.vocab)


def test_fit_gbm_rejects_single_class_labels():
    p = make_problem()
    with pytest.raises(ValueError, match="single class"):
        fit_gbm(p.design, np.zeros(len(p.y), dtype=np.int64), p.vocab)
