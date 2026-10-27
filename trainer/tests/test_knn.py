import numpy as np
import pytest

from routeforge_train.features import featurize
from routeforge_train.knn import K, fit_knn


def _cluster_texts(cluster: int, n: int) -> list[str]:
    return [f"cl{cluster} wd{cluster} tk{cluster} it{j % 3}" for j in range(n)]


def test_k_is_25():
    assert K == 25


def test_knn_raw_is_neighbor_success_rate():
    texts = _cluster_texts(0, 30) + _cluster_texts(1, 30) + _cluster_texts(2, 30)
    success = np.zeros((90, 2), dtype=bool)
    success[:30, 0] = True
    success[:, 1] = True
    model = fit_knn(featurize(texts).hashed, success, k=25)
    query = featurize(["cl0 wd0 tk0", "cl1 wd1 tk1"]).hashed
    raw = model.raw(query)
    assert raw.shape == (2, 2)
    assert raw[0].tolist() == [1.0, 1.0]
    assert raw[1].tolist() == [0.0, 1.0]


def test_knn_default_k_and_fractional_rate():
    texts = _cluster_texts(0, 40)
    success = np.zeros((40, 1), dtype=bool)
    success[:10, 0] = True
    model = fit_knn(featurize(texts).hashed, success)
    assert model.k == 25
    raw = model.raw(featurize(["cl0 wd0 tk0 it0"]).hashed)
    assert 0.0 <= raw[0, 0] <= 1.0


def test_knn_rejects_k_larger_than_train():
    texts = _cluster_texts(0, 10)
    with pytest.raises(ValueError, match="k"):
        fit_knn(featurize(texts).hashed, np.ones((10, 1), dtype=bool), k=25)
