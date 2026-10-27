from itertools import pairwise

import numpy as np
import pytest

from routeforge_train.certify import (
    DELTA,
    LAMBDA_GRID,
    PROFILES,
    SENTINEL_LAMBDA,
    ProfileCert,
    certify_profiles,
    clopper_pearson_upper,
    ltt_lambda,
    regret_counts,
)


def test_lambda_grid():
    assert LAMBDA_GRID[0] == 1.0 and LAMBDA_GRID[-1] == 0.0
    assert len(LAMBDA_GRID) == 101
    assert LAMBDA_GRID[7] == 0.93
    assert all(a > b for a, b in pairwise(LAMBDA_GRID))


def test_constants():
    assert SENTINEL_LAMBDA == 1.01 and DELTA == 0.05
    assert PROFILES == (("strict", 0.02), ("balanced", 0.05), ("economy", 0.10))


def test_ltt_sentinel_below_minimum_n():
    assert ltt_lambda(148, [0] * 101, 0.02) == (1.01, 0)
    assert ltt_lambda(149, [0] * 5 + [149] * 96, 0.02) == (0.96, 0)


def test_ltt_rejects_while_binomial_tail_below_delta():
    assert ltt_lambda(200, [0] * 5 + [12, 13] + [0] * 94, 0.10) == (0.95, 12)


def test_ltt_stops_at_first_non_rejection():
    assert ltt_lambda(200, [0, 13] + [0] * 99, 0.10) == (1.0, 0)


def test_ltt_rejects_wrong_count_length():
    with pytest.raises(ValueError):
        ltt_lambda(200, [0] * 100, 0.10)


def test_clopper_pearson_upper_known():
    assert clopper_pearson_upper(0, 100) == pytest.approx(0.029513049607039932, abs=1e-12)
    assert clopper_pearson_upper(0, 148) == pytest.approx(0.02003795168910416, abs=1e-12)
    assert clopper_pearson_upper(5, 5) == 1.0


def _synthetic(n, seed, m=3):
    rng = np.random.default_rng(seed)
    lcb = rng.random((n, m)) * 0.95
    exp_cost = rng.random((n, m)) + 0.1
    success = rng.random((n, m)) < 0.7
    return lcb, exp_cost, success


def test_certify_profiles_strict_sentinel_on_small_n():
    lcb, exp_cost, success = _synthetic(88, 11)
    certs = certify_profiles(lcb, exp_cost, success, 0)
    assert [c.name for c in certs] == ["strict", "balanced", "economy"]
    strict, balanced, economy = certs
    assert strict.lambda_ == 1.01 and strict.sentinel
    assert strict.cert_regret == 0 and strict.cert_n == 88
    assert strict.regret_ucb == clopper_pearson_upper(0, 88)
    for cert in (balanced, economy):
        assert cert.lambda_ in LAMBDA_GRID and not cert.sentinel
        assert cert.cert_n == 88 and cert.delta == 0.05
    assert set(strict.to_json()) == {
        "name",
        "alpha",
        "delta",
        "lambda",
        "cert_n",
        "cert_regret",
        "regret_ucb",
    }
    assert [c.alpha for c in certs] == [0.02, 0.05, 0.10]


def test_certify_profiles_json_round_trip():
    lcb, exp_cost, success = _synthetic(300, 4)
    for cert in certify_profiles(lcb, exp_cost, success, 1):
        assert ProfileCert.from_json(cert.to_json()) == cert


def test_certify_profiles_uses_counts_at_certified_lambda():
    lcb, exp_cost, success = _synthetic(400, 9)
    counts = regret_counts(lcb, exp_cost, success, 0)
    for cert in certify_profiles(lcb, exp_cost, success, 0):
        if cert.sentinel:
            assert cert.cert_regret == 0
        else:
            assert cert.cert_regret == counts[LAMBDA_GRID.index(cert.lambda_)]


def test_regret_counts_zero_at_lambda_one():
    for seed in range(5):
        lcb, exp_cost, success = _synthetic(120, seed)
        counts = regret_counts(lcb, exp_cost, success, seed % 3)
        assert len(counts) == 101
        assert counts[0] == 0


def test_regret_counts_known_example():
    lcb = np.array([[0.9, 0.2], [0.9, 0.2]])
    exp_cost = np.array([[2.0, 1.0], [2.0, 1.0]])
    success = np.array([[True, False], [True, True]])
    counts = regret_counts(lcb, exp_cost, success, 0, grid=(1.0, 0.5, 0.1))
    # lam 1.0 and 0.5 escalate to model 0; lam 0.1 picks cheap model 1, which fails on row 0.
    assert counts == [0, 0, 1]
