import numpy as np
import pytest
import scipy.stats

from routeforge_train.metrics import StrategyOutcome
from routeforge_train.stats import (
    N_RESAMPLES,
    SEED,
    TOST_MARGIN,
    bootstrap_p_value,
    bootstrap_totals,
    clopper_pearson_interval,
    compare,
    holm,
    mcnemar_exact,
    percentile_ci,
    tost,
)


def outcome(success, cost, name="s"):
    success = np.asarray(success, dtype=np.float64)
    return StrategyOutcome(
        name=name,
        success=success,
        cost=np.asarray(cost, dtype=np.float64),
        regret=np.zeros_like(success),
        shares={},
    )


def test_constants():
    assert N_RESAMPLES == 10_000 and SEED == 20261026 and TOST_MARGIN == 0.02


def test_mcnemar_exact_known():
    a = np.array([1] * 10 + [0] * 2 + [1] * 5, bool)
    b = np.array([0] * 10 + [1] * 2 + [1] * 5, bool)
    assert mcnemar_exact(a, b) == {"b": 10, "c": 2, "p_value": pytest.approx(0.03857421875)}
    assert mcnemar_exact(a, a)["p_value"] == 1.0


def test_mcnemar_exact_is_symmetric_and_returns_python_ints():
    a = np.array([1, 1, 1, 1, 0, 0], bool)
    b = np.array([0, 0, 0, 0, 1, 1], bool)
    forward, backward = mcnemar_exact(a, b), mcnemar_exact(b, a)
    assert (forward["b"], forward["c"]) == (4, 2) and (backward["b"], backward["c"]) == (2, 4)
    assert forward["p_value"] == backward["p_value"] == pytest.approx(0.6875)
    assert type(forward["b"]) is int and type(forward["c"]) is int


def test_mcnemar_exact_caps_at_one():
    a = np.array([1, 0], bool)
    b = np.array([0, 1], bool)
    assert mcnemar_exact(a, b)["p_value"] == 1.0


def test_mcnemar_exact_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        mcnemar_exact(np.array([True]), np.array([True, False]))


def test_holm_known():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one_keeps_order_and_handles_edges():
    assert holm([]) == []
    assert holm([0.2]) == [0.2]
    assert holm([0.6, 0.5]) == [1.0, 1.0]
    got = holm([0.04, 0.001, 0.02, 0.2])
    assert got == pytest.approx([0.08, 0.004, 0.06, 0.2])
    assert all(adjusted >= raw for adjusted, raw in zip(got, [0.04, 0.001, 0.02, 0.2], strict=True))


def test_holm_ties_are_adjusted_alike():
    assert holm([0.01, 0.01, 0.5]) == pytest.approx([0.03, 0.03, 0.5])


def test_holm_rejects_values_outside_unit_interval():
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="p-value"):
            holm([0.5, bad])


def test_bootstrap_preserves_stratum_sizes():
    strata = np.array(["a"] * 5 + ["b"] * 3 + ["c"] * 7)
    values = np.column_stack([strata == s for s in ("a", "b", "c")]).astype(np.float64)
    totals = bootstrap_totals(strata, values, n_resamples=300, seed=1, chunk=64)
    assert totals.shape == (300, 3)
    assert (totals == np.array([5.0, 3.0, 7.0])).all()


def test_bootstrap_chunking_does_not_change_the_shape_or_stratum_sizes():
    strata = np.array(["a"] * 4 + ["b"] * 6)
    values = np.column_stack([strata == "a", strata == "b"]).astype(np.float64)
    for chunk in (1, 7, 500):
        totals = bootstrap_totals(strata, values, n_resamples=50, seed=1, chunk=chunk)
        assert totals.shape == (50, 2) and (totals == np.array([4.0, 6.0])).all()


def test_bootstrap_is_deterministic_and_seed_sensitive():
    rng = np.random.default_rng(0)
    strata = np.array(["a"] * 20 + ["b"] * 20)
    values = rng.random((40, 2))
    first = bootstrap_totals(strata, values, n_resamples=200, seed=5)
    assert np.array_equal(first, bootstrap_totals(strata, values, n_resamples=200, seed=5))
    assert not np.array_equal(first, bootstrap_totals(strata, values, n_resamples=200, seed=6))


def test_bootstrap_default_resamples_and_mean():
    strata = np.array(["a"] * 30)
    values = np.arange(30, dtype=np.float64).reshape(30, 1)
    totals = bootstrap_totals(strata, values)
    assert totals.shape == (N_RESAMPLES, 1)
    assert totals.mean() == pytest.approx(values.sum(), rel=0.02)


def test_bootstrap_rejects_bad_shapes():
    strata = np.array(["a", "b"])
    with pytest.raises(ValueError, match="2-D"):
        bootstrap_totals(strata, np.ones(2), n_resamples=10)
    with pytest.raises(ValueError, match="rows"):
        bootstrap_totals(strata, np.ones((3, 1)), n_resamples=10)
    with pytest.raises(ValueError, match="no rows"):
        bootstrap_totals(np.array([], dtype=str), np.ones((0, 1)), n_resamples=10)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_totals(strata, np.ones((2, 1)), n_resamples=0)
    with pytest.raises(ValueError, match="chunk"):
        bootstrap_totals(strata, np.ones((2, 1)), n_resamples=10, chunk=0)


def test_percentile_ci_known_and_nan_aware():
    samples = np.arange(101, dtype=np.float64)
    assert percentile_ci(samples) == pytest.approx((2.5, 97.5))
    assert percentile_ci(samples, level=0.5) == pytest.approx((25.0, 75.0))
    with_nan = np.append(samples, [np.nan, np.nan])
    assert percentile_ci(with_nan) == pytest.approx((2.5, 97.5))


def test_percentile_ci_all_nan_is_nan_pair():
    low, high = percentile_ci(np.full(5, np.nan))
    assert np.isnan(low) and np.isnan(high)


def test_bootstrap_p_value_bounds():
    assert bootstrap_p_value(np.full(999, 0.1)) == pytest.approx(2 / 1000)
    assert bootstrap_p_value(np.full(999, -0.1)) == pytest.approx(2 / 1000)
    assert bootstrap_p_value(np.linspace(-1.0, 1.0, 999)) == 1.0
    assert bootstrap_p_value(np.zeros(10)) == 1.0


def test_bootstrap_p_value_excludes_nan_and_rejects_empty():
    clean = np.full(99, 0.1)
    assert bootstrap_p_value(np.append(clean, [np.nan] * 50)) == bootstrap_p_value(clean)
    with pytest.raises(ValueError, match="finite"):
        bootstrap_p_value(np.full(3, np.nan))


def _strata_and_values(n_per=30, seed=0):
    rng = np.random.default_rng(seed)
    strata = np.array(["a"] * n_per + ["b"] * n_per)
    return strata, rng


def test_compare_identical_strategies_has_zero_diff():
    strata, rng = _strata_and_values()
    success = (rng.random(60) < 0.6).astype(np.float64)
    cost = rng.random(60) + 0.1
    totals = bootstrap_totals(strata, np.column_stack([success, cost]), n_resamples=400, seed=2)
    o = outcome(success, cost)
    got = compare(o, o, totals, totals)
    for key in ("accuracy_diff", "cost_per_success_diff"):
        assert got[key]["estimate"] == 0.0
        assert got[key]["ci_low"] == got[key]["ci_high"] == 0.0
        assert got[key]["p_value"] == 1.0
    reduction = got["relative_cps_reduction"]
    assert reduction["estimate"] == 0.0
    assert reduction["ci_low"] == reduction["ci_high"] == 0.0
    assert got["degenerate_resamples"] == 0
    assert set(got) == {
        "accuracy_diff",
        "cost_per_success_diff",
        "relative_cps_reduction",
        "degenerate_resamples",
    }
    assert set(got["accuracy_diff"]) == {"estimate", "ci_low", "ci_high", "p_value"}
    assert set(got["relative_cps_reduction"]) == {"estimate", "ci_low", "ci_high"}


def test_compare_detects_half_cost():
    strata, rng = _strata_and_values(seed=1)
    success = (rng.random(60) < 0.7).astype(np.float64)
    cost = rng.random(60) + 0.1
    values = np.column_stack([success, cost / 2, success, cost])
    totals = bootstrap_totals(strata, values, n_resamples=500, seed=3)
    a, b = outcome(success, cost / 2), outcome(success, cost)
    got = compare(a, b, totals[:, 0:2], totals[:, 2:4])
    assert got["accuracy_diff"]["estimate"] == 0.0
    assert got["relative_cps_reduction"]["estimate"] == pytest.approx(0.5)
    assert got["relative_cps_reduction"]["ci_low"] > 0
    assert got["relative_cps_reduction"]["ci_low"] == pytest.approx(0.5)
    assert got["cost_per_success_diff"]["estimate"] < 0
    assert got["cost_per_success_diff"]["ci_high"] < 0
    assert got["cost_per_success_diff"]["p_value"] < 0.05


def test_compare_accuracy_diff_matches_resampled_totals():
    strata, rng = _strata_and_values(seed=2)
    sa = (rng.random(60) < 0.8).astype(np.float64)
    sb = (rng.random(60) < 0.4).astype(np.float64)
    cost = np.ones(60)
    values = np.column_stack([sa, cost, sb, cost])
    totals = bootstrap_totals(strata, values, n_resamples=300, seed=4)
    got = compare(outcome(sa, cost), outcome(sb, cost), totals[:, 0:2], totals[:, 2:4])
    assert got["accuracy_diff"]["estimate"] == pytest.approx(sa.mean() - sb.mean())
    samples = (totals[:, 0] - totals[:, 2]) / 60
    assert got["accuracy_diff"]["ci_low"] == pytest.approx(np.percentile(samples, 2.5))
    assert got["accuracy_diff"]["ci_high"] == pytest.approx(np.percentile(samples, 97.5))
    assert got["accuracy_diff"]["ci_low"] > 0 and got["accuracy_diff"]["p_value"] < 0.05


def test_compare_counts_degenerate_resamples_and_excludes_them():
    # Column 0 is success, column 1 cost; two resamples have zero successes for a.
    totals_a = np.array([[0.0, 5.0], [10.0, 5.0], [0.0, 4.0], [5.0, 5.0], [8.0, 4.0]])
    totals_b = np.array([[10.0, 10.0]] * 5)
    a, b = outcome([1, 1, 0, 0], [1, 1, 1, 1]), outcome([1, 1, 1, 1], [1, 1, 1, 1])
    got = compare(a, b, totals_a, totals_b)
    assert got["degenerate_resamples"] == 2
    # Valid resamples give cost-per-success differences -0.5, 0.0 and -0.5 (b is always 1.0).
    cps = got["cost_per_success_diff"]
    assert cps["estimate"] == 1.0
    assert cps["ci_low"] == -0.5 and cps["ci_high"] == pytest.approx(-0.025)
    reduction = got["relative_cps_reduction"]
    assert reduction["ci_low"] == pytest.approx(0.025)
    assert reduction["ci_high"] == pytest.approx(0.5)


def test_compare_without_successes_has_undefined_cost_per_success():
    strata, _ = _strata_and_values()
    zero, some = np.zeros(60), np.ones(60)
    cost = np.ones(60)
    values = np.column_stack([zero, cost, some, cost])
    totals = bootstrap_totals(strata, values, n_resamples=50, seed=1)
    got = compare(outcome(zero, cost), outcome(some, cost), totals[:, 0:2], totals[:, 2:4])
    assert got["degenerate_resamples"] == 50
    assert got["cost_per_success_diff"] == {
        "estimate": None,
        "ci_low": None,
        "ci_high": None,
        "p_value": None,
    }
    assert got["relative_cps_reduction"] == {"estimate": None, "ci_low": None, "ci_high": None}
    assert got["accuracy_diff"]["estimate"] == -1.0


def test_compare_rejects_mismatched_inputs():
    a, b = outcome([1, 0], [1, 1]), outcome([1, 0, 1], [1, 1, 1])
    totals = np.ones((4, 2))
    with pytest.raises(ValueError, match="same length"):
        compare(a, b, totals, totals)
    with pytest.raises(ValueError, match="shape"):
        compare(a, a, np.ones((4, 3)), np.ones((4, 3)))
    with pytest.raises(ValueError, match="shape"):
        compare(a, a, np.ones((4, 2)), np.ones((5, 2)))


def test_tost_equivalent_when_tight_near_zero():
    samples = np.random.default_rng(1).normal(0.0, 0.002, 2000)
    got = tost(samples, 0.0)
    assert got["equivalent"] and got["non_inferior"]
    assert got["margin"] == TOST_MARGIN and got["estimate"] == 0.0
    assert got["p_lower"] == got["p_upper"] == 1 / 2001
    assert got["p_value"] == max(got["p_lower"], got["p_upper"])
    low, high = got["ci90"]
    assert low == pytest.approx(np.percentile(samples, 5)) and high == pytest.approx(
        np.percentile(samples, 95)
    )


def test_tost_not_equivalent_beyond_margin():
    rng = np.random.default_rng(2)
    below = tost(rng.normal(-0.05, 0.002, 2000), -0.05)
    assert not below["non_inferior"] and not below["equivalent"]
    assert below["p_lower"] == 1.0
    above = tost(rng.normal(0.05, 0.002, 2000), 0.05)
    assert above["non_inferior"] and not above["equivalent"]
    assert above["p_upper"] == 1.0 and above["p_value"] == 1.0


def test_tost_margin_is_a_parameter_and_boundaries_count():
    samples = np.array([-0.02, 0.0, 0.0, 0.02])
    got = tost(samples, 0.0, margin=0.02)
    assert got["p_lower"] == 2 / 5 and got["p_upper"] == 2 / 5
    assert tost(samples, 0.0, margin=0.5)["p_lower"] == 1 / 5


def test_tost_rejects_no_samples():
    with pytest.raises(ValueError, match="no samples"):
        tost(np.array([]), 0.0)


def test_clopper_pearson_interval_known():
    lo, hi = clopper_pearson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.30849710781876083, abs=1e-12)
    assert clopper_pearson_interval(10, 10)[1] == 1.0
    lo, hi = clopper_pearson_interval(10, 10)
    assert lo == pytest.approx(1 - 0.30849710781876083, abs=1e-12)


def test_clopper_pearson_interval_two_sided_against_beta_quantiles():
    lo, hi = clopper_pearson_interval(7, 40)
    assert lo == pytest.approx(scipy.stats.beta.ppf(0.025, 7, 34), abs=1e-12)
    assert hi == pytest.approx(scipy.stats.beta.ppf(0.975, 8, 33), abs=1e-12)
    lo90, hi90 = clopper_pearson_interval(7, 40, level=0.90)
    assert lo < lo90 < 7 / 40 < hi90 < hi


def test_clopper_pearson_interval_rejects_bad_counts():
    for k, n in ((-1, 5), (6, 5), (0, 0)):
        with pytest.raises(ValueError, match="k"):
            clopper_pearson_interval(k, n)
