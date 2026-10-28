import dataclasses

import numpy as np
import pytest

from routeforge_train import pipeline
from routeforge_train.metrics import (
    ECE_BINS,
    StrategyOutcome,
    auroc,
    brier,
    ece,
    reliability,
    summarize,
)


def outcome(success, cost, regret=None, **kwargs):
    success = np.asarray(success, dtype=np.float64)
    regret = np.zeros_like(success) if regret is None else np.asarray(regret, dtype=np.float64)
    return StrategyOutcome(
        name=kwargs.pop("name", "s"),
        success=success,
        cost=np.asarray(cost, dtype=np.float64),
        regret=regret,
        shares=kwargs.pop("shares", {}),
        **kwargs,
    )


def test_ece_bins_constant():
    assert ECE_BINS == 15


def test_auroc_known():
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) == 0.75


def test_auroc_single_class_is_none():
    assert auroc(np.array([1, 1]), np.array([0.2, 0.3])) is None
    assert auroc(np.array([0, 0, 0]), np.array([0.2, 0.3, 0.1])) is None


def test_auroc_returns_a_python_float():
    got = auroc(np.array([False, True]), np.array([0.2, 0.9]))
    assert got == 1.0 and type(got) is float


def test_brier_known():
    assert brier(np.array([1, 0]), np.array([0.8, 0.4])) == pytest.approx((0.04 + 0.16) / 2)


def test_brier_accepts_bool_labels():
    assert brier(np.array([True, False]), np.array([1.0, 0.0])) == 0.0


def _ece_fixture():
    p = np.arange(30) / 100
    y = np.arange(30) >= 15
    return y, p


def test_reliability_equal_mass_bins_known():
    y, p = _ece_fixture()
    bins = reliability(y, p)
    assert len(bins) == ECE_BINS == 15
    assert [b["n"] for b in bins] == [2] * 15
    assert set(bins[0]) == {"n", "mean_p", "frac_success"}
    assert bins[0]["mean_p"] == pytest.approx(0.005) and bins[0]["frac_success"] == 0.0
    assert bins[7]["mean_p"] == pytest.approx(0.145) and bins[7]["frac_success"] == 0.5
    assert bins[14]["mean_p"] == pytest.approx(0.285) and bins[14]["frac_success"] == 1.0


def test_reliability_orders_by_score_not_input_position():
    y, p = _ece_fixture()
    order = np.random.default_rng(3).permutation(30)
    assert reliability(y[order], p[order]) == reliability(y, p)


def test_reliability_skips_empty_bins():
    bins = reliability(np.array([1, 0, 1]), np.array([0.9, 0.1, 0.5]))
    assert [b["n"] for b in bins] == [1, 1, 1]
    assert [b["mean_p"] for b in bins] == [0.1, 0.5, 0.9]


def test_reliability_uneven_split_keeps_every_row():
    y = np.arange(32) % 2 == 0
    bins = reliability(y, np.linspace(0, 1, 32))
    assert sum(b["n"] for b in bins) == 32 and len(bins) == 15


def test_reliability_custom_bin_count():
    y, p = _ece_fixture()
    assert [b["n"] for b in reliability(y, p, bins=3)] == [10, 10, 10]


def test_ece_equal_mass_bins_known():
    y, p = _ece_fixture()
    gaps = []
    for i in range(0, 30, 2):
        frac = y[i : i + 2].mean()
        gaps.append(abs(frac - p[i : i + 2].mean()))
    expected = sum(2 / 30 * gap for gap in gaps)
    assert ece(y, p) == pytest.approx(expected)
    assert ece(y, p) == pytest.approx(float(np.mean(gaps)))


def test_ece_of_a_perfectly_calibrated_constant_score_is_zero():
    y = np.array([1, 0, 1, 0])
    assert ece(y, np.full(4, 0.5), bins=2) == pytest.approx(0.0)


def test_ece_and_reliability_reject_no_rows():
    with pytest.raises(ValueError, match="no rows"):
        ece(np.array([], dtype=bool), np.array([]))
    with pytest.raises(ValueError, match="no rows"):
        reliability(np.array([], dtype=bool), np.array([]))


def test_summarize_fields():
    o = outcome(
        [1, 0, 1, 1],
        [0.5, 0.25, 0.25, 1.0],
        regret=[0, 1, 0, 0],
        shares={"a": 0.75, "b": 0.25},
        escalated=np.array([True, False, False, True]),
        chosen=np.array([0, 1, 1, 0]),
    )
    got = summarize(o)
    assert got == {
        "n": 4,
        "accuracy": 0.75,
        "successes": 3.0,
        "total_cost_usd": 2.0,
        "cost_per_success_usd": pytest.approx(2.0 / 3),
        "regret_count": 1.0,
        "regret_rate": 0.25,
        "escalation_rate": 0.5,
        "shares": {"a": 0.75, "b": 0.25},
    }


def test_summarize_zero_success_has_no_cost_per_success():
    got = summarize(outcome([0, 0], [1.0, 2.0], shares={"a": 1.0}))
    assert got["successes"] == 0.0 and got["accuracy"] == 0.0
    assert got["cost_per_success_usd"] is None
    assert got["total_cost_usd"] == 3.0
    assert got["escalation_rate"] is None
    assert got["shares"] == {"a": 1.0}


def test_summarize_handles_fractional_share_matched_outcomes():
    got = summarize(outcome([0.5, 0.25], [1.0, 1.0], regret=[0.25, 0.0]))
    assert got["successes"] == 0.75 and got["accuracy"] == 0.375
    assert got["regret_count"] == 0.25 and got["regret_rate"] == 0.125
    assert got["cost_per_success_usd"] == pytest.approx(2.0 / 0.75)


def test_summarize_copies_shares():
    shares = {"a": 1.0}
    got = summarize(outcome([1.0], [1.0], shares=shares))
    got["shares"]["a"] = 0.0
    assert shares == {"a": 1.0}


def test_summarize_rejects_no_rows_and_ragged_arrays():
    with pytest.raises(ValueError, match="no rows"):
        summarize(outcome([], []))
    with pytest.raises(ValueError, match="same length"):
        summarize(outcome([1, 0], [1.0], regret=[0, 0]))


def test_strategy_outcome_is_frozen_and_defaults_to_no_extras():
    o = outcome([1.0], [1.0])
    assert o.escalated is None and o.chosen is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.name = "other"  # type: ignore[misc]


def test_routing_metrics_is_summarize_of_the_routed_columns(monkeypatch):
    seen = []
    real = pipeline.summarize

    def spy(o):
        seen.append(o)
        return real(o)

    monkeypatch.setattr(pipeline, "summarize", spy)
    success = np.array([[True, False], [False, True], [True, True]])
    cost = np.array([[1.0, 0.5], [2.0, 0.25], [3.0, 0.75]])
    got = pipeline.routing_metrics(np.array([1, 0, 1]), success, cost, strongest=0)
    assert len(seen) == 1
    assert seen[0].success.tolist() == [0.0, 0.0, 1.0] and seen[0].cost.tolist() == [0.5, 2.0, 0.75]
    assert seen[0].regret.tolist() == [1.0, 0.0, 0.0]
    assert got == {
        "accuracy": pytest.approx(1 / 3),
        "total_cost_usd": 3.25,
        "cost_per_success_usd": 3.25,
        "regret_rate": pytest.approx(1 / 3),
    }
