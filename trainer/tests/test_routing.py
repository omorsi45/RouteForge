import numpy as np
import pytest

from routeforge_train.llmrouterbench import POOL
from routeforge_train.prices import Price, PriceTable
from routeforge_train.routing import (
    REASON_ESCALATED,
    REASON_PASSING,
    estimated_prompt_tokens,
    expected_cost_usd,
    regret_mask,
    route_cheapest,
    route_quality,
    route_static,
    strongest_index,
)
from routeforge_train.rules import KeywordRules


def test_reason_constants():
    assert REASON_PASSING == "cheapest_passing"
    assert REASON_ESCALATED == "escalated_no_model_passed"


def test_estimated_prompt_tokens_is_ceil_bytes_over_four():
    got = estimated_prompt_tokens(np.array([0, 1, 4, 5, 8, 9]))
    assert got.tolist() == [0, 1, 1, 2, 2, 3]


def test_expected_cost_usd_formula():
    prices = PriceTable("2026-10-26", "s", {"gpt-5": Price(1.25, 10.0), "cheap": Price(0.1, 0.5)})
    eot = {"gpt-5": 100.0, "cheap": 40.0}
    got = expected_cost_usd(np.array([5, 0]), ["gpt-5", "cheap"], prices, eot)
    assert got.shape == (2, 2)
    assert got[0, 0] == pytest.approx((2 * 1.25 + 100 * 10.0) / 1e6)
    assert got[0, 1] == pytest.approx((2 * 0.1 + 40 * 0.5) / 1e6)
    assert got[1, 0] == pytest.approx(100 * 10.0 / 1e6)


def test_route_quality_picks_cheapest_passing_else_strongest():
    cost = np.array([[3.0, 1.0, 2.0]])
    lcb = np.array([[0.9, 0.5, 0.8]])
    chosen, escalated = route_quality(lcb, cost, 0, 0.7)
    assert chosen.tolist() == [2] and escalated.tolist() == [False]
    chosen, escalated = route_quality(lcb, cost, 0, 0.95)
    assert chosen.tolist() == [0] and escalated.tolist() == [True]
    chosen, escalated = route_quality(lcb, cost, 0, 1.0)
    assert chosen.tolist() == [0] and escalated.tolist() == [True]


def test_route_quality_threshold_is_inclusive():
    chosen, escalated = route_quality(np.array([[0.5, 0.8]]), np.array([[1.0, 2.0]]), 1, 0.5)
    assert chosen.tolist() == [0] and escalated.tolist() == [False]


def test_route_quality_escalates_to_given_strongest():
    cost = np.array([[3.0, 1.0, 2.0]])
    chosen, escalated = route_quality(np.array([[0.1, 0.1, 0.1]]), cost, 2, 0.5)
    assert chosen.tolist() == [2] and escalated.tolist() == [True]


def test_route_quality_cost_ties_break_by_pool_order():
    cost = np.array([[1.0, 1.0, 5.0]])
    chosen, _ = route_quality(np.array([[0.9, 0.9, 0.9]]), cost, 2, 0.5)
    assert chosen.tolist() == [0]
    chosen, escalated = route_quality(np.array([[0.1, 0.9, 0.9]]), cost, 2, 0.5)
    assert chosen.tolist() == [1] and escalated.tolist() == [False]


def test_route_quality_handles_rows_independently():
    cost = np.array([[3.0, 1.0], [1.0, 3.0]])
    lcb = np.array([[0.9, 0.9], [0.2, 0.9]])
    chosen, escalated = route_quality(lcb, cost, 0, 0.5)
    assert chosen.tolist() == [1, 1] and escalated.tolist() == [False, False]


def test_regret_mask():
    success = np.array([[True, False], [False, False], [True, True], [False, True]])
    chosen = np.array([1, 1, 0, 0])
    assert regret_mask(chosen, success, 0).tolist() == [True, False, False, False]
    assert regret_mask(np.array([1, 1, 1, 1]), success, 1).tolist() == [False] * 4


def test_strongest_index_by_accuracy():
    success = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    cost = np.ones((3, 3))
    assert strongest_index(success, cost) == 1


def test_strongest_index_ties_by_cost_then_order():
    success = np.array([[1, 1, 1], [0, 0, 0]], dtype=bool)
    cost = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    assert strongest_index(success, cost) == 2
    assert strongest_index(success, np.ones((2, 3))) == 0


def test_route_cheapest_ties_by_pool_order():
    cost = np.array([[2.0, 1.0, 1.0], [1.0, 1.0, 1.0], [3.0, 2.0, 0.5]])
    assert route_cheapest(cost).tolist() == [1, 0, 2]


def test_route_static_classified_and_oracle_family():
    pool = ["a", "b", "c"]
    rules = KeywordRules(
        family_keywords={"code": ("python",), "qa": ("capital",)},
        family_model={"code": "c", "qa": "a"},
        family_order=("code", "qa"),
        default_family="qa",
    )
    texts = ["write python", "capital of x", "zzz"]
    assert route_static(texts, rules, pool, classify=True).tolist() == [2, 0, 0]
    assert route_static(["qa", "code"], rules, pool, classify=False).tolist() == [0, 2]


def test_committed_pool_prices_cost_tie_for_first_and_third():
    # gpt-5 and gemini-2.5-pro share list prices, so their expected costs tie.
    prices = PriceTable(
        "2026-10-26",
        "s",
        {"gpt-5": Price(1.25, 10.0), "gemini-2.5-pro": Price(1.25, 10.0)},
    )
    eot = {"gpt-5": 300.0, "gemini-2.5-pro": 300.0}
    cost = expected_cost_usd(np.array([400]), ["gpt-5", "gemini-2.5-pro"], prices, eot)
    assert cost[0, 0] == cost[0, 1]
    assert POOL[0] == "gpt-5" and POOL[2] == "gemini-2.5-pro"
