import math
from pathlib import Path

import pytest

from routeforge_train.errors import TrainerError
from routeforge_train.latency import (
    LatencyProfile,
    LatencyProfiles,
    expected_latency_profiles,
    load_latency_profiles,
    predicted_p95_ms,
    route_latency,
)
from routeforge_train.llmrouterbench import POOL
from routeforge_train.prices import load_prices

REPO_ROOT = Path(__file__).resolve().parents[2]

VALID = """\
simulated: true
seed: 7
jitter: 0.1
models:
  - {name: a, rank: 0, ttft_ms: 200, decode_tokens_per_s: 120}
  - {name: b, rank: 1, ttft_ms: 300, decode_tokens_per_s: 105}
"""


def write(tmp_path, text):
    path = tmp_path / "latency.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_committed_latency_profiles_follow_rule():
    lp = load_latency_profiles(REPO_ROOT / "bench/latency_profiles.yaml")
    assert lp.seed == 20261026 and lp.jitter == 0.1
    prices = load_prices(REPO_ROOT / "deploy/prices.yaml")
    assert sorted(lp.models.values(), key=lambda p: p.rank) == expected_latency_profiles(
        prices, POOL
    )
    assert lp.models["gemini-2.5-pro"].rank == 3
    assert lp.models["gpt-5"].rank == 4


def test_expected_latency_profiles_values():
    prices = load_prices(REPO_ROOT / "deploy/prices.yaml")
    got = expected_latency_profiles(prices, POOL)
    assert [p.name for p in got] == [
        "qwen3-235b-a22b-2507",
        "deepseek-v3.1-terminus",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gpt-5",
        "claude-sonnet-4",
    ]
    assert [(p.rank, p.ttft_ms, p.decode_tokens_per_s) for p in got][5] == (5, 700.0, 45.0)


def test_predicted_p95_ms():
    profile = LatencyProfile("m", 0, 200.0, 120.0)
    want = (200 + 1000) * (1 + 1.645 * 0.1 / math.sqrt(3))
    assert predicted_p95_ms(profile, 120.0, 0.1) == pytest.approx(want, rel=1e-12)


def test_route_latency_picks_lowest_and_ties_by_pool_order():
    profiles = LatencyProfiles(
        1,
        0.1,
        {
            "a": LatencyProfile("a", 1, 300.0, 100.0),
            "b": LatencyProfile("b", 0, 200.0, 120.0),
            "c": LatencyProfile("c", 0, 200.0, 120.0),
        },
    )
    eot = {"a": 50.0, "b": 50.0, "c": 50.0}
    assert route_latency(["a", "b", "c"], profiles, eot) == 1
    assert route_latency(["c", "b", "a"], profiles, eot) == 0


def test_load_latency_profiles_valid(tmp_path):
    lp = load_latency_profiles(write(tmp_path, VALID))
    assert lp == LatencyProfiles(
        7,
        0.1,
        {"a": LatencyProfile("a", 0, 200.0, 120.0), "b": LatencyProfile("b", 1, 300.0, 105.0)},
    )


def test_load_latency_profiles_requires_simulated_flag(tmp_path):
    with pytest.raises(TrainerError, match="simulated"):
        load_latency_profiles(write(tmp_path, VALID.replace("simulated: true", "simulated: false")))
    with pytest.raises(TrainerError, match="simulated"):
        load_latency_profiles(write(tmp_path, VALID.replace("simulated: true\n", "")))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.replace("seed: 7", "seed: x"),
        lambda s: s.replace("jitter: 0.1", "jitter: -0.1"),
        lambda s: s.replace("name: b", "name: a"),
        lambda s: s.replace("decode_tokens_per_s: 105", "decode_tokens_per_s: 0"),
        lambda s: s.replace("rank: 1", "rank: 0"),
        lambda s: s[: s.index("models:")] + "models: []\n",
        lambda s: "- just a list\n",
        lambda s: "models: [unclosed",
    ],
)
def test_load_latency_profiles_rejects_bad_files(tmp_path, mutation):
    with pytest.raises(TrainerError):
        load_latency_profiles(write(tmp_path, mutation(VALID)))


def test_load_latency_profiles_missing_file(tmp_path):
    with pytest.raises(TrainerError, match="nope.yaml"):
        load_latency_profiles(tmp_path / "nope.yaml")
