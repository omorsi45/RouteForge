"""Simulated latency profiles (spec 8.6): synthetic, not measurements of any provider."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from routeforge_train.errors import TrainerError
from routeforge_train.prices import PriceTable

P95_Z = 1.645


@dataclass(frozen=True)
class LatencyProfile:
    name: str
    rank: int
    ttft_ms: float
    decode_tokens_per_s: float


@dataclass(frozen=True)
class LatencyProfiles:
    seed: int
    jitter: float
    models: dict[str, LatencyProfile]


def expected_latency_profiles(prices: PriceTable, pool: Sequence[str]) -> list[LatencyProfile]:
    """Rank by output price ascending (ties: input price, then name); ordered by rank."""
    ranked = sorted(
        pool,
        key=lambda m: (prices.price(m).output_per_mtok, prices.price(m).input_per_mtok, m),
    )
    return [LatencyProfile(m, r, 200.0 + 100 * r, 120.0 - 15 * r) for r, m in enumerate(ranked)]


def predicted_p95_ms(
    profile: LatencyProfile, expected_output_tokens: float, jitter: float
) -> float:
    """(TTFT + decode time) scaled to the p95 of uniform +/-jitter noise (sd = jitter / sqrt 3)."""
    base = profile.ttft_ms + 1000 * expected_output_tokens / profile.decode_tokens_per_s
    return base * (1 + P95_Z * jitter / math.sqrt(3))


def route_latency(
    pool: Sequence[str],
    profiles: LatencyProfiles,
    expected_output_tokens: Mapping[str, float],
) -> int:
    """Index of the lowest predicted p95 latency; ties go to the earlier pool entry."""
    predicted = [
        predicted_p95_ms(profiles.models[m], expected_output_tokens[m], profiles.jitter)
        for m in pool
    ]
    return min(range(len(pool)), key=lambda i: (predicted[i], i))


def load_latency_profiles(path: Path) -> LatencyProfiles:
    """Read and validate a profile file; it must be marked `simulated: true`."""
    path = Path(path)
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise TrainerError(f"{path}: cannot read latency profiles: {exc}") from exc

    def fail(problem: str) -> TrainerError:
        return TrainerError(f"{path}: {problem}")

    if not isinstance(doc, dict):
        raise fail("expected a mapping with simulated, seed, jitter and models")
    if doc.get("simulated") is not True:
        raise fail("simulated must be true: these profiles are synthetic")
    seed = doc.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise fail("seed must be an integer")
    jitter = doc.get("jitter")
    if isinstance(jitter, bool) or not isinstance(jitter, int | float):
        raise fail("jitter must be a number")
    if not 0 <= jitter < 1:
        raise fail("jitter must be in [0, 1)")
    entries = doc.get("models")
    if not isinstance(entries, list) or not entries:
        raise fail("models must be a non-empty list")
    models: dict[str, LatencyProfile] = {}
    ranks: set[int] = set()
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise fail(f"models[{position}] must be a mapping")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise fail(f"models[{position}].name must be a non-empty string")
        if name in models:
            raise fail(f"duplicate model {name!r}")
        rank = entry.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise fail(f"{name}.rank must be a non-negative integer")
        if rank in ranks:
            raise fail(f"{name}.rank {rank} is already used")
        ranks.add(rank)
        models[name] = LatencyProfile(
            name,
            rank,
            _positive(entry, "ttft_ms", name, fail),
            _positive(entry, "decode_tokens_per_s", name, fail),
        )
    return LatencyProfiles(seed, float(jitter), models)


def _positive(entry: dict[str, Any], key: str, name: str, fail: Any) -> float:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise fail(f"{name}.{key} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise fail(f"{name}.{key} must be finite and > 0")
    return float(value)
