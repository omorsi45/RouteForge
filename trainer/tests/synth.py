"""Synthetic outcome tables and project trees for the trainer tests. No real data."""

import math
import string
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from routeforge_train.outcomes import OutcomeRow, write_outcomes
from routeforge_train.paths import Paths
from routeforge_train.prepare import PrepareResult, prepare
from routeforge_train.prices import Price, list_cost_usd, load_prices

SYNTH_POOL = ("m-strong", "m-mid", "m-cheap")
SYNTH_DATASETS = {
    "alpha": "reasoning",
    "beta": "code",
    "gamma": "qa",
    "delta": "instruction",
}
SYNTH_PRICES = {
    "m-strong": Price(5.0, 20.0),
    "m-mid": Price(1.0, 4.0),
    "m-cheap": Price(0.1, 0.4),
}
SYNTH_NOW = datetime(2026, 10, 26, 12, 0, 0, tzinfo=UTC)
FIXED_NOW = datetime(2026, 10, 26, tzinfo=UTC)
SYNTH_SOURCE = {
    "dataset": "synthetic",
    "revision": "synthetic",
    "url": "synthetic",
    "archive_bytes": 0,
    "archive_sha256": "synthetic",
    "members": 0,
}

_PRICES_YAML = """\
effective_date: "2026-10-26"
source: "synthetic test prices"
currency: USD
models:
  - name: m-strong
    input_per_mtok: 5.0
    output_per_mtok: 20.0
  - name: m-mid
    input_per_mtok: 1.0
    output_per_mtok: 4.0
  - name: m-cheap
    input_per_mtok: 0.1
    output_per_mtok: 0.4
"""

# Ranks follow the simulated-latency rule: output price ascending, r = 0 for the cheapest.
_LATENCY_YAML = """\
# SIMULATED. Synthetic latency profiles for benchmarks; not measurements of any provider.
simulated: true
seed: 20261026
jitter: 0.1
models:
  - {name: m-cheap, rank: 0, ttft_ms: 200, decode_tokens_per_s: 120}
  - {name: m-mid, rank: 1, ttft_ms: 300, decode_tokens_per_s: 105}
  - {name: m-strong, rank: 2, ttft_ms: 400, decode_tokens_per_s: 90}
"""

_MID_RECORDED_FACTOR = 0.8


def _vocabulary(rng: np.random.Generator) -> list[str]:
    letters = np.array(list(string.ascii_lowercase))
    words: set[str] = set()
    while len(words) < 200:
        words.add("".join(rng.choice(letters, size=int(rng.integers(3, 9)))))
    return sorted(words)


def _success_probability(model: str, family: str, easy: bool) -> float:
    if model == "m-strong":
        return 0.9
    if model == "m-mid":
        return 0.85 if easy and family in ("reasoning", "code") else 0.35
    return 0.95 if easy and family in ("qa", "instruction") else 0.2


def synthetic_rows(n_per_dataset: int = 150, seed: int = 11) -> list[OutcomeRow]:
    """Rows ordered by dataset, then pool model, then prompt index (converter order)."""
    rng = np.random.default_rng(seed)
    vocabulary = _vocabulary(rng)
    rows: list[OutcomeRow] = []
    for dataset, family in SYNTH_DATASETS.items():
        prompts = []
        for i in range(n_per_dataset):
            easy = bool(rng.random() < 0.5)
            filler = " ".join(rng.choice(vocabulary, size=int(rng.integers(6, 31))))
            prompts.append((i, easy, f"{dataset} task {i} {'easy' if easy else 'hard'} {filler}"))
        for model in SYNTH_POOL:
            for i, easy, prompt in prompts:
                success = bool(rng.random() < _success_probability(model, family, easy))
                prompt_tokens = math.ceil(len(prompt.encode("utf-8")) / 4)
                completion_tokens = int(rng.integers(20, 400))
                cost = list_cost_usd(SYNTH_PRICES[model], prompt_tokens, completion_tokens)
                if model == "m-mid":
                    cost *= _MID_RECORDED_FACTOR
                rows.append(
                    OutcomeRow(
                        prompt_id=f"{dataset}:{i}",
                        dataset=dataset,
                        family=family,
                        prompt=prompt,
                        model=model,
                        score=1.0 if success else 0.0,
                        success=success,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cost_usd=cost,
                    )
                )
    return rows


def conversion_for(rows: list[OutcomeRow]) -> dict[str, list]:
    """The `conversion` section the converter would report for `rows`."""
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        counts[(row.dataset, row.model)] = counts.get((row.dataset, row.model), 0) + 1
    records = [
        {"dataset": dataset, "model": model, "records": n} for (dataset, model), n in counts.items()
    ]
    return {"records": records, "usage_clamped": []}


def write_synthetic_inputs(root: Path, rows: list[OutcomeRow]) -> Paths:
    """Write prices, latency profiles and `outcomes.jsonl` for `rows` under `root`."""
    paths = Paths(root)
    paths.prices.parent.mkdir(parents=True, exist_ok=True)
    paths.prices.write_text(_PRICES_YAML, encoding="utf-8")
    paths.latency_profiles.parent.mkdir(parents=True, exist_ok=True)
    paths.latency_profiles.write_text(_LATENCY_YAML, encoding="utf-8")
    write_outcomes(rows, paths.outcomes, root)
    return paths


def prepare_synthetic(paths: Paths, rows: list[OutcomeRow]) -> PrepareResult:
    return prepare(
        paths,
        pool=SYNTH_POOL,
        source=SYNTH_SOURCE,
        conversion=conversion_for(rows),
        prices=load_prices(paths.prices),
        git_sha="synthetic",
        now=SYNTH_NOW,
    )


def build_synthetic_project(root: Path, *, n_per_dataset: int = 150, seed: int = 11) -> Paths:
    rows = synthetic_rows(n_per_dataset, seed)
    paths = write_synthetic_inputs(root, rows)
    prepare_synthetic(paths, rows)
    return paths
