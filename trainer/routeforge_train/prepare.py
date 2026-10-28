"""Coverage filtering, label checks, stratified splits and the data manifest."""

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from routeforge_train.features import prompt_sha256
from routeforge_train.jsonio import atomic_writer, dumps_line, file_info, write_json
from routeforge_train.labels import LabelRule, label_rule
from routeforge_train.outcomes import OutcomeRow, read_outcomes
from routeforge_train.paths import Paths
from routeforge_train.prices import PriceTable, list_cost_usd
from routeforge_train.provenance import utc_iso
from routeforge_train.splits import (
    SEED,
    SPLIT_NAMES,
    SPLIT_PERCENT,
    assign_splits,
    write_splits,
)

SCHEMA_VERSION = 1
_DROP_REASONS = ("missing_model", "duplicate_row", "prompt_mismatch")


@dataclass(frozen=True)
class CoverageReport:
    kept: list[OutcomeRow]
    datasets: dict[str, dict]
    dropped_datasets: dict[str, list[str]]
    non_pool_rows: int
    dropped_prompts: list[dict[str, str]]


@dataclass(frozen=True)
class PrepareResult:
    manifest: dict
    test_ids: frozenset[str]
    rules: dict[str, LabelRule]


def apply_coverage(rows: Iterable[OutcomeRow], pool: Sequence[str]) -> CoverageReport:
    """Keep only prompts that every pool model answered once, with one identical prompt text."""
    pool_set = set(pool)
    non_pool = 0
    pool_rows: list[OutcomeRow] = []
    dataset_of: dict[str, str] = {}
    prompts: dict[str, dict[str, list[OutcomeRow]]] = {}
    for row in rows:
        if row.model not in pool_set:
            non_pool += 1
            continue
        if dataset_of.setdefault(row.prompt_id, row.dataset) != row.dataset:
            raise ValueError(f"{row.prompt_id}: appears under more than one dataset")
        pool_rows.append(row)
        prompts.setdefault(row.dataset, {}).setdefault(row.prompt_id, []).append(row)

    datasets: dict[str, dict] = {}
    dropped_datasets: dict[str, list[str]] = {}
    kept_ids: set[str] = set()
    dropped_prompts: list[dict[str, str]] = []
    for dataset, by_prompt in prompts.items():
        present = {r.model for group in by_prompt.values() for r in group}
        absent = [f"missing model {m}" for m in pool if m not in present]
        if absent:
            dropped_datasets[dataset] = absent
            continue
        dropped = dict.fromkeys(_DROP_REASONS, 0)
        kept = 0
        for prompt_id, group in by_prompt.items():
            reason = _drop_reason(group, pool)
            if reason is None:
                kept_ids.add(prompt_id)
                kept += 1
            else:
                dropped[reason] += 1
                dropped_prompts.append(
                    {"dataset": dataset, "prompt_id": prompt_id, "reason": reason}
                )
        datasets[dataset] = {
            "family": next(iter(by_prompt.values()))[0].family,
            "prompts_in": len(by_prompt),
            "prompts_kept": kept,
            "dropped": dropped,
        }
    return CoverageReport(
        kept=[r for r in pool_rows if r.prompt_id in kept_ids],
        datasets=datasets,
        dropped_datasets=dropped_datasets,
        non_pool_rows=non_pool,
        dropped_prompts=sorted(dropped_prompts, key=lambda d: (d["dataset"], d["prompt_id"])),
    )


def _drop_reason(group: list[OutcomeRow], pool: Sequence[str]) -> str | None:
    per_model = Counter(r.model for r in group)
    if any(per_model[m] == 0 for m in pool):
        return "missing_model"
    if any(per_model[m] > 1 for m in pool):
        return "duplicate_row"
    if len({prompt_sha256(r.prompt) for r in group}) > 1:
        return "prompt_mismatch"
    return None


def prepare(
    paths: Paths,
    *,
    pool: Sequence[str],
    source: Mapping[str, object],
    conversion: Mapping[str, object],
    prices: PriceTable,
    git_sha: str,
    now: datetime,
) -> PrepareResult:
    """Filter `outcomes.jsonl`, split it, write the splits, replay prompts and the manifest."""
    pool = tuple(pool)
    rows = list(read_outcomes(paths.outcomes))
    report = apply_coverage(rows, pool)

    rules = _label_rules(rows, pool, report)
    for row in report.kept:
        if row.success != rules[row.dataset].is_success(row.score):
            raise ValueError(
                f"{row.prompt_id} ({row.model}): success={row.success} contradicts "
                f"{rules[row.dataset].describe()} for score {row.score}"
            )

    dataset_of = {r.prompt_id: r.dataset for r in report.kept}
    assignment = assign_splits(dataset_of)
    infos = write_splits(report.kept, assignment, paths.splits_dir, paths.root)
    test_ids = frozenset(p for p, name in assignment.items() if name == "test")
    _write_replay_prompts(report.kept, test_ids, paths)

    split_counts = {dataset: dict.fromkeys(SPLIT_NAMES, 0) for dataset in report.datasets}
    for prompt_id, name in assignment.items():
        split_counts[dataset_of[prompt_id]][name] += 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_iso(now),
        "git_sha": git_sha,
        "source": dict(source),
        "pool": list(pool),
        "datasets": {
            dataset: {
                **stats,
                "score_values": list(rules[dataset].score_values),
                "label_rule": rules[dataset].describe(),
                "split_counts": split_counts[dataset],
            }
            for dataset, stats in report.datasets.items()
        },
        "dropped_datasets": report.dropped_datasets,
        "dropped_prompts": report.dropped_prompts,
        "non_pool_rows": report.non_pool_rows,
        "conversion": {
            "records": conversion["records"],
            "usage_clamped": conversion["usage_clamped"],
        },
        "splits": {
            "seed": SEED,
            "percent": dict(SPLIT_PERCENT),
            "counts": {
                name: sum(counts[name] for counts in split_counts.values()) for name in SPLIT_NAMES
            },
        },
        "files": {
            "outcomes": file_info(paths.outcomes, paths.root, len(rows)).to_json(),
            **{name: infos[name].to_json() for name in SPLIT_NAMES},
        },
        "max_completion_tokens": _max_completion_tokens(report.kept, pool),
        "costs": _costs(rows, report, prices, pool),
    }
    write_json(paths.manifest, manifest)
    return PrepareResult(manifest, test_ids, rules)


def _label_rules(
    rows: Sequence[OutcomeRow], pool: Sequence[str], report: CoverageReport
) -> dict[str, LabelRule]:
    """Rules from every pool row of each kept dataset, as the converter derived them."""
    pool_set = set(pool)
    scores: dict[str, list[float]] = {dataset: [] for dataset in report.datasets}
    for row in rows:
        if row.model in pool_set and row.dataset in scores:
            scores[row.dataset].append(row.score)
    return {dataset: label_rule(values) for dataset, values in scores.items()}


def _write_replay_prompts(
    kept: Sequence[OutcomeRow], test_ids: frozenset[str], paths: Paths
) -> None:
    first: dict[str, OutcomeRow] = {}
    for row in kept:
        if row.prompt_id in test_ids:
            first.setdefault(row.prompt_id, row)
    with atomic_writer(paths.replay_dir / "prompts.jsonl") as f:
        for prompt_id in sorted(first):
            row = first[prompt_id]
            f.write(
                dumps_line(
                    {
                        "prompt_id": prompt_id,
                        "dataset": row.dataset,
                        "family": row.family,
                        "prompt_sha256": prompt_sha256(row.prompt),
                        "prompt": row.prompt,
                    }
                )
            )


def _max_completion_tokens(kept: Sequence[OutcomeRow], pool: Sequence[str]) -> dict[str, Any]:
    by_model = dict.fromkeys(pool, 0)
    for row in kept:
        by_model[row.model] = max(by_model[row.model], row.completion_tokens)
    return {"overall": max(by_model.values(), default=0), "by_model": by_model}


def _costs(
    rows: Sequence[OutcomeRow], report: CoverageReport, prices: PriceTable, pool: Sequence[str]
) -> dict[str, Any]:
    """Recorded versus list cost, over every pool row and over the kept rows."""
    pool_set = set(pool)
    kept_ids = {r.prompt_id for r in report.kept}
    recorded_all: dict[str, list[float]] = {m: [] for m in pool}
    listed_all: dict[str, list[float]] = {m: [] for m in pool}
    recorded_kept: dict[str, list[float]] = {m: [] for m in pool}
    listed_kept: dict[str, list[float]] = {m: [] for m in pool}
    cells: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for row in rows:
        if row.model not in pool_set:
            continue
        listed = list_cost_usd(prices.price(row.model), row.prompt_tokens, row.completion_tokens)
        recorded_all[row.model].append(row.cost_usd)
        listed_all[row.model].append(listed)
        if row.prompt_id in kept_ids:
            recorded_kept[row.model].append(row.cost_usd)
            listed_kept[row.model].append(listed)
            cell = cells.setdefault((row.dataset, row.model), ([], []))
            cell[0].append(row.cost_usd)
            cell[1].append(listed)

    by_dataset_model = []
    for dataset in report.datasets:
        for model in pool:
            if (dataset, model) not in cells:
                continue
            recorded = math.fsum(cells[(dataset, model)][0])
            listed = math.fsum(cells[(dataset, model)][1])
            by_dataset_model.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "recorded_usd": recorded,
                    "list_usd": listed,
                    "ratio": recorded / listed if listed > 0 else None,
                }
            )
    return {
        "by_model": {
            model: {
                "recorded_usd_all_rows": math.fsum(recorded_all[model]),
                "list_usd_all_rows": math.fsum(listed_all[model]),
                "recorded_usd_kept": math.fsum(recorded_kept[model]),
                "list_usd_kept": math.fsum(listed_kept[model]),
            }
            for model in pool
        },
        "by_dataset_model": by_dataset_model,
    }
