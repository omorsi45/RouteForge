"""The `evaluate` command: the only reader of the held-out test split.

Nothing is fitted on the test split. `train` supplies the served and offline candidates; the
test prompts are scored once, every strategy is replayed on the recorded outcomes, and the
paired statistics, calibration metrics, leave-one-dataset-out runs and replay decisions follow.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from routeforge_train.calibration import Calibration, fit_platt
from routeforge_train.certify import LAMBDA_GRID, PROFILES, ProfileCert
from routeforge_train.errors import TrainerError
from routeforge_train.features import Features, featurize, prompt_sha256
from routeforge_train.jsonio import atomic_writer, dumps_line, write_json
from routeforge_train.knn import fit_knn
from routeforge_train.latency import load_latency_profiles, route_latency
from routeforge_train.metrics import StrategyOutcome, auroc, brier, ece, reliability, summarize
from routeforge_train.outcomes import OutcomeRow
from routeforge_train.paths import Paths
from routeforge_train.pipeline import KINDS, certify, fit_pipeline, request_texts, simulate
from routeforge_train.plots import reliability_png
from routeforge_train.policy import Policy, load_policy, score_policy
from routeforge_train.prices import PriceTable, list_cost_usd, load_prices
from routeforge_train.provenance import current_git_sha, utc_iso
from routeforge_train.report_md import (
    COST_NOTE,
    EXTRACTION_NOTE,
    EXTRACTION_PRESENT_NOTE,
    GUARANTEE_NOTE,
    LATENCY_NOTE,
    TEMPLATE_NOTE,
    render_evaluation_markdown,
)
from routeforge_train.routing import (
    REASON_ESCALATED,
    REASON_PASSING,
    expected_cost_usd,
    regret_mask,
    route_cheapest,
    route_quality,
    route_static,
)
from routeforge_train.splits import read_test_split
from routeforge_train.stats import (
    N_RESAMPLES,
    SEED,
    bootstrap_totals,
    clopper_pearson_interval,
    compare,
    holm,
    mcnemar_exact,
    tost,
)
from routeforge_train.table import OutcomeTable, pivot
from routeforge_train.train import load_table, read_manifest

SCHEMA_VERSION = 1
H1_PROFILE = "balanced"
H1_TARGET = 0.25
BASELINES = (
    "always_strongest",
    "always_cheapest",
    "static_keyword",
    "static_oracle_family",
    "latency",
)
PROFILE_NAMES = tuple(name for name, _ in PROFILES)
_ALPHA = dict(PROFILES)
_SWEEP_KEYS = (
    "accuracy",
    "total_cost_usd",
    "cost_per_success_usd",
    "regret_rate",
    "escalation_rate",
)


def oracle_choice(success: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Per prompt, the cheapest successful model by recorded cost, else the cheapest model.

    Ties go to the earlier pool entry.
    """
    cheapest_success = np.argmin(np.where(success, cost, np.inf), axis=1)
    cheapest = np.argmin(cost, axis=1)
    return np.where(success.any(axis=1), cheapest_success, cheapest).astype(np.int64)


def share_matched_random(
    chosen: np.ndarray,
    success: np.ndarray,
    cost: np.ndarray,
    strongest: int,
    pool: Sequence[str],
    *,
    n_seeds: int = 1_000,
    seed: int = SEED,
) -> StrategyOutcome:
    """Random routing with the same model counts as `chosen`, averaged over `n_seeds` shuffles.

    Success, cost and regret per prompt are the mean over the seeds, so they are fractional.
    """
    n = len(chosen)
    if n == 0:
        raise ValueError("cannot match the shares of no rows")
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be at least 1, got {n_seeds}")
    counts = np.bincount(chosen, minlength=len(pool))
    if len(counts) != len(pool):
        raise ValueError("chosen holds a model index outside the pool")
    labels = np.repeat(np.arange(len(pool)), counts)
    rows = np.arange(n)
    got_success, got_cost, got_regret = np.zeros(n), np.zeros(n), np.zeros(n)
    for s in range(n_seeds):
        assign = np.random.default_rng(seed + s).permutation(labels)
        got_success += success[rows, assign]
        got_cost += cost[rows, assign]
        got_regret += regret_mask(assign, success, strongest)
    return StrategyOutcome(
        name="share_matched_random",
        success=got_success / n_seeds,
        cost=got_cost / n_seeds,
        regret=got_regret / n_seeds,
        shares={model: float(count) / n for model, count in zip(pool, counts, strict=True)},
    )


def _beats(entry: Mapping[str, Any]) -> bool:
    """The cost-per-success difference interval lies entirely below zero."""
    high = entry["cost_per_success_diff"]["ci_high"]
    return high is not None and high < 0


def h1_verdict(report: Mapping[str, Any]) -> dict:
    """H1 for the balanced profile and the served predictor.

    H1 is falsified unless the cost per success beats both share-matched random and the oracle
    family rule with an interval entirely below zero, the realized regret interval does not
    lie above alpha, and the pooled leave-one-dataset-out reduction stays above zero. Whether
    the 25% target is met is reported separately and does not decide falsification.
    """
    stats = report["statistics"][H1_PROFILE]
    baselines = stats["baselines"]
    reduction = baselines["always_strongest"]["relative_cps_reduction"]
    matched = baselines[f"share_matched_random/{H1_PROFILE}"]
    oracle = baselines["static_oracle_family"]
    regret = stats["regret"]
    pooled = report["lodo"]["pooled"]
    alpha = _ALPHA[H1_PROFILE]

    estimate = reduction["estimate"]
    beats_random, beats_oracle = _beats(matched), _beats(oracle)
    exceeds = regret["ci_low"] > alpha
    lodo_low = pooled["ci_low"]
    gains_disappear = lodo_low is None or lodo_low <= 0
    falsified_by = [
        name
        for name, failed in (
            ("share_matched_random", not beats_random),
            ("static_oracle_family", not beats_oracle),
            ("regret_exceeds_alpha", exceeds),
            ("lodo_gains_disappear", gains_disappear),
        )
        if failed
    ]
    interval = ("estimate", "ci_low", "ci_high")
    return {
        "profile": H1_PROFILE,
        "predictor": report["served_predictor"],
        "cps_reduction_vs_strongest": {key: reduction[key] for key in interval},
        "target": H1_TARGET,
        "target_met": estimate is not None and estimate >= H1_TARGET,
        "regret": {
            **{key: regret[key] for key in ("rate", "count", "n", "ci_low", "ci_high")},
            "alpha": alpha,
        },
        "regret_exceeds_alpha_beyond_ci": exceeds,
        "vs_share_matched_random": {
            "cost_per_success_diff": {
                key: matched["cost_per_success_diff"][key] for key in interval
            },
            "beats": beats_random,
        },
        "vs_static_oracle_family": {
            "cost_per_success_diff": {
                key: oracle["cost_per_success_diff"][key] for key in interval
            },
            "beats": beats_oracle,
        },
        "lodo_pooled_reduction": {key: pooled[key] for key in interval},
        "lodo_gains_disappear": gains_disappear,
        "falsified": bool(falsified_by),
        "falsified_by": falsified_by,
    }


def extraction_status(paths: Paths) -> dict[str, str]:
    """Whether extraction labels exist; this command never evaluates them."""
    if (paths.extraction_dir / "outcomes.jsonl").is_file():
        return {"status": "labels_present_not_evaluated", "note": EXTRACTION_PRESENT_NOTE}
    return {"status": "not_run", "note": EXTRACTION_NOTE}


def _shares(chosen: np.ndarray, pool: Sequence[str]) -> dict[str, float]:
    counts = np.bincount(chosen, minlength=len(pool))
    return {model: float(count) / len(chosen) for model, count in zip(pool, counts, strict=True)}


def _outcome(
    name: str,
    chosen: np.ndarray,
    table: OutcomeTable,
    strongest: int,
    escalated: np.ndarray | None = None,
) -> StrategyOutcome:
    rows = np.arange(len(table))
    return StrategyOutcome(
        name=name,
        success=table.success[rows, chosen].astype(np.float64),
        cost=table.cost[rows, chosen],
        regret=regret_mask(chosen, table.success, strongest).astype(np.float64),
        shares=_shares(chosen, table.pool),
        escalated=escalated,
        chosen=chosen,
    )


def _subset(o: StrategyOutcome, mask: np.ndarray) -> StrategyOutcome:
    """The rows of one dataset. Strategies without one chosen model keep their overall shares."""
    return StrategyOutcome(
        name=o.name,
        success=o.success[mask],
        cost=o.cost[mask],
        regret=o.regret[mask],
        shares=o.shares if o.chosen is None else _shares(o.chosen[mask], tuple(o.shares)),
        escalated=None if o.escalated is None else o.escalated[mask],
        chosen=None if o.chosen is None else o.chosen[mask],
    )


@dataclass(frozen=True, eq=False)
class KindScores:
    """One predictor type scored on the calibration and test prompts."""

    kind: str
    lambdas: dict[str, float]
    calibrations: tuple[Calibration, ...]
    cal_raw: np.ndarray
    test_raw: np.ndarray
    test_p: np.ndarray
    test_lcb: np.ndarray


def _lambdas(path: Any, profiles: Sequence[ProfileCert]) -> dict[str, float]:
    lambdas = {profile.name: profile.lambda_ for profile in profiles}
    if sorted(lambdas) != sorted(PROFILE_NAMES):
        raise TrainerError(f"{path}: profiles {sorted(lambdas)} are not {sorted(PROFILE_NAMES)}")
    return lambdas


def _check_pool(label: Any, names: Sequence[str], pool: tuple[str, ...]) -> None:
    if tuple(names) != pool:
        raise TrainerError(f"{label}: pool {tuple(names)} differs from the data pool {pool}")


def _policy_scores(
    paths: Paths,
    kind: str,
    policy: Policy,
    pool: tuple[str, ...],
    cal_features: Features,
    test_features: Features,
) -> KindScores:
    path = paths.candidates_dir / f"{kind}.json"
    candidate = load_policy(path)
    if kind == policy.predictor_type and candidate.version != policy.version:
        raise TrainerError(
            f"{path}: version {candidate.version} differs from the served policy "
            f"{policy.version}; rerun train"
        )
    if candidate.predictor_type != kind:
        raise TrainerError(f"{path}: holds a {candidate.predictor_type} predictor, not {kind}")
    _check_pool(path, [entry.name for entry in candidate.models], pool)
    test = score_policy(candidate, test_features)
    return KindScores(
        kind=kind,
        lambdas=_lambdas(path, candidate.profiles),
        calibrations=tuple(entry.calibration for entry in candidate.models),
        cal_raw=score_policy(candidate, cal_features).raw,
        test_raw=test.raw,
        test_p=test.p,
        test_lcb=test.lcb,
    )


def _knn_scores(
    paths: Paths,
    pool: tuple[str, ...],
    train: OutcomeTable,
    train_features: Features,
    cal_features: Features,
    test_features: Features,
) -> KindScores:
    path = paths.candidates_dir / "knn.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        k = int(doc["k"])
        models = [entry["model"] for entry in doc["calibrations"]]
        calibrations = tuple(Calibration.from_json(entry) for entry in doc["calibrations"])
        profiles = tuple(ProfileCert.from_json(entry) for entry in doc["profiles"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TrainerError(
            f"{path}: cannot read the kNN candidate (run train first): {exc}"
        ) from exc
    _check_pool(path, models, pool)
    knn = fit_knn(train_features.hashed, train.success, k)
    test_raw = knn.raw(test_features.hashed)
    return KindScores(
        kind="knn",
        lambdas=_lambdas(path, profiles),
        calibrations=calibrations,
        cal_raw=knn.raw(cal_features.hashed),
        test_raw=test_raw,
        test_p=np.column_stack([c.p_of(test_raw[:, j]) for j, c in enumerate(calibrations)]),
        test_lcb=np.column_stack([c.lcb_of(test_raw[:, j]) for j, c in enumerate(calibrations)]),
    )


def _strategies(
    test: OutcomeTable,
    policy: Policy,
    scores: Mapping[str, KindScores],
    exp_cost: np.ndarray,
    latency_index: int,
    n_random_seeds: int,
) -> tuple[
    dict[str, StrategyOutcome], dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]
]:
    """Every strategy on the test split, the served quality choices and the simple choices."""
    pool, n = test.pool, len(test)
    strongest = pool.index(policy.strongest_model)
    rules = policy.static_rules
    unknown = sorted(set(test.families.tolist()) - set(rules.family_model))
    if unknown:
        raise TrainerError(f"test families {unknown} have no static rule in the policy")
    simple = {
        "strongest": np.full(n, strongest, dtype=np.int64),
        "cheapest": route_cheapest(exp_cost).astype(np.int64),
        "static": route_static(request_texts(test), rules, pool, classify=True),
        "latency": np.full(n, latency_index, dtype=np.int64),
    }
    outcomes: dict[str, StrategyOutcome] = {}

    def add(name: str, chosen: np.ndarray, escalated: np.ndarray | None = None) -> None:
        outcomes[name] = _outcome(name, chosen, test, strongest, escalated)

    add("always_strongest", simple["strongest"])
    add("always_cheapest", simple["cheapest"])
    add("static_keyword", simple["static"])
    add("static_oracle_family", route_static(test.families.tolist(), rules, pool, classify=False))
    add("latency", simple["latency"])
    served: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind in KINDS:
        for profile in PROFILE_NAMES:
            lam = scores[kind].lambdas[profile]
            chosen, escalated = route_quality(scores[kind].test_lcb, exp_cost, strongest, lam)
            add(f"quality/{kind}/{profile}", chosen, escalated)
            if kind == policy.predictor_type:
                served[profile] = (chosen, escalated)
    for profile in PROFILE_NAMES:
        name = f"share_matched_random/{profile}"
        matched = share_matched_random(
            served[profile][0], test.success, test.cost, strongest, pool, n_seeds=n_random_seeds
        )
        outcomes[name] = replace(matched, name=name)
    add("oracle", oracle_choice(test.success, test.cost))
    return outcomes, served, simple


def _summaries(
    outcomes: Mapping[str, StrategyOutcome],
    scores: Mapping[str, KindScores],
    test: OutcomeTable,
) -> tuple[dict[str, dict], dict[str, dict[str, dict]]]:
    strategies: dict[str, dict] = {}
    for name, o in outcomes.items():
        strategies[name] = summarize(o)
        kind_profile = name.split("/")
        if kind_profile[0] == "quality":
            strategies[name]["lambda"] = scores[kind_profile[1]].lambdas[kind_profile[2]]
    per_dataset = {
        dataset: {
            name: summarize(_subset(o, test.datasets == dataset)) for name, o in outcomes.items()
        }
        for dataset in sorted(set(test.datasets.tolist()))
    }
    return strategies, per_dataset


def _lambda_sweep(
    lcb: np.ndarray, exp_cost: np.ndarray, test: OutcomeTable, strongest: int
) -> list[dict]:
    sweep = []
    for lam in LAMBDA_GRID:
        chosen, escalated = route_quality(lcb, exp_cost, strongest, lam)
        summary = summarize(_outcome("sweep", chosen, test, strongest, escalated))
        sweep.append({"lambda": lam, **{key: summary[key] for key in _SWEEP_KEYS}})
    return sweep


def _calibration_report(
    scores: Mapping[str, KindScores], test: OutcomeTable, calibration: OutcomeTable
) -> dict[str, dict[str, dict]]:
    families = sorted(set(test.families.tolist()))
    report: dict[str, dict[str, dict]] = {}
    for kind in KINDS:
        s = scores[kind]
        report[kind] = {}
        for j, model in enumerate(test.pool):
            y, raw, p_iso = test.success[:, j], s.test_raw[:, j], s.test_p[:, j]
            p_platt = fit_platt(s.cal_raw[:, j], calibration.success[:, j]).predict(raw)
            report[kind][model] = {
                "auroc_raw": auroc(y, raw),
                "auroc_by_family": {
                    family: auroc(y[test.families == family], raw[test.families == family])
                    for family in families
                },
                "brier_isotonic": brier(y, p_iso),
                "brier_platt": brier(y, p_platt),
                "ece_isotonic": ece(y, p_iso),
                "ece_platt": ece(y, p_platt),
                "reliability": {
                    "isotonic": reliability(y, p_iso),
                    "platt": reliability(y, p_platt),
                },
            }
    return report


def _attach_holm(comparisons: Mapping[str, dict], key: str) -> None:
    """Holm-adjust the `key` p-values over the baselines; undefined p-values stay undefined."""
    defined = [name for name, c in comparisons.items() if c[key]["p_value"] is not None]
    adjusted = holm([comparisons[name][key]["p_value"] for name in defined])
    for c in comparisons.values():
        c[key]["p_holm"] = None
    for name, p in zip(defined, adjusted, strict=True):
        comparisons[name][key]["p_holm"] = p


def _statistics(
    outcomes: Mapping[str, StrategyOutcome],
    test: OutcomeTable,
    served: str,
    n_resamples: int,
    n_random_seeds: int,
) -> dict[str, dict]:
    """Paired comparisons of the served quality strategy, one bootstrap for every strategy."""
    n = len(test)
    names = list(outcomes)
    values = np.column_stack(
        [part for name in names for part in (outcomes[name].success, outcomes[name].cost)]
    )
    totals = bootstrap_totals(test.datasets, values, n_resamples=n_resamples)
    columns = {name: totals[:, 2 * i : 2 * i + 2] for i, name in enumerate(names)}
    strongest = outcomes["always_strongest"]
    report: dict[str, dict] = {}
    for profile, alpha in PROFILES:
        quality_name = f"quality/{served}/{profile}"
        quality = outcomes[quality_name]
        comparisons = {
            baseline: compare(quality, outcomes[baseline], columns[quality_name], columns[baseline])
            for baseline in (*BASELINES, f"share_matched_random/{profile}")
        }
        _attach_holm(comparisons, "cost_per_success_diff")
        _attach_holm(comparisons, "accuracy_diff")
        accuracy_diff = (columns[quality_name][:, 0] - columns["always_strongest"][:, 0]) / n
        summary = summarize(quality)
        count = round(summary["regret_count"])
        low, high = clopper_pearson_interval(count, n)
        report[profile] = {
            "quality": quality_name,
            "n": n,
            "n_resamples": n_resamples,
            "n_random_seeds": n_random_seeds,
            "baselines": comparisons,
            "mcnemar_vs_always_strongest": mcnemar_exact(
                quality.success.astype(bool), strongest.success.astype(bool)
            ),
            "tost_vs_always_strongest": tost(
                accuracy_diff, summary["accuracy"] - summarize(strongest)["accuracy"]
            ),
            "regret": {
                "alpha": alpha,
                "rate": count / n,
                "count": count,
                "n": n,
                "ci_low": low,
                "ci_high": high,
            },
        }
    return report


def _main_lr_c(paths: Paths, policy: Policy) -> dict[str, float] | None:
    """The C values of the main logistic regression fit, needed only when lr is served."""
    if policy.predictor_type != "lr":
        return None
    path = paths.reports / "training.json"
    try:
        training = json.loads(path.read_text(encoding="utf-8"))
        version = training["policy_version"]
        c_by_model = {model: float(c) for model, c in training["lr"]["c_by_model"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise TrainerError(f"{path}: cannot read the logistic regression C values: {exc}") from exc
    if version != policy.version:
        raise TrainerError(f"{path}: describes policy {version}, not {policy.version}; rerun train")
    return c_by_model


def _without(table: OutcomeTable, dataset: str) -> OutcomeTable:
    return table.subset(table.datasets != dataset)


def _concat(name: str, parts: Sequence[StrategyOutcome]) -> StrategyOutcome:
    return StrategyOutcome(
        name=name,
        success=np.concatenate([o.success for o in parts]),
        cost=np.concatenate([o.cost for o in parts]),
        regret=np.concatenate([o.regret for o in parts]),
        shares={},
    )


def _lodo(
    *,
    served: str,
    lr_c: Mapping[str, float] | None,
    train: OutcomeTable,
    calibration: OutcomeTable,
    certification: OutcomeTable,
    test: OutcomeTable,
    prices: PriceTable,
    n_resamples: int,
) -> dict:
    """Retrain the served predictor without each test dataset and route that dataset."""
    entries: dict[str, dict] = {}
    quality_parts: list[StrategyOutcome] = []
    strongest_parts: list[StrategyOutcome] = []
    strata: list[np.ndarray] = []
    for dataset in sorted(set(test.datasets.tolist())):
        fitted = fit_pipeline(
            _without(train, dataset),
            _without(calibration, dataset),
            prices,
            kinds=(served,),
            lr_c=lr_c,
        )
        certs = certify(fitted, served, _without(certification, dataset), prices)
        held = test.subset(test.datasets == dataset)
        lcb, exp_cost = simulate(fitted, served, held, prices)
        lambdas = {cert.name: cert.lambda_ for cert in certs}
        quality = {}
        for profile in PROFILE_NAMES:
            chosen, escalated = route_quality(lcb, exp_cost, fitted.strongest, lambdas[profile])
            quality[profile] = _outcome(
                f"lodo/quality/{profile}", chosen, held, fitted.strongest, escalated
            )
        strongest = _outcome(
            "lodo/always_strongest",
            np.full(len(held), fitted.strongest, dtype=np.int64),
            held,
            fitted.strongest,
        )
        entries[dataset] = {
            "n": len(held),
            "strongest_model": fitted.pool[fitted.strongest],
            "lambdas": lambdas,
            "quality": {profile: summarize(o) for profile, o in quality.items()},
            "always_strongest": summarize(strongest),
        }
        quality_parts.append(quality[H1_PROFILE])
        strongest_parts.append(strongest)
        strata.append(np.full(len(held), dataset))

    pooled_quality = _concat(f"lodo/quality/{H1_PROFILE}", quality_parts)
    pooled_strongest = _concat("lodo/always_strongest", strongest_parts)
    values = np.column_stack(
        [
            pooled_quality.success,
            pooled_quality.cost,
            pooled_strongest.success,
            pooled_strongest.cost,
        ]
    )
    totals = bootstrap_totals(np.concatenate(strata), values, n_resamples=n_resamples)
    comparison = compare(pooled_quality, pooled_strongest, totals[:, 0:2], totals[:, 2:4])
    pooled = {
        "profile": H1_PROFILE,
        "predictor": served,
        "n": len(pooled_quality.success),
        "n_resamples": n_resamples,
        **comparison["relative_cps_reduction"],
        "accuracy_diff": comparison["accuracy_diff"],
        "cost_per_success_diff": comparison["cost_per_success_diff"],
        "degenerate_resamples": comparison["degenerate_resamples"],
        "quality": summarize(pooled_quality),
        "always_strongest": summarize(pooled_strongest),
    }
    return {"datasets": entries, "pooled": pooled}


def _test_costs(rows: Sequence[OutcomeRow], prices: PriceTable, pool: Sequence[str]) -> dict:
    """Recorded versus list-price totals of the test rows, per model and per dataset and model."""
    recorded: dict[tuple[str, str], list[float]] = {}
    listed: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (row.dataset, row.model)
        recorded.setdefault(key, []).append(row.cost_usd)
        listed.setdefault(key, []).append(
            list_cost_usd(prices.price(row.model), row.prompt_tokens, row.completion_tokens)
        )
    by_model = {
        model: {
            "recorded_usd": math.fsum(
                cost for (_, name), costs in recorded.items() if name == model for cost in costs
            ),
            "list_usd": math.fsum(
                cost for (_, name), costs in listed.items() if name == model for cost in costs
            ),
        }
        for model in pool
    }
    by_dataset_model = []
    for dataset in sorted({dataset for dataset, _ in recorded}):
        for model in pool:
            if (dataset, model) not in recorded:
                continue
            total_recorded = math.fsum(recorded[(dataset, model)])
            total_listed = math.fsum(listed[(dataset, model)])
            by_dataset_model.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "recorded_usd": total_recorded,
                    "list_usd": total_listed,
                    "ratio": total_recorded / total_listed if total_listed > 0 else None,
                }
            )
    return {"by_model": by_model, "by_dataset_model": by_dataset_model}


def _write_decisions(
    path: Path,
    test: OutcomeTable,
    quality: Mapping[str, tuple[np.ndarray, np.ndarray]],
    simple: Mapping[str, np.ndarray],
) -> None:
    """Gateway decisions per test prompt: quality for each profile, then the four references."""
    pool = test.pool
    with atomic_writer(path) as f:
        for i, prompt_id in enumerate(test.prompt_ids):
            sha = prompt_sha256(test.texts[i])
            for profile in PROFILE_NAMES:
                chosen, escalated = quality[profile]
                reason = REASON_ESCALATED if escalated[i] else REASON_PASSING
                f.write(_decision(prompt_id, sha, "quality", profile, pool[chosen[i]], reason))
            for strategy in ("strongest", "cheapest", "static", "latency"):
                f.write(_decision(prompt_id, sha, strategy, None, pool[simple[strategy][i]], None))


def _decision(
    prompt_id: str, sha: str, strategy: str, profile: str | None, chosen: str, reason: str | None
) -> str:
    return dumps_line(
        {
            "prompt_id": prompt_id,
            "prompt_sha256": sha,
            "strategy": strategy,
            "profile": profile,
            "chosen": chosen,
            "reason": reason,
        }
    )


def _test_checksum(manifest: Mapping[str, Any]) -> str:
    entry = manifest["files"].get("test")
    sha = entry.get("sha256") if isinstance(entry, Mapping) else None
    if not isinstance(sha, str):
        raise TrainerError("the data manifest records no sha256 for the test split")
    return sha


def run_evaluate(
    paths: Paths,
    *,
    git_sha: str | None = None,
    now: datetime | None = None,
    n_resamples: int = N_RESAMPLES,
    n_random_seeds: int = 1_000,
) -> dict:
    """Evaluate the trained policy on the held-out test split; writes and returns the report.

    The test split is the only file read through `read_test_split`, after its checksum is
    verified against the manifest. The other splits are verified the same way as in `train`.
    """
    now = datetime.now(UTC) if now is None else now
    git_sha = current_git_sha(paths.root) if git_sha is None else git_sha
    manifest = read_manifest(paths.manifest)
    prices = load_prices(paths.prices)
    pool = tuple(manifest["pool"])
    test_sha = _test_checksum(manifest)
    try:
        test_rows = list(read_test_split(paths.splits_dir, test_sha))
        test = pivot(test_rows, pool)
    except (OSError, ValueError) as exc:
        raise TrainerError(f"cannot load the test split: {exc}") from exc
    train = load_table(paths, manifest, "train")
    calibration = load_table(paths, manifest, "calibration")
    certification = load_table(paths, manifest, "certification")

    policy = load_policy(paths.policy)
    _check_pool(paths.policy, [entry.name for entry in policy.models], pool)
    served = policy.predictor_type
    lr_c = _main_lr_c(paths, policy)
    strongest = pool.index(policy.strongest_model)
    expected_tokens = {entry.name: entry.expected_output_tokens for entry in policy.models}
    exp_cost = expected_cost_usd(test.prompt_bytes, pool, prices, expected_tokens)
    latency_index = route_latency(
        pool, load_latency_profiles(paths.latency_profiles), expected_tokens
    )

    train_features = featurize(request_texts(train))
    cal_features = featurize(request_texts(calibration))
    test_features = featurize(request_texts(test))
    scores = {
        kind: _policy_scores(paths, kind, policy, pool, cal_features, test_features)
        for kind in ("lr", "gbm")
    }
    scores["knn"] = _knn_scores(paths, pool, train, train_features, cal_features, test_features)

    outcomes, quality, simple = _strategies(
        test, policy, scores, exp_cost, latency_index, n_random_seeds
    )
    strategies, per_dataset = _summaries(outcomes, scores, test)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_iso(now),
        "git_sha": git_sha,
        "policy_version": policy.version,
        "served_predictor": served,
        "source": manifest["source"],
        "test": {
            "n": len(test),
            "sha256": test_sha,
            "pool": list(pool),
            "by_dataset": {
                dataset: int((test.datasets == dataset).sum())
                for dataset in sorted(set(test.datasets.tolist()))
            },
        },
        "strategies": strategies,
        "per_dataset": per_dataset,
        "lambda_sweep": {
            kind: _lambda_sweep(scores[kind].test_lcb, exp_cost, test, strongest) for kind in KINDS
        },
        "calibration": _calibration_report(scores, test, calibration),
        "statistics": _statistics(outcomes, test, served, n_resamples, n_random_seeds),
        "lodo": _lodo(
            served=served,
            lr_c=lr_c,
            train=train,
            calibration=calibration,
            certification=certification,
            test=test,
            prices=prices,
            n_resamples=n_resamples,
        ),
        "costs": _test_costs(test_rows, prices, pool),
        "extraction": extraction_status(paths),
        "notes": [GUARANTEE_NOTE, TEMPLATE_NOTE, LATENCY_NOTE, COST_NOTE],
    }
    report["h1"] = h1_verdict(report)
    # Render from the JSON round trip, the same input a later re-render from the file sees.
    report = json.loads(json.dumps(report, allow_nan=False))

    for kind in KINDS:
        reliability_png(
            paths.reports / f"reliability-{kind}.png",
            f"Reliability on the test split ({kind})",
            {model: entry["reliability"] for model, entry in report["calibration"][kind].items()},
        )
    _write_decisions(paths.replay_dir / "decisions.jsonl", test, quality, simple)
    write_json(paths.reports / "evaluation.json", report)
    with atomic_writer(paths.reports / "evaluation.md") as f:
        f.write(render_evaluation_markdown(report))
    return report


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "evaluate", help="evaluate the trained policy on the held-out test split"
    )
    parser.set_defaults(func=_run)


def _run(paths: Paths, _args: argparse.Namespace) -> int:
    report = run_evaluate(paths)
    verdict = "falsified" if report["h1"]["falsified"] else "supported"
    path = (paths.reports / "evaluation.json").relative_to(paths.root).as_posix()
    print(f"evaluate OK: {path}; H1 {verdict}")
    return 0
