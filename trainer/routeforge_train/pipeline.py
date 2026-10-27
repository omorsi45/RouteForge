"""Fit, select and certify the routing candidates (spec 8.2 to 8.5).

Shared by the `train` command and the leave-one-dataset-out runs. Data isolation is
structural: `fit_pipeline` sees the train and calibration tables only, `select_served` sees
the calibration table only, and `certify` is the one function that takes the certification
table.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from routeforge_train.calibration import Calibration, fit_calibration
from routeforge_train.certify import ProfileCert, certify_profiles
from routeforge_train.errors import TrainerError
from routeforge_train.features import (
    DenseScaler,
    Features,
    design_matrix,
    feature_text,
    featurize,
    select_vocab,
)
from routeforge_train.gbm import GBMModel, fit_gbm
from routeforge_train.knn import KNNModel, fit_knn
from routeforge_train.lr import LRModel, choose_c, fit_lr
from routeforge_train.metrics import StrategyOutcome, summarize
from routeforge_train.policy import PREDICTOR_TYPES, ModelEntry, Policy
from routeforge_train.prices import PriceTable
from routeforge_train.routing import (
    expected_cost_usd,
    regret_mask,
    route_quality,
    strongest_index,
)
from routeforge_train.rules import KeywordRules, fit_keyword_rules
from routeforge_train.table import OutcomeTable

KINDS = ("lr", "gbm", "knn")
SELECTION_PROFILE = "balanced"


@dataclass(frozen=True, eq=False)
class Candidate:
    """One predictor type for the whole pool, with a calibration per model."""

    kind: str
    predictors: tuple[LRModel | GBMModel, ...] | None
    knn: KNNModel | None
    calibrations: tuple[Calibration, ...]

    def raw(self, features: Features, scaler: DenseScaler) -> np.ndarray:
        """Shape (n, M): the uncalibrated success score of every model."""
        if self.predictors is None:
            if self.knn is None:
                raise ValueError(f"candidate {self.kind!r} has neither predictors nor a kNN model")
            return self.knn.raw(features.hashed)
        dense_z = scaler.transform(features.dense_raw)
        return np.column_stack([model.raw(features.hashed, dense_z) for model in self.predictors])

    def scores(
        self, features: Features, scaler: DenseScaler
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw score, calibrated success rate `p` and Wilson lower bound, each (n, M)."""
        raw = self.raw(features, scaler)
        p = np.column_stack([c.p_of(raw[:, j]) for j, c in enumerate(self.calibrations)])
        lcb = np.column_stack([c.lcb_of(raw[:, j]) for j, c in enumerate(self.calibrations)])
        return raw, p, lcb


@dataclass(frozen=True, eq=False)
class Fitted:
    pool: tuple[str, ...]
    scaler: DenseScaler
    vocab: np.ndarray
    strongest: int
    cheapest: int
    expected_output_tokens: dict[str, float]
    rules: KeywordRules
    lr_c: dict[str, float]
    lr_cv: dict[str, dict[float, float]]
    candidates: dict[str, Candidate]


@dataclass(frozen=True)
class Selection:
    winner: str
    profile: str
    calibration_n: int
    candidates: dict[str, dict]


def request_texts(table: OutcomeTable) -> list[str]:
    """Feature text of each prompt, sent as one user message as the gateway would see it."""
    return [feature_text([{"role": "user", "content": text}]) for text in table.texts]


def _check_labels(train: OutcomeTable, kinds: Sequence[str]) -> None:
    for kind in kinds:
        if kind == "knn":
            continue
        for j, model in enumerate(train.pool):
            if len(np.unique(train.success[:, j])) < 2:
                raise TrainerError(f"model {model!r}: {kind} training labels have a single class")


def fit_pipeline(
    train: OutcomeTable,
    calibration: OutcomeTable,
    prices: PriceTable,
    *,
    kinds: Sequence[str] = KINDS,
    lr_c: Mapping[str, float] | None = None,
) -> Fitted:
    """Fit every predictor on the train rows and calibrate it on the calibration rows."""
    kinds = tuple(kinds)
    if not kinds:
        raise ValueError("kinds must name at least one predictor type")
    for kind in kinds:
        if kind not in KINDS:
            raise ValueError(f"unknown predictor kind {kind!r}; expected one of {KINDS}")
    if len(set(kinds)) != len(kinds):
        raise ValueError("kinds must not repeat")
    if train.pool != calibration.pool:
        raise ValueError(
            f"train pool {train.pool} differs from calibration pool {calibration.pool}"
        )
    _check_labels(train, kinds)

    pool = train.pool
    texts = request_texts(train)
    features = featurize(texts)
    scaler = DenseScaler.fit(features.dense_raw)
    vocab = select_vocab(features.hashed)
    design = design_matrix(features.hashed, scaler.transform(features.dense_raw), vocab)
    expected_output_tokens = {
        model: float(np.median(train.completion_tokens[:, j])) for j, model in enumerate(pool)
    }
    exp_cost = expected_cost_usd(train.prompt_bytes, pool, prices, expected_output_tokens)
    rules = fit_keyword_rules(texts, train.families.tolist(), train.success, exp_cost, pool)

    calibration_features = featurize(request_texts(calibration))
    chosen_c: dict[str, float] = {}
    cv_losses: dict[str, dict[float, float]] = {}
    candidates: dict[str, Candidate] = {}
    for kind in kinds:
        predictors: tuple[LRModel | GBMModel, ...] | None = None
        knn: KNNModel | None = None
        if kind == "lr":
            models = []
            for j, model in enumerate(pool):
                y = train.success[:, j].astype(np.int64)
                c = None if lr_c is None else lr_c.get(model)
                if c is None:
                    c, cv_losses[model] = choose_c(design, y)
                chosen_c[model] = float(c)
                models.append(fit_lr(design, y, vocab, c))
            predictors = tuple(models)
        elif kind == "gbm":
            predictors = tuple(
                fit_gbm(design, train.success[:, j].astype(np.int64), vocab)
                for j in range(len(pool))
            )
        else:
            knn = fit_knn(features.hashed, train.success)
        draft = Candidate(kind, predictors, knn, ())
        raw = draft.raw(calibration_features, scaler)
        calibrations = tuple(
            fit_calibration(raw[:, j], calibration.success[:, j]) for j in range(len(pool))
        )
        candidates[kind] = dataclasses.replace(draft, calibrations=calibrations)

    return Fitted(
        pool=pool,
        scaler=scaler,
        vocab=vocab,
        strongest=strongest_index(train.success, train.cost),
        cheapest=int(np.argmin(exp_cost.mean(axis=0))),
        expected_output_tokens=expected_output_tokens,
        rules=rules,
        lr_c=chosen_c,
        lr_cv=cv_losses,
        candidates=candidates,
    )


def simulate(
    fitted: Fitted, kind: str, table: OutcomeTable, prices: PriceTable
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson lower bounds and expected USD costs, each (n, M), as the gateway would see them."""
    features = featurize(request_texts(table))
    _, _, lcb = fitted.candidates[kind].scores(features, fitted.scaler)
    exp_cost = expected_cost_usd(
        table.prompt_bytes, fitted.pool, prices, fitted.expected_output_tokens
    )
    return lcb, exp_cost


def certify(
    fitted: Fitted, kind: str, table: OutcomeTable, prices: PriceTable
) -> tuple[ProfileCert, ...]:
    """Learn-then-Test lambdas for the three profiles on `table`."""
    lcb, exp_cost = simulate(fitted, kind, table, prices)
    return certify_profiles(lcb, exp_cost, table.success, fitted.strongest)


def routing_metrics(
    chosen: np.ndarray, success: np.ndarray, cost: np.ndarray, strongest: int
) -> dict[str, float | None]:
    """Accuracy, recorded cost, cost per success and regret rate of routing `chosen`.

    `cost_per_success_usd` is None when nothing succeeded, so the dict stays valid JSON.
    """
    n = len(chosen)
    if n == 0:
        raise ValueError("cannot compute routing metrics for no rows")
    rows = np.arange(n)
    summary = summarize(
        StrategyOutcome(
            name="routed",
            success=success[rows, chosen],
            cost=cost[rows, chosen],
            regret=regret_mask(chosen, success, strongest),
            shares={},
        )
    )
    return {
        key: summary[key]
        for key in ("accuracy", "total_cost_usd", "cost_per_success_usd", "regret_rate")
    }


def select_served(fitted: Fitted, calibration: OutcomeTable, prices: PriceTable) -> Selection:
    """Pick the served predictor type on the calibration split alone (spec 8.5, step 3).

    Each served type is certified for the balanced profile on the calibration table, routed at
    that lambda, and compared by recorded cost per success; ties go to `lr`.
    """
    kinds = [kind for kind in PREDICTOR_TYPES if kind in fitted.candidates]
    if not kinds:
        raise TrainerError(f"no served predictor type ({', '.join(PREDICTOR_TYPES)}) was fitted")
    results: dict[str, dict] = {}
    for kind in kinds:
        lcb, exp_cost = simulate(fitted, kind, calibration, prices)
        certs = certify_profiles(lcb, exp_cost, calibration.success, fitted.strongest)
        lam = next(c.lambda_ for c in certs if c.name == SELECTION_PROFILE)
        chosen, _ = route_quality(lcb, exp_cost, fitted.strongest, lam)
        results[kind] = {
            "lambda": lam,
            **routing_metrics(chosen, calibration.success, calibration.cost, fitted.strongest),
        }

    def rank(kind: str) -> tuple[float, int]:
        cost_per_success = results[kind]["cost_per_success_usd"]
        return (math.inf if cost_per_success is None else cost_per_success, kinds.index(kind))

    return Selection(
        winner=min(kinds, key=rank),
        profile=SELECTION_PROFILE,
        calibration_n=len(calibration),
        candidates=results,
    )


def to_policy(
    fitted: Fitted,
    kind: str,
    profiles: Sequence[ProfileCert],
    source: Mapping[str, str],
    created_at: str,
) -> Policy:
    """The servable policy for one served predictor type."""
    if kind not in PREDICTOR_TYPES or kind not in fitted.candidates:
        raise ValueError(f"{kind!r} is not a fitted served predictor type")
    candidate = fitted.candidates[kind]
    predictors = candidate.predictors or ()
    return Policy(
        source=dict(source),
        scaler=fitted.scaler,
        predictor_type=kind,
        models=tuple(
            ModelEntry(
                name=name,
                predictor=predictors[j],
                calibration=candidate.calibrations[j],
                expected_output_tokens=fitted.expected_output_tokens[name],
            )
            for j, name in enumerate(fitted.pool)
        ),
        strongest_model=fitted.pool[fitted.strongest],
        cheapest_model=fitted.pool[fitted.cheapest],
        static_rules=fitted.rules,
        profiles=tuple(profiles),
        created_at=created_at,
    )
