"""Session fixtures shared by the trainer tests. Synthetic data only."""

import json
from dataclasses import dataclass

import pytest

from routeforge_train.certify import ProfileCert
from routeforge_train.pipeline import Fitted, certify, fit_pipeline, to_policy
from routeforge_train.policy import Policy
from routeforge_train.prices import PriceTable, load_prices
from routeforge_train.splits import read_split
from routeforge_train.table import OutcomeTable, pivot
from routeforge_train.train import run_train
from tests.synth import FIXED_NOW, SYNTH_POOL, SYNTH_SOURCE, build_synthetic_project

POLICY_SOURCE = {
    "dataset": SYNTH_SOURCE["dataset"],
    "revision": SYNTH_SOURCE["revision"],
    "git_sha": "synthetic",
}
POLICY_CREATED_AT = "2026-10-26T00:00:00Z"


@dataclass(frozen=True)
class SynthTables:
    train: OutcomeTable
    calibration: OutcomeTable
    certification: OutcomeTable
    prices: PriceTable


@pytest.fixture(scope="session")
def trained_paths(tmp_path_factory):
    """A synthetic project after one full `run_train`; tests must not modify it."""
    paths = build_synthetic_project(tmp_path_factory.mktemp("trained"))
    run_train(paths, git_sha="synthetic", now=FIXED_NOW)
    return paths


@pytest.fixture(scope="session")
def training(trained_paths):
    return json.loads(
        (trained_paths.reports / "training.json").read_text(encoding="utf-8"),
    )


@pytest.fixture(scope="session")
def synth_tables(tmp_path_factory) -> SynthTables:
    paths = build_synthetic_project(tmp_path_factory.mktemp("tables"))
    tables = {
        name: pivot(read_split(paths.splits_dir, name), SYNTH_POOL)
        for name in ("train", "calibration", "certification")
    }
    return SynthTables(prices=load_prices(paths.prices), **tables)


@pytest.fixture(scope="session")
def fitted(synth_tables) -> Fitted:
    """LR, GBM and kNN candidates fit on the synthetic train split."""
    return fit_pipeline(synth_tables.train, synth_tables.calibration, synth_tables.prices)


def _build_policy(fitted: Fitted, tables: SynthTables, kind: str) -> Policy:
    profiles: tuple[ProfileCert, ...] = certify(fitted, kind, tables.certification, tables.prices)
    return to_policy(fitted, kind, profiles, dict(POLICY_SOURCE), POLICY_CREATED_AT)


@pytest.fixture(scope="session")
def lr_policy(fitted, synth_tables) -> Policy:
    return _build_policy(fitted, synth_tables, "lr")


@pytest.fixture(scope="session")
def gbm_policy(fitted, synth_tables) -> Policy:
    return _build_policy(fitted, synth_tables, "gbm")
