import dataclasses
import inspect
import json

import numpy as np
import pytest

from routeforge_train.calibration import fit_calibration
from routeforge_train.certify import LAMBDA_GRID, PROFILES, ProfileCert, regret_counts
from routeforge_train.errors import TrainerError
from routeforge_train.features import VOCAB_MAX, DenseScaler, featurize
from routeforge_train.jsonio import sha256_file, write_json
from routeforge_train.lr import C_GRID
from routeforge_train.outcomes import read_outcomes, write_outcomes
from routeforge_train.pipeline import (
    Candidate,
    Fitted,
    Selection,
    certify,
    fit_pipeline,
    request_texts,
    routing_metrics,
    select_served,
    simulate,
    to_policy,
)
from routeforge_train.policy import PREDICTOR_TYPES, Policy
from routeforge_train.routing import expected_cost_usd, strongest_index
from routeforge_train.splits import split_path
from routeforge_train.train import run_train
from tests.synth import FIXED_NOW, SYNTH_POOL, build_synthetic_project

METRIC_KEYS = {"lambda", "accuracy", "total_cost_usd", "cost_per_success_usd", "regret_rate"}


def test_expected_output_tokens_is_train_median(fitted, synth_tables):
    for j, model in enumerate(SYNTH_POOL):
        want = float(np.median(synth_tables.train.completion_tokens[:, j]))
        assert fitted.expected_output_tokens[model] == want
    assert list(fitted.expected_output_tokens) == list(SYNTH_POOL)


def test_strongest_is_highest_train_accuracy(fitted, synth_tables):
    train = synth_tables.train
    assert fitted.strongest == strongest_index(train.success, train.cost)
    assert fitted.pool[fitted.strongest] == "m-strong"
    assert train.success.mean(axis=0).argmax() == fitted.strongest


def test_cheapest_is_lowest_train_mean_expected_cost(fitted, synth_tables):
    train = synth_tables.train
    exp_cost = expected_cost_usd(
        train.prompt_bytes, SYNTH_POOL, synth_tables.prices, fitted.expected_output_tokens
    )
    assert fitted.cheapest == int(np.argmin(exp_cost.mean(axis=0)))
    assert fitted.pool[fitted.cheapest] == "m-cheap"


def test_fit_pipeline_builds_every_candidate(fitted, synth_tables):
    assert fitted.pool == SYNTH_POOL
    assert list(fitted.candidates) == ["lr", "gbm", "knn"]
    assert set(fitted.lr_c) == set(SYNTH_POOL) and set(fitted.lr_c.values()) <= set(C_GRID)
    assert all(tuple(cv) == C_GRID for cv in fitted.lr_cv.values())
    assert fitted.vocab.dtype == np.int64 and np.all(np.diff(fitted.vocab) > 0)
    assert 0 < len(fitted.vocab) <= VOCAB_MAX
    for kind, candidate in fitted.candidates.items():
        assert candidate.kind == kind
        assert len(candidate.calibrations) == len(SYNTH_POOL)
        for calibration in candidate.calibrations:
            assert calibration.blocks[0].lo == 0.0 and calibration.blocks[-1].hi == 1.0
    assert fitted.candidates["knn"].predictors is None
    assert fitted.candidates["knn"].knn.k == 25
    assert fitted.candidates["lr"].knn is None and len(fitted.candidates["gbm"].predictors) == 3
    assert fitted.scaler == DenseScaler.fit(featurize(request_texts(synth_tables.train)).dense_raw)


def test_request_texts_are_feature_text_of_one_user_message(synth_tables):
    texts = request_texts(synth_tables.train)
    assert texts == list(synth_tables.train.texts)


def test_fit_pipeline_respects_kinds_and_given_c(synth_tables):
    given = dict.fromkeys(SYNTH_POOL, 1.0)
    fitted = fit_pipeline(
        synth_tables.train,
        synth_tables.calibration,
        synth_tables.prices,
        kinds=("lr",),
        lr_c=given,
    )
    assert list(fitted.candidates) == ["lr"]
    assert fitted.lr_c == given and fitted.lr_cv == {}


def test_fit_pipeline_rejects_unknown_or_empty_kinds(synth_tables):
    args = (synth_tables.train, synth_tables.calibration, synth_tables.prices)
    with pytest.raises(ValueError, match="svm"):
        fit_pipeline(*args, kinds=("lr", "svm"))
    with pytest.raises(ValueError, match="kinds"):
        fit_pipeline(*args, kinds=())


def test_fit_pipeline_rejects_tables_with_different_pools(synth_tables):
    other = dataclasses.replace(synth_tables.calibration, pool=("a", "b", "c"))
    with pytest.raises(ValueError, match="pool"):
        fit_pipeline(synth_tables.train, other, synth_tables.prices, kinds=("lr",))


def test_fit_uses_train_rows_only_and_calibration_only_for_blocks(synth_tables):
    given = dict.fromkeys(SYNTH_POOL, 1.0)
    cal = synth_tables.calibration
    flipped = dataclasses.replace(cal, success=~cal.success)
    args = (synth_tables.train,)
    a = fit_pipeline(*args, cal, synth_tables.prices, kinds=("lr",), lr_c=given)
    b = fit_pipeline(*args, flipped, synth_tables.prices, kinds=("lr",), lr_c=given)
    assert [m.to_json() for m in a.candidates["lr"].predictors] == [
        m.to_json() for m in b.candidates["lr"].predictors
    ]
    assert a.scaler == b.scaler and np.array_equal(a.vocab, b.vocab)
    assert a.expected_output_tokens == b.expected_output_tokens
    assert (a.strongest, a.cheapest, a.rules) == (b.strongest, b.cheapest, b.rules)
    assert a.candidates["lr"].calibrations != b.candidates["lr"].calibrations
    feats = featurize(request_texts(cal))
    raw = a.candidates["lr"].raw(feats, a.scaler)
    for j in range(len(SYNTH_POOL)):
        assert a.candidates["lr"].calibrations[j] == fit_calibration(raw[:, j], cal.success[:, j])


def test_fit_pipeline_rules_use_the_pool_and_train_families(fitted):
    assert set(fitted.rules.family_model.values()) <= set(SYNTH_POOL)
    assert set(fitted.rules.family_order) == {"reasoning", "code", "qa", "instruction"}


def test_fit_pipeline_single_class_model_names_it(synth_tables):
    train = synth_tables.train
    success = train.success.copy()
    success[:, SYNTH_POOL.index("m-cheap")] = True
    single = dataclasses.replace(train, success=success)
    with pytest.raises(TrainerError, match="m-cheap") as info:
        fit_pipeline(single, synth_tables.calibration, synth_tables.prices)
    assert "lr" in str(info.value) and "single class" in str(info.value)


def test_gbm_single_class_model_is_named_too(synth_tables):
    train = synth_tables.train
    success = train.success.copy()
    success[:, SYNTH_POOL.index("m-mid")] = False
    single = dataclasses.replace(train, success=success)
    with pytest.raises(TrainerError, match="m-mid") as info:
        fit_pipeline(single, synth_tables.calibration, synth_tables.prices, kinds=("gbm",))
    assert "gbm" in str(info.value)


def test_knn_only_fit_does_not_need_two_classes_per_model(synth_tables):
    train = synth_tables.train
    success = train.success.copy()
    success[:, SYNTH_POOL.index("m-cheap")] = True
    single = dataclasses.replace(train, success=success)
    fitted = fit_pipeline(single, synth_tables.calibration, synth_tables.prices, kinds=("knn",))
    assert list(fitted.candidates) == ["knn"] and fitted.lr_c == {} and fitted.lr_cv == {}


def test_candidate_without_predictors_or_knn_cannot_score():
    scaler = DenseScaler(mean=(0.0,) * 8, std=(1.0,) * 8)
    with pytest.raises(ValueError, match="neither"):
        Candidate("lr", None, None, ()).raw(featurize(["a"]), scaler)


def test_candidate_scores_agree_with_raw_and_calibration(fitted, synth_tables):
    feats = featurize(request_texts(synth_tables.certification))
    for kind, candidate in fitted.candidates.items():
        raw = candidate.raw(feats, fitted.scaler)
        s_raw, p, lcb = candidate.scores(feats, fitted.scaler)
        n = len(synth_tables.certification)
        assert raw.shape == s_raw.shape == p.shape == lcb.shape == (n, len(SYNTH_POOL)), kind
        assert np.array_equal(raw, s_raw)
        for j, calibration in enumerate(candidate.calibrations):
            assert np.array_equal(p[:, j], calibration.p_of(raw[:, j]))
            assert np.array_equal(lcb[:, j], calibration.lcb_of(raw[:, j]))
        assert np.all((raw >= 0.0) & (raw <= 1.0))


def test_simulate_returns_lcb_and_expected_cost(fitted, synth_tables):
    table = synth_tables.certification
    lcb, exp_cost = simulate(fitted, "lr", table, synth_tables.prices)
    n = len(table)
    assert lcb.shape == exp_cost.shape == (n, len(SYNTH_POOL))
    want = expected_cost_usd(
        table.prompt_bytes, SYNTH_POOL, synth_tables.prices, fitted.expected_output_tokens
    )
    assert np.array_equal(exp_cost, want)
    feats = featurize(request_texts(table))
    assert np.array_equal(lcb, fitted.candidates["lr"].scores(feats, fitted.scaler)[2])


def test_certify_yields_the_three_profiles_for_every_kind(fitted, synth_tables):
    n = len(synth_tables.certification)
    for kind in fitted.candidates:
        certs = certify(fitted, kind, synth_tables.certification, synth_tables.prices)
        assert [c.name for c in certs] == [name for name, _ in PROFILES]
        assert all(isinstance(c, ProfileCert) and c.cert_n == n for c in certs)


def test_routing_metrics_hand_computed():
    chosen = np.array([0, 1, 1])
    success = np.array([[True, False], [False, True], [True, False]])
    cost = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    got = routing_metrics(chosen, success, cost, strongest=0)
    assert got == {
        "accuracy": 2 / 3,
        "total_cost_usd": 11.0,
        "cost_per_success_usd": 5.5,
        "regret_rate": 1 / 3,
    }


def test_routing_metrics_without_successes_has_no_cost_per_success():
    got = routing_metrics(
        np.array([0, 0]), np.zeros((2, 2), dtype=bool), np.ones((2, 2)), strongest=1
    )
    assert got["accuracy"] == 0.0 and got["cost_per_success_usd"] is None
    json.dumps(got, allow_nan=False)


def test_routing_metrics_rejects_empty_input():
    with pytest.raises(ValueError, match="no rows"):
        routing_metrics(
            np.array([], dtype=np.int64), np.zeros((0, 2), dtype=bool), np.zeros((0, 2)), 0
        )


def test_select_served_compares_the_balanced_profile_on_calibration(fitted, synth_tables):
    selection = select_served(fitted, synth_tables.calibration, synth_tables.prices)
    assert isinstance(selection, Selection)
    assert selection.profile == "balanced"
    assert selection.calibration_n == len(synth_tables.calibration)
    assert list(selection.candidates) == list(PREDICTOR_TYPES)
    assert all(set(m) == METRIC_KEYS for m in selection.candidates.values())
    cps = {k: m["cost_per_success_usd"] for k, m in selection.candidates.items()}
    assert selection.winner == min(cps, key=lambda k: (cps[k], PREDICTOR_TYPES.index(k)))
    json.dumps(selection.candidates, allow_nan=False)


def test_select_served_balanced_lambda_matches_calibration_certification(fitted, synth_tables):
    selection = select_served(fitted, synth_tables.calibration, synth_tables.prices)
    for kind, metrics in selection.candidates.items():
        certs = certify(fitted, kind, synth_tables.calibration, synth_tables.prices)
        assert metrics["lambda"] == next(c.lambda_ for c in certs if c.name == "balanced")


def test_select_served_breaks_ties_to_lr(fitted, synth_tables):
    twin = Candidate(
        "gbm", fitted.candidates["lr"].predictors, None, fitted.candidates["lr"].calibrations
    )
    tied = dataclasses.replace(fitted, candidates={"lr": fitted.candidates["lr"], "gbm": twin})
    selection = select_served(tied, synth_tables.calibration, synth_tables.prices)
    assert selection.candidates["lr"] == selection.candidates["gbm"]
    assert selection.winner == "lr"


def test_select_served_never_takes_a_certification_table():
    assert list(inspect.signature(select_served).parameters) == [
        "fitted",
        "calibration",
        "prices",
    ]


def test_select_served_needs_a_served_type(fitted, synth_tables):
    only_knn = dataclasses.replace(fitted, candidates={"knn": fitted.candidates["knn"]})
    with pytest.raises(TrainerError, match="served"):
        select_served(only_knn, synth_tables.calibration, synth_tables.prices)


@pytest.mark.parametrize("kind", PREDICTOR_TYPES)
def test_to_policy_carries_the_fitted_values(kind, fitted, synth_tables):
    profiles = certify(fitted, kind, synth_tables.certification, synth_tables.prices)
    source = {"dataset": "d", "revision": "r", "git_sha": "g"}
    policy = to_policy(fitted, kind, profiles, source, "2026-10-26T00:00:00Z")
    assert isinstance(policy, Policy) and policy.predictor_type == kind
    assert [m.name for m in policy.models] == list(SYNTH_POOL)
    assert policy.strongest_model == "m-strong" and policy.cheapest_model == "m-cheap"
    assert policy.profiles == profiles and policy.source == source
    assert policy.scaler == fitted.scaler and policy.static_rules == fitted.rules
    for entry, calibration in zip(policy.models, fitted.candidates[kind].calibrations, strict=True):
        assert entry.calibration == calibration
        assert entry.expected_output_tokens == fitted.expected_output_tokens[entry.name]


def test_to_policy_refuses_offline_only_kinds(fitted, synth_tables):
    profiles = certify(fitted, "knn", synth_tables.certification, synth_tables.prices)
    with pytest.raises(ValueError, match="knn"):
        to_policy(fitted, "knn", profiles, {}, "2026-10-26T00:00:00Z")


def test_fitted_is_a_frozen_value(fitted):
    assert isinstance(fitted, Fitted)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fitted.strongest = 1  # type: ignore[misc]


def _flip_certification(paths) -> str:
    """Invert every certification label and record the new checksum in the manifest."""
    path = split_path(paths.splits_dir, "certification")
    rows = [
        dataclasses.replace(r, score=1.0 - r.score, success=not r.success)
        for r in read_outcomes(path)
    ]
    info = write_outcomes(rows, path, paths.root)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["files"]["certification"] = info.to_json()
    write_json(paths.manifest, manifest)
    return info.sha256


def test_flipping_certification_labels_changes_the_regret_counts(fitted, synth_tables):
    certification = synth_tables.certification
    lcb, exp_cost = simulate(fitted, "lr", certification, synth_tables.prices)
    original = regret_counts(lcb, exp_cost, certification.success, fitted.strongest)
    inverted = regret_counts(lcb, exp_cost, ~certification.success, fitted.strongest)
    assert len(original) == len(inverted) == len(LAMBDA_GRID)
    assert original != inverted


def test_selection_never_sees_certification(trained_paths, training, tmp_path):
    flipped_paths = build_synthetic_project(tmp_path)
    original_sha = json.loads(trained_paths.manifest.read_text(encoding="utf-8"))["files"][
        "certification"
    ]["sha256"]
    flipped_sha = _flip_certification(flipped_paths)
    assert flipped_sha != original_sha
    assert flipped_sha == sha256_file(split_path(flipped_paths.splits_dir, "certification"))

    run_train(flipped_paths, git_sha="synthetic", now=FIXED_NOW)
    flipped = json.loads((flipped_paths.reports / "training.json").read_text(encoding="utf-8"))

    assert flipped["data"]["manifest_sha256"] != training["data"]["manifest_sha256"]
    assert flipped["selection"] == training["selection"]
    assert flipped["selection"]["winner"] == training["selection"]["winner"]
    assert flipped["lr"] == training["lr"]
    assert flipped["calibration"] == training["calibration"]
    assert flipped["expected_output_tokens"] == training["expected_output_tokens"]
    assert (
        flipped["certification"]["served"][0]["cert_n"]
        == (training["certification"]["served"][0]["cert_n"])
    )


def test_offline_kinds_certified_but_not_exported(trained_paths, training):
    winner = training["selection"]["winner"]
    offline = training["certification"]["offline"]
    assert winner in PREDICTOR_TYPES
    assert set(offline) == {"lr", "gbm", "knn"} - {winner}
    policy_doc_ = json.loads(trained_paths.policy.read_text(encoding="utf-8"))
    assert policy_doc_["predictor_type"] == winner
    assert [p["name"] for p in training["certification"]["served"]] == [n for n, _ in PROFILES]
    assert policy_doc_["profiles"] == training["certification"]["served"]
    for profiles in offline.values():
        assert [p["name"] for p in profiles] == [n for n, _ in PROFILES]
