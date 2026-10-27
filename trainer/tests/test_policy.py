import copy
import dataclasses
import hashlib
import json
import re

import numpy as np
import pytest

from routeforge_train import policy as policy_module
from routeforge_train.certify import LAMBDA_GRID
from routeforge_train.errors import TrainerError
from routeforge_train.features import DENSE_NAMES, HASH_DIM, featurize
from routeforge_train.jsonio import canonical_bytes
from routeforge_train.policy import (
    MAX_POLICY_BYTES,
    PREDICTOR_TYPES,
    SCHEMA_VERSION,
    Policy,
    PolicyError,
    PolicyTooLargeError,
    export_policy,
    load_policy,
    policy_doc,
    policy_version,
    score_policy,
    validate_policy_doc,
)

POLICY_FIXTURES = ["lr_policy", "gbm_policy"]


def test_constants_and_error_hierarchy():
    assert SCHEMA_VERSION == 1
    assert MAX_POLICY_BYTES == 5_000_000
    assert PREDICTOR_TYPES == ("lr", "gbm")
    assert issubclass(PolicyError, TrainerError)
    assert issubclass(PolicyTooLargeError, PolicyError)


def test_policy_version_is_first_12_hex_of_canonical_sha256(lr_policy):
    doc = policy_doc(lr_policy)
    body = {k: v for k, v in doc.items() if k not in ("version", "created_at")}
    assert doc["version"] == hashlib.sha256(canonical_bytes(body)).hexdigest()[:12]
    assert policy_version(doc) == doc["version"]
    assert re.fullmatch(r"[0-9a-f]{12}", doc["version"])


def test_policy_version_ignores_created_at(lr_policy):
    later = dataclasses.replace(lr_policy, created_at="2031-01-02T03:04:05Z")
    assert policy_doc(later)["created_at"] == "2031-01-02T03:04:05Z"
    assert policy_doc(later)["version"] == policy_doc(lr_policy)["version"]


def test_policy_version_covers_git_sha_and_content(lr_policy):
    other_sha = dataclasses.replace(lr_policy, source={**lr_policy.source, "git_sha": "abc"})
    other_model = dataclasses.replace(lr_policy, strongest_model=lr_policy.cheapest_model)
    versions = {policy_doc(p)["version"] for p in (lr_policy, other_sha, other_model)}
    assert len(versions) == 3


def test_policy_doc_has_the_spec_9_shape(lr_policy, gbm_policy):
    doc = policy_doc(lr_policy)
    assert set(doc) == {
        "schema_version",
        "version",
        "created_at",
        "source",
        "feature_spec",
        "predictor_type",
        "models",
        "strongest_model",
        "cheapest_model",
        "static_rules",
        "profiles",
    }
    assert doc["schema_version"] == 1 and doc["predictor_type"] == "lr"
    assert set(doc["source"]) == {"dataset", "revision", "git_sha"}
    spec = doc["feature_spec"]
    assert set(spec) == {"hash_dim", "ngram", "dense", "dense_mean", "dense_std"}
    assert spec["hash_dim"] == HASH_DIM == 262144
    assert spec["ngram"] == [1, 2] and spec["dense"] == list(DENSE_NAMES)
    assert [m["name"] for m in doc["models"]] == ["m-strong", "m-mid", "m-cheap"]
    assert set(doc["models"][0]) == {"name", "predictor", "calibration", "expected_output_tokens"}
    assert set(doc["static_rules"]) == {
        "family_keywords",
        "family_model",
        "family_order",
        "default_family",
    }
    assert [p["name"] for p in doc["profiles"]] == ["strict", "balanced", "economy"]
    assert set(doc["models"][0]["predictor"]) == {
        "intercept",
        "sparse_indices",
        "sparse_weights",
        "dense_weights",
    }
    assert policy_doc(gbm_policy)["models"][0]["predictor"]["feature_dtype"] == "float32"


@pytest.mark.parametrize("fixture", POLICY_FIXTURES)
def test_export_then_load_round_trip_scores_exact(fixture, request, tmp_path):
    policy = request.getfixturevalue(fixture)
    path = tmp_path / "policy.json"
    version, size = export_policy(policy, path)
    loaded = load_policy(path)
    feats = featurize(["a", "Write code", ""])
    want, got = score_policy(policy, feats), score_policy(loaded, feats)
    for name in ("raw", "p", "lcb", "block"):
        assert np.array_equal(getattr(got, name), getattr(want, name)), name
    assert loaded.version == version == policy_doc(policy)["version"]
    assert loaded.scaler == policy.scaler
    assert policy_doc(loaded) == policy_doc(policy)
    assert size == path.stat().st_size


def test_export_writes_canonical_bytes_with_one_trailing_newline(lr_policy, tmp_path):
    path = tmp_path / "out" / "policy.json"
    version, size = export_policy(lr_policy, path)
    data = path.read_bytes()
    assert data == canonical_bytes(policy_doc(lr_policy)) + b"\n"
    assert size == len(data) < MAX_POLICY_BYTES
    assert version == json.loads(data)["version"]
    assert [p.name for p in path.parent.iterdir()] == ["policy.json"]


def test_export_rejects_artifact_over_cap(lr_policy, tmp_path, monkeypatch):
    monkeypatch.setattr(policy_module, "MAX_POLICY_BYTES", 100)
    path = tmp_path / "policy.json"
    with pytest.raises(PolicyTooLargeError, match="100"):
        export_policy(lr_policy, path)
    assert not path.exists() and list(tmp_path.iterdir()) == []


def test_export_refuses_an_invalid_policy(lr_policy, tmp_path):
    broken = dataclasses.replace(lr_policy, strongest_model="nope")
    path = tmp_path / "policy.json"
    with pytest.raises(PolicyError, match="strongest_model"):
        export_policy(broken, path)
    assert not path.exists()


@pytest.mark.parametrize("fixture", POLICY_FIXTURES)
def test_validate_accepts_exported_documents(fixture, request):
    validate_policy_doc(policy_doc(request.getfixturevalue(fixture)))


GOOD_BLOCKS = [
    {"lo": 0.0, "hi": 0.5, "n": 50, "k": 20, "p": 0.4, "lcb": 0.3},
    {"lo": 0.5, "hi": 1.0, "n": 50, "k": 40, "p": 0.8, "lcb": 0.7},
]


def _with_blocks(change):
    def mutate(d):
        blocks = copy.deepcopy(GOOD_BLOCKS)
        change(blocks)
        d["models"][0]["calibration"]["blocks"] = blocks

    return mutate


def _set(blocks, i, **fields):
    blocks[i].update(fields)


def _internal_tree(d):
    return next(t for t in d["models"][0]["predictor"]["trees"] if t["left"][0] != -1)


def _swap_first_two_sparse_indices(d):
    idx = d["models"][0]["predictor"]["sparse_indices"]
    idx[0], idx[1] = idx[1], idx[0]


def _set_first_tree(d, key, position, value):
    _internal_tree(d)[key][position] = value


def _pop_first_tree(d, key):
    _internal_tree(d)[key].pop()


# (id, fixture, mutate, expected path fragment, recompute version after mutating)
INVALID_DOCUMENTS = [
    ("schema_version", "lr_policy", lambda d: d.update(schema_version=2), "schema_version", False),
    ("schema_bool", "lr_policy", lambda d: d.update(schema_version=True), "schema_version", False),
    ("version_not_hex", "lr_policy", lambda d: d.update(version="XYZ"), "version", False),
    ("version_wrong", "lr_policy", lambda d: d.update(version="0" * 12), "version", False),
    ("created_at", "lr_policy", lambda d: d.update(created_at=5), "created_at", True),
    ("source", "lr_policy", lambda d: d["source"].pop("dataset"), "source.dataset", True),
    (
        "predictor_type",
        "lr_policy",
        lambda d: d.update(predictor_type="knn"),
        "predictor_type",
        True,
    ),
    ("models_empty", "lr_policy", lambda d: d["models"].clear(), "models", True),
    (
        "model_name_duplicate",
        "lr_policy",
        lambda d: d["models"][1].update(name=d["models"][0]["name"]),
        "models[1].name",
        True,
    ),
    (
        "expected_output_negative",
        "lr_policy",
        lambda d: d["models"][0].update(expected_output_tokens=-1.0),
        "models[0].expected_output_tokens",
        True,
    ),
    (
        "expected_output_nan",
        "lr_policy",
        lambda d: d["models"][0].update(expected_output_tokens=float("nan")),
        "models[0].expected_output_tokens",
        False,
    ),
    ("strongest", "lr_policy", lambda d: d.update(strongest_model="nope"), "strongest_model", True),
    ("cheapest", "lr_policy", lambda d: d.update(cheapest_model="nope"), "cheapest_model", True),
    (
        "hash_dim",
        "lr_policy",
        lambda d: d["feature_spec"].update(hash_dim=1024),
        "feature_spec.hash_dim",
        True,
    ),
    (
        "ngram",
        "lr_policy",
        lambda d: d["feature_spec"].update(ngram=[1, 3]),
        "feature_spec.ngram",
        True,
    ),
    (
        "dense_names",
        "lr_policy",
        lambda d: d["feature_spec"]["dense"].reverse(),
        "feature_spec.dense",
        True,
    ),
    (
        "dense_mean_length",
        "lr_policy",
        lambda d: d["feature_spec"]["dense_mean"].pop(),
        "feature_spec.dense_mean",
        True,
    ),
    (
        "dense_std_zero",
        "lr_policy",
        lambda d: d["feature_spec"]["dense_std"].__setitem__(3, 0.0),
        "feature_spec.dense_std[3]",
        True,
    ),
    (
        "family_model_unknown",
        "lr_policy",
        lambda d: d["static_rules"]["family_model"].update(
            {d["static_rules"]["family_order"][0]: "nope"}
        ),
        "static_rules.family_model",
        True,
    ),
    (
        "default_family",
        "lr_policy",
        lambda d: d["static_rules"].update(default_family="nope"),
        "static_rules.default_family",
        True,
    ),
    ("blocks_empty", "lr_policy", _with_blocks(lambda b: b.clear()), "blocks", True),
    ("first_lo", "lr_policy", _with_blocks(lambda b: _set(b, 0, lo=0.1)), "blocks[0].lo", True),
    (
        "first_lo_half",
        "lr_policy",
        lambda d: d["models"][0]["calibration"]["blocks"][0].update(lo=0.5),
        "blocks[0].lo",
        True,
    ),
    ("last_hi", "lr_policy", _with_blocks(lambda b: _set(b, 1, hi=0.9)), "blocks[1].hi", True),
    ("gap", "lr_policy", _with_blocks(lambda b: _set(b, 0, hi=0.4)), "blocks[0].hi", True),
    (
        "lo_not_increasing",
        "lr_policy",
        _with_blocks(lambda b: (_set(b, 0, hi=0.0), _set(b, 1, lo=0.0))),
        "blocks[1].lo",
        True,
    ),
    (
        "p_decreasing",
        "lr_policy",
        _with_blocks(lambda b: _set(b, 1, k=10, p=0.2, lcb=0.1)),
        "blocks[1].p",
        True,
    ),
    ("n_zero", "lr_policy", _with_blocks(lambda b: _set(b, 0, n=0)), "blocks[0].n", True),
    ("k_above_n", "lr_policy", _with_blocks(lambda b: _set(b, 0, k=51)), "blocks[0].k", True),
    ("k_negative", "lr_policy", _with_blocks(lambda b: _set(b, 0, k=-1)), "blocks[0].k", True),
    (
        "p_not_k_over_n",
        "lr_policy",
        _with_blocks(lambda b: _set(b, 0, p=0.41)),
        "blocks[0].p",
        True,
    ),
    (
        "lcb_above_p",
        "lr_policy",
        _with_blocks(lambda b: _set(b, 0, lcb=0.5)),
        "blocks[0].lcb",
        True,
    ),
    (
        "lcb_negative",
        "lr_policy",
        _with_blocks(lambda b: _set(b, 0, lcb=-0.1)),
        "blocks[0].lcb",
        True,
    ),
    ("lcb_missing", "lr_policy", _with_blocks(lambda b: b[0].pop("lcb")), "blocks[0].lcb", True),
    (
        "profile_lambda_missing",
        "lr_policy",
        lambda d: d["profiles"][0].pop("lambda"),
        "profiles[0].lambda",
        True,
    ),
    (
        "profile_lambda_off_grid",
        "lr_policy",
        lambda d: d["profiles"][1].update({"lambda": 0.955}),
        "profiles[1].lambda",
        True,
    ),
    ("profiles_short", "lr_policy", lambda d: d["profiles"].pop(), "profiles", True),
    (
        "profile_order",
        "lr_policy",
        lambda d: d["profiles"].reverse(),
        "profiles[0].name",
        True,
    ),
    (
        "profile_alpha",
        "lr_policy",
        lambda d: d["profiles"][1].update(alpha=0.06),
        "profiles[1].alpha",
        True,
    ),
    (
        "profile_delta",
        "lr_policy",
        lambda d: d["profiles"][2].update(delta=0.1),
        "profiles[2].delta",
        True,
    ),
    (
        "profile_regret_above_n",
        "lr_policy",
        lambda d: d["profiles"][1].update(cert_regret=d["profiles"][1]["cert_n"] + 1),
        "profiles[1].cert_regret",
        True,
    ),
    (
        "profile_ucb",
        "lr_policy",
        lambda d: d["profiles"][1].update(regret_ucb=1.5),
        "profiles[1].regret_ucb",
        True,
    ),
    (
        "lr_indices_descending",
        "lr_policy",
        _swap_first_two_sparse_indices,
        "models[0].predictor.sparse_indices",
        True,
    ),
    (
        "lr_index_out_of_range",
        "lr_policy",
        lambda d: d["models"][0]["predictor"]["sparse_indices"].append(HASH_DIM),
        "models[0].predictor.sparse_indices",
        True,
    ),
    (
        "lr_weights_length",
        "lr_policy",
        lambda d: d["models"][0]["predictor"]["sparse_weights"].pop(),
        "models[0].predictor.sparse_weights",
        True,
    ),
    (
        "lr_dense_weights",
        "lr_policy",
        lambda d: d["models"][0]["predictor"]["dense_weights"].pop(),
        "models[0].predictor.dense_weights",
        True,
    ),
    (
        "gbm_dtype",
        "gbm_policy",
        lambda d: d["models"][0]["predictor"].update(feature_dtype="float64"),
        "models[0].predictor.feature_dtype",
        True,
    ),
    (
        "gbm_array_length",
        "gbm_policy",
        lambda d: _pop_first_tree(d, "value"),
        "models[0].predictor.trees[",
        True,
    ),
    (
        "gbm_child_out_of_range",
        "gbm_policy",
        lambda d: _set_first_tree(d, "left", 0, 10**6),
        "left",
        True,
    ),
    (
        "gbm_child_not_after_parent",
        "gbm_policy",
        lambda d: _set_first_tree(d, "right", 0, 0),
        "right",
        True,
    ),
    (
        "gbm_feature_out_of_range",
        "gbm_policy",
        lambda d: _set_first_tree(d, "feature", 0, HASH_DIM + len(DENSE_NAMES)),
        "feature",
        True,
    ),
    (
        "gbm_feature_negative_on_internal_node",
        "gbm_policy",
        lambda d: _set_first_tree(d, "feature", 0, -1),
        "feature",
        True,
    ),
]


@pytest.mark.parametrize(
    ("fixture", "mutate", "where", "revote"),
    [pytest.param(f, m, w, r, id=i) for i, f, m, w, r in INVALID_DOCUMENTS],
)
def test_validate_rejects_bad_documents(fixture, mutate, where, revote, request):
    doc = copy.deepcopy(policy_doc(request.getfixturevalue(fixture)))
    mutate(doc)
    if revote:
        doc["version"] = policy_version(doc)
    with pytest.raises(PolicyError, match=re.escape(where)):
        validate_policy_doc(doc)


def test_validate_accepts_a_valid_multi_block_calibration(lr_policy):
    doc = copy.deepcopy(policy_doc(lr_policy))
    _with_blocks(lambda blocks: None)(doc)
    doc["version"] = policy_version(doc)
    validate_policy_doc(doc)


def test_validate_rejects_a_document_that_is_not_an_object():
    with pytest.raises(PolicyError, match="document"):
        validate_policy_doc([])


def test_validate_rejects_missing_top_level_keys(lr_policy):
    doc = copy.deepcopy(policy_doc(lr_policy))
    del doc["profiles"]
    with pytest.raises(PolicyError, match="profiles"):
        validate_policy_doc(doc)


def test_profile_lambdas_in_exported_documents_are_grid_values_or_sentinel(lr_policy):
    for profile in policy_doc(lr_policy)["profiles"]:
        assert profile["lambda"] in (*LAMBDA_GRID, 1.01)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_load_policy_rejects_non_finite_numbers(literal, lr_policy, tmp_path):
    path = tmp_path / "policy.json"
    export_policy(lr_policy, path)
    text = path.read_text(encoding="utf-8")
    bad = re.sub(r'("expected_output_tokens":)[^,}]+', rf"\g<1>{literal}", text, count=1)
    assert bad != text
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(PolicyError, match=r"non-finite"):
        load_policy(path)


def test_load_policy_rejects_a_tampered_version(lr_policy, tmp_path):
    path = tmp_path / "policy.json"
    version, _ = export_policy(lr_policy, path)
    text = path.read_text(encoding="utf-8")
    assert text.count(version) == 1
    path.write_text(text.replace(version, "0" * 12), encoding="utf-8")
    with pytest.raises(PolicyError, match="version"):
        load_policy(path)


def test_load_policy_rejects_tampered_content_with_an_unchanged_version(lr_policy, tmp_path):
    path = tmp_path / "policy.json"
    export_policy(lr_policy, path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["strongest_model"] = lr_policy.cheapest_model
    path.write_bytes(canonical_bytes(doc) + b"\n")
    with pytest.raises(PolicyError, match="version"):
        load_policy(path)


def test_load_policy_wraps_unreadable_and_malformed_files(tmp_path):
    with pytest.raises(PolicyError, match="missing.json"):
        load_policy(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text('{"schema_version": ', encoding="utf-8")
    with pytest.raises(PolicyError, match="broken.json"):
        load_policy(broken)
    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(PolicyError, match="binary.json"):
        load_policy(binary)
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(PolicyError, match="array.json"):
        load_policy(array)


@pytest.mark.parametrize("fixture", POLICY_FIXTURES)
def test_score_policy_matches_the_per_model_predictors(fixture, request):
    policy: Policy = request.getfixturevalue(fixture)
    feats = featurize(["a", "Write code", ""])
    scores = score_policy(policy, feats)
    dense_z = policy.scaler.transform(feats.dense_raw)
    assert scores.raw.shape == scores.p.shape == scores.lcb.shape == scores.block.shape == (3, 3)
    for j, entry in enumerate(policy.models):
        raw = entry.predictor.raw(feats.hashed, dense_z)
        assert np.array_equal(scores.raw[:, j], raw)
        assert np.array_equal(scores.block[:, j], entry.calibration.lookup(raw))
        assert np.array_equal(scores.p[:, j], entry.calibration.p_of(raw))
        assert np.array_equal(scores.lcb[:, j], entry.calibration.lcb_of(raw))
    assert np.all((scores.lcb >= 0.0) & (scores.lcb <= scores.p) & (scores.p <= 1.0))
