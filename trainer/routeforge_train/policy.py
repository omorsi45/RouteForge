"""The served policy artifact: document, export, validation, loading and scoring.

`validate_policy_doc` is the contract the Go gateway must enforce when it loads an artifact;
every failure names the JSON path of the offending value. The file is described in
docs/architecture.md, "From the trainer to the gateway".
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from routeforge_train.calibration import Calibration
from routeforge_train.certify import DELTA, LAMBDA_GRID, PROFILES, SENTINEL_LAMBDA, ProfileCert
from routeforge_train.errors import TrainerError
from routeforge_train.features import DENSE_NAMES, HASH_DIM, DenseScaler, Features
from routeforge_train.gbm import GBMModel
from routeforge_train.jsonio import atomic_writer, canonical_bytes
from routeforge_train.lr import LRModel
from routeforge_train.rules import KeywordRules

SCHEMA_VERSION = 1
MAX_POLICY_BYTES = 5_000_000
PREDICTOR_TYPES = ("lr", "gbm")

_NGRAM = [1, 2]
_FEATURE_ID_LIMIT = HASH_DIM + len(DENSE_NAMES)
_VERSION = re.compile(r"[0-9a-f]{12}")
_UNVERSIONED = ("version", "created_at")
_TREE_KEYS = ("feature", "threshold", "left", "right", "value")
_TOP_LEVEL = (
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
)


class PolicyError(TrainerError):
    """A policy artifact is invalid, unreadable or cannot be exported."""


class PolicyTooLargeError(PolicyError):
    """The serialized policy exceeds `MAX_POLICY_BYTES`."""


@dataclass(frozen=True)
class ModelEntry:
    name: str
    predictor: LRModel | GBMModel
    calibration: Calibration
    expected_output_tokens: float


@dataclass(frozen=True)
class Policy:
    source: dict
    scaler: DenseScaler
    predictor_type: str
    models: tuple[ModelEntry, ...]
    strongest_model: str
    cheapest_model: str
    static_rules: KeywordRules
    profiles: tuple[ProfileCert, ...]
    created_at: str
    version: str = ""


@dataclass(frozen=True, eq=False)
class PolicyScores:
    """Per-model scores for a batch of prompts; every array has shape (n, M)."""

    raw: np.ndarray
    p: np.ndarray
    lcb: np.ndarray
    block: np.ndarray


def policy_version(doc: Mapping[str, Any]) -> str:
    """First 12 hex of the sha256 of the canonical document without version and created_at."""
    body = {k: v for k, v in doc.items() if k not in _UNVERSIONED}
    return hashlib.sha256(canonical_bytes(body)).hexdigest()[:12]


def policy_doc(policy: Policy) -> dict[str, Any]:
    """The full policy document, with `version` filled in."""
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": policy.created_at,
        "source": dict(policy.source),
        "feature_spec": {
            "hash_dim": HASH_DIM,
            "ngram": list(_NGRAM),
            "dense": list(DENSE_NAMES),
            "dense_mean": list(policy.scaler.mean),
            "dense_std": list(policy.scaler.std),
        },
        "predictor_type": policy.predictor_type,
        "models": [
            {
                "name": entry.name,
                "predictor": entry.predictor.to_json(),
                "calibration": entry.calibration.to_json(),
                "expected_output_tokens": float(entry.expected_output_tokens),
            }
            for entry in policy.models
        ],
        "strongest_model": policy.strongest_model,
        "cheapest_model": policy.cheapest_model,
        "static_rules": policy.static_rules.to_json(),
        "profiles": [profile.to_json() for profile in policy.profiles],
    }
    doc["version"] = policy_version(doc)
    return doc


def export_policy(policy: Policy, path: Path) -> tuple[str, int]:
    """Validate and atomically write the canonical document; returns (version, bytes)."""
    path = Path(path)
    doc = policy_doc(policy)
    validate_policy_doc(doc)
    data = canonical_bytes(doc) + b"\n"
    if len(data) > MAX_POLICY_BYTES:
        raise PolicyTooLargeError(
            f"{path}: policy is {len(data)} bytes, over the {MAX_POLICY_BYTES} byte limit"
        )
    with atomic_writer(path, "wb") as f:
        f.write(data)
    return doc["version"], len(data)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _fail(path: str, problem: str) -> PolicyError:
    return PolicyError(f"{path}: {problem}")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _mapping(value: object, path: str, keys: Sequence[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(path or "document", "must be a JSON object")
    for key in keys:
        if key not in value:
            raise _fail(_join(path, key), "is missing")
    return value


def _list(value: object, path: str, *, non_empty: bool = False) -> list:
    if not isinstance(value, list):
        raise _fail(path, "must be a list")
    if non_empty and not value:
        raise _fail(path, "must be a non-empty list")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(path, "must be a non-empty string")
    return value


def _number(value: object, path: str) -> float:
    if not _is_number(value):
        raise _fail(path, "must be a finite number")
    return float(value)  # type: ignore[arg-type]


def _integer(value: object, path: str) -> int:
    if not _is_int(value):
        raise _fail(path, "must be an integer")
    return value  # type: ignore[return-value]


def _numbers(value: object, path: str, count: int | None = None) -> list[float]:
    items = _list(value, path)
    if count is not None and len(items) != count:
        raise _fail(path, f"must have {count} entries, has {len(items)}")
    return [_number(v, f"{path}[{i}]") for i, v in enumerate(items)]


def validate_policy_doc(doc: Mapping[str, Any]) -> None:
    """Raise `PolicyError` naming the JSON path of the first violation."""
    root = _mapping(doc, "", ())
    if not _is_int(root.get("schema_version")) or root["schema_version"] != SCHEMA_VERSION:
        raise _fail("schema_version", f"must be {SCHEMA_VERSION}")
    _mapping(root, "", _TOP_LEVEL)
    version = root["version"]
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise _fail("version", "must be 12 lowercase hex characters")
    _text(root["created_at"], "created_at")
    source = _mapping(root["source"], "source", ("dataset", "revision", "git_sha"))
    for key in ("dataset", "revision", "git_sha"):
        _text(source[key], f"source.{key}")
    if root["predictor_type"] not in PREDICTOR_TYPES:
        raise _fail("predictor_type", f"must be one of {', '.join(PREDICTOR_TYPES)}")
    _validate_feature_spec(root["feature_spec"])
    names = _validate_models(root["models"], root["predictor_type"])
    for key in ("strongest_model", "cheapest_model"):
        if root[key] not in names:
            raise _fail(key, "is not a model of the policy")
    _validate_static_rules(root["static_rules"], names)
    _validate_profiles(root["profiles"])
    try:
        expected = policy_version(root)
    except (TypeError, ValueError) as exc:
        raise _fail("document", f"cannot be serialized as canonical JSON: {exc}") from exc
    if version != expected:
        raise _fail("version", f"does not match the document content ({expected})")


def _validate_feature_spec(spec: object) -> None:
    spec = _mapping(spec, "feature_spec", ("hash_dim", "ngram", "dense", "dense_mean", "dense_std"))
    if not _is_int(spec["hash_dim"]) or spec["hash_dim"] != HASH_DIM:
        raise _fail("feature_spec.hash_dim", f"must be {HASH_DIM}")
    ngram = spec["ngram"]
    if not isinstance(ngram, list) or not all(map(_is_int, ngram)) or ngram != _NGRAM:
        raise _fail("feature_spec.ngram", f"must be {_NGRAM}")
    if spec["dense"] != list(DENSE_NAMES):
        raise _fail("feature_spec.dense", f"must be {list(DENSE_NAMES)}")
    _numbers(spec["dense_mean"], "feature_spec.dense_mean", len(DENSE_NAMES))
    for i, std in enumerate(
        _numbers(spec["dense_std"], "feature_spec.dense_std", len(DENSE_NAMES))
    ):
        if std <= 0.0:
            raise _fail(f"feature_spec.dense_std[{i}]", "must be positive")


def _validate_models(models: object, predictor_type: str) -> set[str]:
    entries = _list(models, "models", non_empty=True)
    names: set[str] = set()
    for i, entry in enumerate(entries):
        path = f"models[{i}]"
        entry = _mapping(
            entry, path, ("name", "predictor", "calibration", "expected_output_tokens")
        )
        name = _text(entry["name"], f"{path}.name")
        if name in names:
            raise _fail(f"{path}.name", f"duplicates an earlier model {name!r}")
        names.add(name)
        if _number(entry["expected_output_tokens"], f"{path}.expected_output_tokens") < 0.0:
            raise _fail(f"{path}.expected_output_tokens", "must not be negative")
        if predictor_type == "lr":
            _validate_lr(entry["predictor"], f"{path}.predictor")
        else:
            _validate_gbm(entry["predictor"], f"{path}.predictor")
        _validate_calibration(entry["calibration"], f"{path}.calibration")
    return names


def _validate_lr(predictor: object, path: str) -> None:
    predictor = _mapping(
        predictor, path, ("intercept", "sparse_indices", "sparse_weights", "dense_weights")
    )
    _number(predictor["intercept"], f"{path}.intercept")
    indices = _list(predictor["sparse_indices"], f"{path}.sparse_indices")
    previous = -1
    for j, index in enumerate(indices):
        if not _is_int(index) or not previous < index < HASH_DIM:
            raise _fail(
                f"{path}.sparse_indices[{j}]",
                f"must be an integer in [0, {HASH_DIM}) and greater than the one before it",
            )
        previous = index
    weights = _numbers(predictor["sparse_weights"], f"{path}.sparse_weights")
    if len(weights) != len(indices):
        raise _fail(
            f"{path}.sparse_weights",
            f"has {len(weights)} entries, sparse_indices has {len(indices)}",
        )
    _numbers(predictor["dense_weights"], f"{path}.dense_weights", len(DENSE_NAMES))


def _validate_gbm(predictor: object, path: str) -> None:
    predictor = _mapping(predictor, path, ("init", "learning_rate", "feature_dtype", "trees"))
    _number(predictor["init"], f"{path}.init")
    _number(predictor["learning_rate"], f"{path}.learning_rate")
    if predictor["feature_dtype"] != "float32":
        raise _fail(f"{path}.feature_dtype", 'must be "float32"')
    for t, tree in enumerate(_list(predictor["trees"], f"{path}.trees")):
        _validate_tree(tree, f"{path}.trees[{t}]")


def _validate_tree(tree: object, path: str) -> None:
    tree = _mapping(tree, path, _TREE_KEYS)
    arrays = {key: _list(tree[key], _join(path, key)) for key in _TREE_KEYS}
    size = len(arrays["feature"])
    if size == 0:
        raise _fail(_join(path, "feature"), "must not be empty")
    for key, array in arrays.items():
        if len(array) != size:
            raise _fail(path, f"arrays differ in length: {key} has {len(array)}, feature {size}")
    for node in range(size):
        _number(arrays["threshold"][node], f"{path}.threshold[{node}]")
        _number(arrays["value"][node], f"{path}.value[{node}]")
        left, right = arrays["left"][node], arrays["right"][node]
        if not _is_int(left):
            raise _fail(f"{path}.left[{node}]", "must be an integer")
        if not _is_int(right):
            raise _fail(f"{path}.right[{node}]", "must be an integer")
        if left == -1:
            if right != -1:
                raise _fail(f"{path}.right[{node}]", "must be -1 on a leaf")
            continue
        feature = arrays["feature"][node]
        if not _is_int(feature) or not 0 <= feature < _FEATURE_ID_LIMIT:
            raise _fail(
                f"{path}.feature[{node}]", f"must be an integer in [0, {_FEATURE_ID_LIMIT})"
            )
        for side, child in (("left", left), ("right", right)):
            if not node < child < size:
                raise _fail(f"{path}.{side}[{node}]", f"must be -1 or in ({node}, {size})")


def _validate_calibration(calibration: object, path: str) -> None:
    calibration = _mapping(calibration, path, ("blocks",))
    blocks = _list(calibration["blocks"], f"{path}.blocks", non_empty=True)
    previous: Mapping[str, Any] | None = None
    for i, block in enumerate(blocks):
        bp = f"{path}.blocks[{i}]"
        block = _mapping(block, bp, ("lo", "hi", "n", "k", "p", "lcb"))
        lo, hi = _number(block["lo"], f"{bp}.lo"), _number(block["hi"], f"{bp}.hi")
        n, k = _integer(block["n"], f"{bp}.n"), _integer(block["k"], f"{bp}.k")
        p, lcb = _number(block["p"], f"{bp}.p"), _number(block["lcb"], f"{bp}.lcb")
        if previous is None:
            if lo != 0.0:
                raise _fail(f"{bp}.lo", "must be 0.0 for the first block")
        else:
            if lo <= previous["lo"]:
                raise _fail(f"{bp}.lo", "must be greater than the previous block's lo")
            if previous["hi"] != lo:
                raise _fail(f"{path}.blocks[{i - 1}].hi", f"must equal blocks[{i}].lo")
        if hi < lo:
            raise _fail(f"{bp}.hi", "must not be below lo")
        if n < 1:
            raise _fail(f"{bp}.n", "must be at least 1")
        if not 0 <= k <= n:
            raise _fail(f"{bp}.k", f"must be in [0, n] with n = {n}")
        if p != k / n:
            raise _fail(f"{bp}.p", f"must equal k / n ({k / n})")
        if not 0.0 <= lcb <= p:
            raise _fail(f"{bp}.lcb", f"must be in [0, p] with p = {p}")
        if previous is not None and p < previous["p"]:
            raise _fail(f"{bp}.p", "must not be below the previous block's p")
        previous = {"lo": lo, "hi": hi, "p": p}
    if previous is not None and previous["hi"] != 1.0:
        raise _fail(f"{path}.blocks[{len(blocks) - 1}].hi", "must be 1.0 for the last block")


def _validate_static_rules(rules: object, models: set[str]) -> None:
    rules = _mapping(
        rules,
        "static_rules",
        ("family_keywords", "family_model", "family_order", "default_family"),
    )
    order = _list(rules["family_order"], "static_rules.family_order", non_empty=True)
    for i, family in enumerate(order):
        _text(family, f"static_rules.family_order[{i}]")
    keywords = _mapping(rules["family_keywords"], "static_rules.family_keywords", ())
    family_model = _mapping(rules["family_model"], "static_rules.family_model", ())
    for key, table in (("family_keywords", keywords), ("family_model", family_model)):
        if set(table) != set(order):
            raise _fail(f"static_rules.{key}", "must have exactly the families of family_order")
    for family, words in keywords.items():
        for j, word in enumerate(_list(words, f"static_rules.family_keywords.{family}")):
            _text(word, f"static_rules.family_keywords.{family}[{j}]")
    for family, model in family_model.items():
        if model not in models:
            raise _fail(f"static_rules.family_model.{family}", "is not a model of the policy")
    if rules["default_family"] not in order:
        raise _fail("static_rules.default_family", "is not in family_order")


def _validate_profiles(profiles: object) -> None:
    items = _list(profiles, "profiles")
    if len(items) != len(PROFILES):
        raise _fail("profiles", f"must list exactly {', '.join(name for name, _ in PROFILES)}")
    for i, (profile, (name, alpha)) in enumerate(zip(items, PROFILES, strict=True)):
        path = f"profiles[{i}]"
        profile = _mapping(
            profile,
            path,
            ("name", "alpha", "delta", "lambda", "cert_n", "cert_regret", "regret_ucb"),
        )
        if profile["name"] != name:
            raise _fail(f"{path}.name", f"must be {name!r}")
        if _number(profile["alpha"], f"{path}.alpha") != alpha:
            raise _fail(f"{path}.alpha", f"must be {alpha}")
        if _number(profile["delta"], f"{path}.delta") != DELTA:
            raise _fail(f"{path}.delta", f"must be {DELTA}")
        lam = _number(profile["lambda"], f"{path}.lambda")
        if lam != SENTINEL_LAMBDA and lam not in LAMBDA_GRID:
            raise _fail(f"{path}.lambda", f"must be on the 0.00 to 1.00 grid or {SENTINEL_LAMBDA}")
        cert_n = _integer(profile["cert_n"], f"{path}.cert_n")
        cert_regret = _integer(profile["cert_regret"], f"{path}.cert_regret")
        if not 0 <= cert_regret <= cert_n:
            raise _fail(f"{path}.cert_regret", f"must be in [0, cert_n] with cert_n = {cert_n}")
        if not 0.0 <= _number(profile["regret_ucb"], f"{path}.regret_ucb") <= 1.0:
            raise _fail(f"{path}.regret_ucb", "must be in [0, 1]")


def _reject_constant(name: str) -> NoReturn:
    raise PolicyError(f"non-finite number {name}")


def _parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise PolicyError(f"non-finite number {text}")
    return value


def load_policy(path: Path) -> Policy:
    """Read, validate and rebuild a policy; every problem is a `PolicyError` naming the file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PolicyError(f"{path}: cannot read policy: {exc}") from exc
    try:
        doc = json.loads(text, parse_constant=_reject_constant, parse_float=_parse_float)
        validate_policy_doc(doc)
    except PolicyError as exc:
        raise PolicyError(f"{path}: {exc}") from exc
    except ValueError as exc:
        raise PolicyError(f"{path}: invalid JSON: {exc}") from exc
    return _policy_from_doc(doc)


def _policy_from_doc(doc: Mapping[str, Any]) -> Policy:
    predictor_cls = LRModel if doc["predictor_type"] == "lr" else GBMModel
    spec = doc["feature_spec"]
    return Policy(
        source=dict(doc["source"]),
        scaler=DenseScaler(
            mean=tuple(float(v) for v in spec["dense_mean"]),
            std=tuple(float(v) for v in spec["dense_std"]),
        ),
        predictor_type=doc["predictor_type"],
        models=tuple(
            ModelEntry(
                name=entry["name"],
                predictor=predictor_cls.from_json(entry["predictor"]),
                calibration=Calibration.from_json(entry["calibration"]),
                expected_output_tokens=float(entry["expected_output_tokens"]),
            )
            for entry in doc["models"]
        ),
        strongest_model=doc["strongest_model"],
        cheapest_model=doc["cheapest_model"],
        static_rules=KeywordRules.from_json(doc["static_rules"]),
        profiles=tuple(ProfileCert.from_json(p) for p in doc["profiles"]),
        created_at=doc["created_at"],
        version=doc["version"],
    )


def score_policy(policy: Policy, features: Features) -> PolicyScores:
    """Raw score, calibrated p, Wilson lcb and block index of every model for every prompt."""
    dense_z = policy.scaler.transform(features.dense_raw)
    n, m = features.hashed.shape[0], len(policy.models)
    raw, p, lcb = (np.empty((n, m), dtype=np.float64) for _ in range(3))
    block = np.empty((n, m), dtype=np.int64)
    for j, entry in enumerate(policy.models):
        scores = entry.predictor.raw(features.hashed, dense_z)
        raw[:, j] = scores
        p[:, j] = entry.calibration.p_of(scores)
        lcb[:, j] = entry.calibration.lcb_of(scores)
        block[:, j] = entry.calibration.lookup(scores)
    return PolicyScores(raw=raw, p=p, lcb=lcb, block=block)
