"""The `train` command: fit, select, certify, export and report."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from routeforge_train.certify import SENTINEL_LAMBDA
from routeforge_train.errors import TrainerError
from routeforge_train.features import HASH_DIM, VOCAB_MAX, VOCAB_MIN_DF
from routeforge_train.gbm import GBM_PARAMS
from routeforge_train.jsonio import atomic_writer, canonical_bytes, sha256_file, write_json
from routeforge_train.knn import K
from routeforge_train.lr import WEIGHT_SIG_DIGITS
from routeforge_train.parity import write_parity_fixtures
from routeforge_train.paths import Paths
from routeforge_train.pipeline import (
    Fitted,
    Selection,
    certify,
    fit_pipeline,
    select_served,
    to_policy,
)
from routeforge_train.policy import PREDICTOR_TYPES, export_policy, policy_doc
from routeforge_train.prices import load_prices
from routeforge_train.provenance import current_git_sha, environment, utc_iso
from routeforge_train.report_md import GUARANTEE_NOTE, render_training_markdown
from routeforge_train.splits import read_verified_split
from routeforge_train.table import OutcomeTable, pivot

SCHEMA_VERSION = 1
_MANIFEST_KEYS = (
    "pool",
    "files",
    "source",
    "splits",
    "datasets",
    "costs",
    "max_completion_tokens",
)


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrainerError(
            f"{path}: cannot read the data manifest (run the data command first): {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise TrainerError(f"{path}: the data manifest must be a JSON object")
    missing = [key for key in _MANIFEST_KEYS if key not in manifest]
    if missing:
        raise TrainerError(f"{path}: the data manifest lacks {', '.join(missing)}")
    return manifest


def load_table(paths: Paths, manifest: dict[str, Any], name: str) -> OutcomeTable:
    try:
        return pivot(read_verified_split(paths, manifest, name), manifest["pool"])
    except ValueError as exc:
        raise TrainerError(f"cannot load the {name} split: {exc}") from exc


def _write_canonical(path: Path, doc: Any) -> None:
    with atomic_writer(path, "wb") as f:
        f.write(canonical_bytes(doc) + b"\n")


def _selection_json(selection: Selection) -> dict[str, Any]:
    return {
        "winner": selection.winner,
        "profile": selection.profile,
        "calibration_n": selection.calibration_n,
        "candidates": selection.candidates,
    }


def _training_report(
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
    fitted: Fitted,
    selection: Selection,
    served: list[dict[str, Any]],
    offline: dict[str, list[dict[str, Any]]],
    policy_version: str,
    policy_bytes: int,
    git_sha: str,
    created_at: str,
) -> dict[str, Any]:
    rules = fitted.rules
    notes = [GUARANTEE_NOTE] + [
        f"{profile['name']}: lambda {SENTINEL_LAMBDA} sentinel, always strongest "
        "(certification n below the minimum)"
        for profile in served
        if profile["lambda"] == SENTINEL_LAMBDA
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "git_sha": git_sha,
        "policy_version": policy_version,
        "policy_bytes": policy_bytes,
        "source": manifest["source"],
        "environment": environment(),
        "data": {
            "pool": list(fitted.pool),
            "split_counts": manifest["splits"]["counts"],
            "datasets": manifest["datasets"],
            "manifest_sha256": manifest_sha256,
        },
        "features": {
            "hash_dim": HASH_DIM,
            "vocab_size": int(len(fitted.vocab)),
            "vocab_min_df": VOCAB_MIN_DF,
            "vocab_max": VOCAB_MAX,
            "lr_weight_sig_digits": WEIGHT_SIG_DIGITS,
        },
        "strongest_model": fitted.pool[fitted.strongest],
        "cheapest_model": fitted.pool[fitted.cheapest],
        "expected_output_tokens": fitted.expected_output_tokens,
        "lr": {
            "c_by_model": fitted.lr_c,
            "cv_log_loss_by_model": {
                model: {str(c): loss for c, loss in losses.items()}
                for model, losses in fitted.lr_cv.items()
            },
        },
        "gbm": {"params": GBM_PARAMS},
        "knn": {"k": K},
        "calibration": {
            kind: {
                model: len(calibration.blocks)
                for model, calibration in zip(fitted.pool, candidate.calibrations, strict=True)
            }
            for kind, candidate in fitted.candidates.items()
        },
        "static_rules": {
            "family_model": rules.family_model,
            "family_order": list(rules.family_order),
            "default_family": rules.default_family,
            "keyword_counts": {f: len(words) for f, words in rules.family_keywords.items()},
        },
        "selection": _selection_json(selection),
        "certification": {"served": served, "offline": offline},
        "costs": manifest["costs"],
        "notes": notes,
    }


def run_train(
    paths: Paths, *, git_sha: str | None = None, now: datetime | None = None
) -> dict[str, Any]:
    """Fit, select, certify and export the policy; returns the training report.

    Only the train, calibration and certification splits are read, each after a checksum
    check against the manifest. The test split is never opened.
    """
    now = datetime.now(UTC) if now is None else now
    git_sha = current_git_sha(paths.root) if git_sha is None else git_sha
    created_at = utc_iso(now)
    manifest = read_manifest(paths.manifest)
    prices = load_prices(paths.prices)
    train = load_table(paths, manifest, "train")
    calibration = load_table(paths, manifest, "calibration")
    certification = load_table(paths, manifest, "certification")

    fitted = fit_pipeline(train, calibration, prices)
    selection = select_served(fitted, calibration, prices)
    winner = selection.winner
    served = certify(fitted, winner, certification, prices)
    # Offline comparison only: certified after the winner is fixed and never exported.
    offline = {
        kind: certify(fitted, kind, certification, prices)
        for kind in fitted.candidates
        if kind != winner
    }

    source = {
        "dataset": manifest["source"]["dataset"],
        "revision": manifest["source"]["revision"],
        "git_sha": git_sha,
    }
    version, policy_bytes = export_policy(
        to_policy(fitted, winner, served, source, created_at), paths.policy
    )
    for kind in PREDICTOR_TYPES:
        profiles = served if kind == winner else offline[kind]
        doc = policy_doc(to_policy(fitted, kind, profiles, source, created_at))
        _write_canonical(paths.candidates_dir / f"{kind}.json", doc)
    knn = fitted.candidates["knn"]
    _write_canonical(
        paths.candidates_dir / "knn.json",
        {
            "kind": "knn",
            "k": K,
            "calibrations": [
                {"model": model, **model_calibration.to_json()}
                for model, model_calibration in zip(fitted.pool, knn.calibrations, strict=True)
            ],
            "profiles": [profile.to_json() for profile in offline["knn"]],
        },
    )
    write_parity_fixtures(paths.policy, paths.prices, paths.parity_fixtures)

    write_json(
        paths.bench_limits,
        {
            "policy_version": version,
            "source": manifest["source"],
            "max_output_tokens": manifest["max_completion_tokens"]["overall"],
            "max_output_tokens_by_model": manifest["max_completion_tokens"]["by_model"],
            "expected_output_tokens": fitted.expected_output_tokens,
        },
    )

    report = _training_report(
        manifest=manifest,
        manifest_sha256=sha256_file(paths.manifest),
        fitted=fitted,
        selection=selection,
        served=[profile.to_json() for profile in served],
        offline={kind: [p.to_json() for p in profiles] for kind, profiles in offline.items()},
        policy_version=version,
        policy_bytes=policy_bytes,
        git_sha=git_sha,
        created_at=created_at,
    )
    # Render from the JSON round trip, the same input a later re-render from the file sees.
    report = json.loads(json.dumps(report, allow_nan=False))
    write_json(paths.reports / "training.json", report)
    with atomic_writer(paths.reports / "training.md") as f:
        f.write(render_training_markdown(report))
    return report


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "train", help="fit, select, certify and export the routing policy"
    )
    parser.set_defaults(func=_run)


def _run(paths: Paths, _args: argparse.Namespace) -> int:
    report = run_train(paths)
    lambdas = {profile["name"]: profile["lambda"] for profile in report["certification"]["served"]}
    print(
        f"train OK: policy {report['policy_version']} ({report['selection']['winner']}), "
        f"{report['policy_bytes']} bytes; lambda strict={lambdas['strict']:.2f} "
        f"balanced={lambdas['balanced']:.2f} economy={lambdas['economy']:.2f}"
    )
    return 0
