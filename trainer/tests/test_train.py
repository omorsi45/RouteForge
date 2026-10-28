import copy
import json
import re
from pathlib import Path

import pytest

from routeforge_train import splits
from routeforge_train.certify import LAMBDA_GRID, SENTINEL_LAMBDA
from routeforge_train.cli import main
from routeforge_train.errors import TrainerError
from routeforge_train.features import HASH_DIM, VOCAB_MAX, VOCAB_MIN_DF
from routeforge_train.gbm import GBM_PARAMS
from routeforge_train.jsonio import sha256_file
from routeforge_train.knn import K
from routeforge_train.parity import PARITY_CASES
from routeforge_train.policy import load_policy, validate_policy_doc
from routeforge_train.report_md import (
    COST_NOTE,
    EXTRACTION_NOTE,
    GUARANTEE_NOTE,
    LATENCY_NOTE,
    TEMPLATE_NOTE,
    fmt_pct,
    fmt_rate,
    fmt_usd,
    render_training_markdown,
)
from routeforge_train.train import run_train
from tests.synth import FIXED_NOW, SYNTH_POOL, build_synthetic_project

SMALL = 100
REPO_ROOT = Path(__file__).resolve().parents[2]

REPORT_KEYS = {
    "schema_version",
    "created_at",
    "git_sha",
    "policy_version",
    "policy_bytes",
    "source",
    "environment",
    "data",
    "features",
    "strongest_model",
    "cheapest_model",
    "expected_output_tokens",
    "lr",
    "gbm",
    "knn",
    "calibration",
    "static_rules",
    "selection",
    "certification",
    "costs",
    "notes",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_note_copy_is_exact():
    assert GUARANTEE_NOTE == (
        "With probability at least 1 - delta over the draw of the certification set, the "
        "expected regret rate on future traffic exchangeable with it is at most alpha, when "
        "all pool models are eligible. It says nothing about any single request."
    )
    assert COST_NOTE == (
        "Offline costs are the dataset-recorded cost field (provider usage times list price "
        "per LLMRouterBench). The gateway price table (deploy/prices.yaml) uses list prices. "
        "Files whose recorded total differs from the list-price recomputation by more than 1% "
        "are listed below."
    )
    assert TEMPLATE_NOTE == (
        "Benchmark prompt templates let a classifier recognize the dataset, which inflates "
        "in-distribution gains; leave-one-dataset-out results are reported next to them."
    )
    assert LATENCY_NOTE == (
        "The latency-aware router uses SIMULATED latency profiles "
        "(bench/latency_profiles.yaml) and is a reference only, not part of the H1 test."
    )
    assert EXTRACTION_NOTE == (
        "Extraction quality is unmeasured: make label-extraction has not been run."
    )


def test_number_formatters():
    assert fmt_usd(1234.5) == "$1,234.5000"
    assert fmt_usd(0.0) == "$0.0000"
    assert fmt_rate(0.12345) == "0.123"
    assert fmt_pct(0.5) == "50.0%"
    assert fmt_pct(0.01234) == "1.2%"


def test_train_writes_policy_reports_parity_and_limits(trained_paths, training):
    paths = trained_paths
    manifest = read_json(paths.manifest)
    for path in (
        paths.policy,
        paths.reports / "training.json",
        paths.reports / "training.md",
        paths.parity_fixtures,
        paths.bench_limits,
        *(paths.candidates_dir / f"{kind}.json" for kind in ("lr", "gbm", "knn")),
    ):
        assert path.is_file(), path
    assert len(paths.parity_fixtures.read_text(encoding="utf-8").splitlines()) == len(PARITY_CASES)
    policy = load_policy(paths.policy)
    limits = read_json(paths.bench_limits)
    assert limits == {
        "policy_version": policy.version,
        "source": manifest["source"],
        "max_output_tokens": manifest["max_completion_tokens"]["overall"],
        "max_output_tokens_by_model": manifest["max_completion_tokens"]["by_model"],
        "expected_output_tokens": {m.name: m.expected_output_tokens for m in policy.models},
    }
    assert training["policy_version"] == policy.version
    assert training["policy_bytes"] == paths.policy.stat().st_size < 5_000_000


def test_training_report_has_the_documented_keys_and_values(trained_paths, training):
    manifest = read_json(trained_paths.manifest)
    assert set(training) == REPORT_KEYS
    assert training["schema_version"] == 1
    assert training["created_at"] == "2026-10-26T00:00:00Z" and training["git_sha"] == "synthetic"
    assert training["source"] == manifest["source"] and training["costs"] == manifest["costs"]
    assert set(training["environment"]) >= {"python", "numpy", "scipy", "sklearn"}
    assert training["data"] == {
        "pool": list(SYNTH_POOL),
        "split_counts": manifest["splits"]["counts"],
        "datasets": manifest["datasets"],
        "manifest_sha256": sha256_file(trained_paths.manifest),
    }
    features = training["features"]
    assert features["hash_dim"] == HASH_DIM and features["vocab_min_df"] == VOCAB_MIN_DF == 5
    assert features["vocab_max"] == VOCAB_MAX == 20000
    assert features["lr_weight_sig_digits"] == 6 and 0 < features["vocab_size"] <= VOCAB_MAX
    assert training["strongest_model"] == "m-strong" and training["cheapest_model"] == "m-cheap"
    assert set(training["expected_output_tokens"]) == set(SYNTH_POOL)
    assert set(training["lr"]) == {"c_by_model", "cv_log_loss_by_model"}
    assert set(training["lr"]["c_by_model"]) == set(SYNTH_POOL)
    assert len(training["lr"]["cv_log_loss_by_model"]["m-mid"]) == 8
    assert training["gbm"] == {"params": GBM_PARAMS} and training["knn"] == {"k": K}
    assert {k: set(v) for k, v in training["calibration"].items()} == {
        "lr": set(SYNTH_POOL),
        "gbm": set(SYNTH_POOL),
        "knn": set(SYNTH_POOL),
    }
    rules = training["static_rules"]
    assert set(rules) == {"family_model", "family_order", "default_family", "keyword_counts"}
    assert set(rules["keyword_counts"]) == set(rules["family_order"])
    selection = training["selection"]
    assert set(selection) == {"winner", "profile", "calibration_n", "candidates"}
    assert selection["profile"] == "balanced"
    assert selection["calibration_n"] == manifest["splits"]["counts"]["calibration"]
    assert set(training["certification"]) == {"served", "offline"}
    assert training["notes"][0] == GUARANTEE_NOTE


def test_train_profiles_and_sentinel(trained_paths, training):
    served = {p["name"]: p for p in training["certification"]["served"]}
    cert_n = read_json(trained_paths.manifest)["splits"]["counts"]["certification"]
    assert cert_n == 88 and all(p["cert_n"] == 88 for p in served.values())
    assert served["strict"]["lambda"] == SENTINEL_LAMBDA == 1.01
    assert served["strict"]["cert_regret"] == 0
    assert served["balanced"]["lambda"] in LAMBDA_GRID
    assert served["economy"]["lambda"] in LAMBDA_GRID
    assert training["notes"][1:] == [
        "strict: lambda 1.01 sentinel, always strongest (certification n below the minimum)"
    ]


def test_candidate_files_are_valid_policies_and_a_knn_summary(trained_paths, training):
    winner = training["selection"]["winner"]
    certification = training["certification"]
    for kind in ("lr", "gbm"):
        doc = read_json(trained_paths.candidates_dir / f"{kind}.json")
        validate_policy_doc(doc)
        assert doc["predictor_type"] == kind
        want = certification["served"] if kind == winner else certification["offline"][kind]
        assert doc["profiles"] == want
    served_doc = read_json(trained_paths.policy)
    assert served_doc == read_json(trained_paths.candidates_dir / f"{winner}.json")
    knn = read_json(trained_paths.candidates_dir / "knn.json")
    assert set(knn) == {"kind", "k", "calibrations", "profiles"}
    assert knn["kind"] == "knn" and knn["k"] == 25
    assert [c["model"] for c in knn["calibrations"]] == list(SYNTH_POOL)
    assert all(c["blocks"] for c in knn["calibrations"])
    assert knn["profiles"] == certification["offline"]["knn"]


def test_training_markdown_has_sections_and_no_em_dash(trained_paths, training):
    text = (trained_paths.reports / "training.md").read_text(encoding="utf-8")
    assert text.startswith("# RouteForge training report\n")
    headings = re.findall(r"^## .*$", text, flags=re.MULTILINE)
    assert headings == [
        "## Provenance",
        "## Data",
        "## Predictors",
        "## Selection (calibration split, balanced)",
        "## Certification",
        "## Costs",
        "## Guarantee",
    ]
    assert GUARANTEE_NOTE in text and COST_NOTE in text
    assert "\u2014" not in text
    assert training["policy_version"] in text and "synthetic" in text
    for dataset in ("alpha", "beta", "gamma", "delta"):
        assert dataset in text
    winner = training["selection"]["winner"]
    assert f"Winner: {winner}" in text
    assert "strict: lambda 1.01 sentinel, always strongest" in text
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_markdown_is_the_render_of_the_committed_json(trained_paths):
    report = read_json(trained_paths.reports / "training.json")
    text = (trained_paths.reports / "training.md").read_text(encoding="utf-8")
    assert render_training_markdown(report) == text


def test_markdown_lists_only_cost_rows_that_differ_by_more_than_one_percent(training):
    report = copy.deepcopy(training)
    cells = [
        ("zz-high", 1.25),
        ("zz-low", 0.5),
        ("zz-near", 1.005),
        ("zz-exact", 1.0),
        ("zz-none", None),
    ]
    report["costs"]["by_dataset_model"] = [
        {
            "dataset": name,
            "model": "m-mid",
            "recorded_usd": 1.0,
            "list_usd": 1.0,
            "ratio": ratio,
        }
        for name, ratio in cells
    ]
    text = render_training_markdown(report)
    assert "zz-high" in text and "zz-low" in text
    assert not any(name in text for name in ("zz-near", "zz-exact", "zz-none"))
    assert "1.250" in text and "0.500" in text


def test_train_never_reads_test_split(tmp_path, monkeypatch):
    paths = build_synthetic_project(tmp_path, n_per_dataset=SMALL)
    splits.split_path(paths.splits_dir, "test").unlink()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the test split was read")

    monkeypatch.setattr(splits, "read_test_split", forbidden)
    read = []
    real_read_split = splits.read_split

    def recording_read_split(splits_dir, name):
        read.append(name)
        return real_read_split(splits_dir, name)

    monkeypatch.setattr(splits, "read_split", recording_read_split)
    report = run_train(paths, git_sha="synthetic", now=FIXED_NOW)
    assert sorted(read) == ["calibration", "certification", "train"]
    assert report["policy_version"] == load_policy(paths.policy).version


def test_train_is_deterministic(tmp_path, trained_paths, training):
    again = build_synthetic_project(tmp_path)
    report = run_train(again, git_sha="synthetic", now=FIXED_NOW)
    assert report == training
    assert report["policy_version"] == training["policy_version"]
    assert again.policy.read_bytes() == trained_paths.policy.read_bytes()
    assert again.parity_fixtures.read_bytes() == trained_paths.parity_fixtures.read_bytes()
    assert again.bench_limits.read_bytes() == trained_paths.bench_limits.read_bytes()
    assert (again.reports / "training.md").read_bytes() == (
        trained_paths.reports / "training.md"
    ).read_bytes()


@pytest.mark.parametrize("name", ["train", "calibration", "certification"])
def test_train_rejects_tampered_split(name, tmp_path):
    paths = build_synthetic_project(tmp_path, n_per_dataset=SMALL)
    with open(splits.split_path(paths.splits_dir, name), "ab") as f:
        f.write(b"\n")
    with pytest.raises(TrainerError, match=name):
        run_train(paths, git_sha="synthetic", now=FIXED_NOW)
    assert not paths.policy.exists() and not paths.bench_limits.exists()
    assert not paths.parity_fixtures.exists() and not paths.reports.exists()


def test_train_requires_the_manifest(tmp_path):
    paths = build_synthetic_project(tmp_path, n_per_dataset=SMALL)
    paths.manifest.unlink()
    with pytest.raises(TrainerError, match="manifest"):
        run_train(paths, git_sha="synthetic", now=FIXED_NOW)


@pytest.mark.parametrize("content", ["not json", "[]", "{}"])
def test_train_rejects_an_unusable_manifest(content, tmp_path):
    paths = build_synthetic_project(tmp_path, n_per_dataset=SMALL)
    paths.manifest.write_text(content, encoding="utf-8")
    with pytest.raises(TrainerError, match="manifest"):
        run_train(paths, git_sha="synthetic", now=FIXED_NOW)
    assert not paths.policy.exists()


def test_train_reports_a_missing_split_file(tmp_path):
    paths = build_synthetic_project(tmp_path, n_per_dataset=SMALL)
    splits.split_path(paths.splits_dir, "train").unlink()
    with pytest.raises(TrainerError, match="train"):
        run_train(paths, git_sha="synthetic", now=FIXED_NOW)


def test_make_train_target_runs_the_train_command():
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    phony = next(line for line in makefile.splitlines() if line.startswith(".PHONY:"))
    assert "train" in phony.split()
    assert "\ntrain:\n\t$(TRAIN) train\n" in makefile


def test_makefile_pins_blas_threads():
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    lines = makefile.splitlines()
    train_line = next(line for line in lines if line.startswith("TRAIN :="))
    py_test_recipe = lines[lines.index("py-test:") + 1]
    for name, line in (("TRAIN", train_line), ("py-test", py_test_recipe)):
        words = line.split()
        for pin in ("OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1"):
            assert pin in words, f"{name} does not set {pin}"


def test_cli_train_prints_summary(tmp_path, capsys):
    paths = build_synthetic_project(tmp_path, n_per_dataset=SMALL)
    assert main(["--root", str(paths.root), "train"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("train OK: policy ")
    assert re.fullmatch(
        r"train OK: policy [0-9a-f]{12} \((lr|gbm)\), \d+ bytes; "
        r"lambda strict=\d\.\d\d balanced=\d\.\d\d economy=\d\.\d\d\n",
        out,
    )
    assert f" {paths.policy.stat().st_size} bytes;" in out


def test_cli_train_reports_trainer_errors(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "train"]) == 1
    assert "manifest" in capsys.readouterr().err
