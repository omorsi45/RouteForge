import json
import os
import subprocess
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from routeforge_train.features import prompt_sha256
from routeforge_train.jsonio import sha256_file
from routeforge_train.outcomes import OutcomeRow, read_outcomes
from routeforge_train.prepare import apply_coverage
from routeforge_train.provenance import current_git_sha, environment, git_executable, utc_iso
from routeforge_train.splits import SPLIT_NAMES, split_path
from routeforge_train.table import pivot
from tests.synth import (
    SYNTH_DATASETS,
    SYNTH_POOL,
    build_synthetic_project,
    prepare_synthetic,
    synthetic_rows,
    write_synthetic_inputs,
)

POOL = ("m1", "m2")


def row(pid, model, prompt="p", dataset=None, success=True, **extra):
    dataset = dataset if dataset is not None else pid.split(":")[0]
    fields = {
        "prompt_id": pid,
        "dataset": dataset,
        "family": "qa",
        "prompt": prompt,
        "model": model,
        "score": 1.0 if success else 0.0,
        "success": success,
        "prompt_tokens": 4,
        "completion_tokens": 8,
        "cost_usd": 0.5,
    }
    return OutcomeRow(**(fields | extra))


def full(pid, prompt="p"):
    return [row(pid, m, prompt) for m in POOL]


def test_coverage_keeps_complete_prompts_in_input_order():
    rows = full("a:1") + full("a:2")
    report = apply_coverage(rows, POOL)
    assert report.kept == rows
    assert report.datasets == {
        "a": {
            "family": "qa",
            "prompts_in": 2,
            "prompts_kept": 2,
            "dropped": {"missing_model": 0, "duplicate_row": 0, "prompt_mismatch": 0},
        }
    }
    assert report.dropped_datasets == {} and report.non_pool_rows == 0
    assert report.dropped_prompts == []


def test_coverage_drops_prompt_missing_a_model():
    rows = full("a:1") + full("a:2") + [row("a:3", "m1")]
    report = apply_coverage(rows, POOL)
    assert report.datasets["a"]["dropped"]["missing_model"] == 1
    assert report.datasets["a"]["prompts_kept"] == report.datasets["a"]["prompts_in"] - 1 == 2
    assert {r.prompt_id for r in report.kept} == {"a:1", "a:2"}
    assert report.dropped_prompts == [
        {"dataset": "a", "prompt_id": "a:3", "reason": "missing_model"}
    ]


def test_coverage_drops_prompt_mismatch():
    rows = full("a:1") + [row("a:2", "m1", "x"), row("a:2", "m2", "y")]
    report = apply_coverage(rows, POOL)
    assert report.datasets["a"]["dropped"]["prompt_mismatch"] == 1
    assert {r.prompt_id for r in report.kept} == {"a:1"}
    assert report.dropped_prompts == [
        {"dataset": "a", "prompt_id": "a:2", "reason": "prompt_mismatch"}
    ]


def test_coverage_drops_duplicate_rows():
    rows = full("a:1") + full("a:2") + [row("a:2", "m1")]
    report = apply_coverage(rows, POOL)
    assert report.datasets["a"]["dropped"]["duplicate_row"] == 1
    assert {r.prompt_id for r in report.kept} == {"a:1"}
    assert report.dropped_prompts == [
        {"dataset": "a", "prompt_id": "a:2", "reason": "duplicate_row"}
    ]


def test_coverage_counts_each_dropped_prompt_once_in_reason_order():
    # a:1 is both missing m2 and duplicated on m1: counted as missing_model only.
    rows = [row("a:1", "m1"), row("a:1", "m1"), *full("a:2")]
    report = apply_coverage(rows, POOL)
    dropped = report.datasets["a"]["dropped"]
    assert dropped == {"missing_model": 1, "duplicate_row": 0, "prompt_mismatch": 0}
    assert report.dropped_prompts == [
        {"dataset": "a", "prompt_id": "a:1", "reason": "missing_model"}
    ]


def test_coverage_lists_dropped_prompts_sorted_by_dataset_then_prompt_id():
    rows = (
        [row("b:2", "m1", dataset="b")]
        + [row("b:1", "m1", dataset="b"), row("b:1", "m2", dataset="b")]
        + [row("a:9", "m1", "x"), row("a:9", "m2", "y")]
        + [row("a:10", "m1"), row("a:10", "m1"), row("a:10", "m2")]
        + [row("b:0", "m1", dataset="b")]
        + [row("a:1", m) for m in POOL]
    )
    report = apply_coverage(rows, POOL)
    assert report.dropped_prompts == [
        {"dataset": "a", "prompt_id": "a:10", "reason": "duplicate_row"},
        {"dataset": "a", "prompt_id": "a:9", "reason": "prompt_mismatch"},
        {"dataset": "b", "prompt_id": "b:0", "reason": "missing_model"},
        {"dataset": "b", "prompt_id": "b:2", "reason": "missing_model"},
    ]


def test_coverage_drops_dataset_missing_a_model_entirely():
    pool = ("m-strong", "m-mid", "m-cheap")
    rows = [row(f"alpha:{i}", m, dataset="alpha") for i in range(2) for m in pool] + [
        row(f"beta:{i}", m, dataset="beta") for i in range(2) for m in pool[:2]
    ]
    report = apply_coverage(rows, pool)
    assert report.dropped_datasets == {"beta": ["missing model m-cheap"]}
    assert report.dropped_prompts == []
    assert list(report.datasets) == ["alpha"]
    assert {r.dataset for r in report.kept} == {"alpha"}


def test_coverage_counts_non_pool_rows():
    rows = full("a:1") + [row("a:1", "other"), row("a:2", "other")]
    report = apply_coverage(rows, POOL)
    assert report.non_pool_rows == 2
    assert report.datasets["a"]["prompts_in"] == 1
    assert report.datasets["a"]["dropped"]["missing_model"] == 0
    assert len(report.kept) == 2


def test_coverage_rejects_prompt_id_under_two_datasets():
    rows = [row("a:1", "m1", dataset="a"), row("a:1", "m2", dataset="b")]
    with pytest.raises(ValueError, match="a:1"):
        apply_coverage(rows, POOL)


def test_prepare_rejects_inconsistent_success_labels(tmp_path):
    rows = synthetic_rows(n_per_dataset=20)
    bad = rows[0]
    rows[0] = OutcomeRow(**(bad.to_json() | {"score": 1.0, "success": False}))
    paths = write_synthetic_inputs(tmp_path, rows)
    with pytest.raises(ValueError, match=bad.prompt_id):
        prepare_synthetic(paths, rows)
    assert not paths.manifest.exists()


def test_prepare_manifest_contents(tmp_path):
    paths = build_synthetic_project(tmp_path)
    m = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert set(m) == {
        "schema_version",
        "created_at",
        "git_sha",
        "source",
        "pool",
        "datasets",
        "dropped_datasets",
        "dropped_prompts",
        "non_pool_rows",
        "conversion",
        "splits",
        "files",
        "max_completion_tokens",
        "costs",
    }
    assert m["schema_version"] == 1 and m["git_sha"] == "synthetic"
    assert m["created_at"] == "2026-10-26T12:00:00Z"
    assert m["splits"]["counts"] == {
        "train": 300,
        "calibration": 120,
        "certification": 88,
        "test": 92,
    }
    assert m["splits"]["seed"] == 20261026
    assert m["splits"]["percent"] == {
        "train": 50,
        "calibration": 20,
        "certification": 15,
        "test": 15,
    }
    assert m["datasets"]["alpha"]["label_rule"] == "score >= 1.0"
    assert m["datasets"]["alpha"]["score_values"] == [0.0, 1.0]
    assert m["datasets"]["alpha"]["family"] == "reasoning"
    assert m["datasets"]["alpha"]["split_counts"] == {
        "train": 75,
        "calibration": 30,
        "certification": 22,
        "test": 23,
    }
    assert m["datasets"]["alpha"]["prompts_in"] == m["datasets"]["alpha"]["prompts_kept"] == 150
    assert m["pool"] == list(SYNTH_POOL) and m["dropped_datasets"] == {}
    assert m["dropped_prompts"] == []
    assert m["non_pool_rows"] == 0
    assert set(m["conversion"]) == {"records", "usage_clamped"}
    assert m["files"]["test"]["sha256"] == sha256_file(paths.splits_dir / "test.jsonl")
    assert m["files"]["outcomes"]["rows"] == 600 * 3
    assert sum(m["files"][n]["rows"] for n in SPLIT_NAMES) == 600 * 3
    for name in SPLIT_NAMES:
        assert m["files"][name]["path"] == f"data/work/splits/{name}.jsonl"
    mid = m["costs"]["by_model"]["m-mid"]
    assert mid["recorded_usd_all_rows"] == pytest.approx(0.8 * mid["list_usd_all_rows"])
    assert mid["recorded_usd_kept"] == pytest.approx(0.8 * mid["list_usd_kept"])
    strong = m["costs"]["by_model"]["m-strong"]
    assert strong["recorded_usd_all_rows"] == pytest.approx(strong["list_usd_all_rows"])
    ratios = {(c["dataset"], c["model"]): c["ratio"] for c in m["costs"]["by_dataset_model"]}
    assert len(ratios) == 4 * 3
    assert ratios[("alpha", "m-mid")] == pytest.approx(0.8)
    assert ratios[("beta", "m-strong")] == pytest.approx(1.0)
    rows = list(read_outcomes(paths.outcomes))
    longest = max(r.completion_tokens for r in rows)
    assert m["max_completion_tokens"]["overall"] == longest
    assert m["max_completion_tokens"]["by_model"]["m-cheap"] == max(
        r.completion_tokens for r in rows if r.model == "m-cheap"
    )
    prompts = [
        json.loads(line)
        for line in (paths.replay_dir / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(prompts) == 92
    assert [p["prompt_id"] for p in prompts] == sorted(p["prompt_id"] for p in prompts)
    assert all(p["prompt_sha256"] == prompt_sha256(p["prompt"]) for p in prompts)
    assert {p["dataset"]: p["family"] for p in prompts} == SYNTH_DATASETS


def test_prepare_returns_test_ids_and_rules(tmp_path):
    rows = synthetic_rows(n_per_dataset=20)
    paths = write_synthetic_inputs(tmp_path, rows)
    result = prepare_synthetic(paths, rows)
    in_test = {r.prompt_id for r in read_outcomes(split_path(paths.splits_dir, "test"))}
    assert result.test_ids == in_test and isinstance(result.test_ids, frozenset)
    assert set(result.rules) == set(SYNTH_DATASETS)
    assert result.rules["alpha"].describe() == "score >= 1.0"
    assert result.manifest == json.loads(paths.manifest.read_text(encoding="utf-8"))


def test_prepare_records_dropped_prompts_and_datasets(tmp_path):
    rows = synthetic_rows(n_per_dataset=20)
    rows = [r for r in rows if not (r.prompt_id == "alpha:3" and r.model == "m-mid")]
    rows = [r for r in rows if not (r.dataset == "delta" and r.model == "m-cheap")]
    rows.append(OutcomeRow(**(rows[0].to_json() | {"model": "other"})))
    paths = write_synthetic_inputs(tmp_path, rows)
    m = prepare_synthetic(paths, rows).manifest
    assert m["dropped_datasets"] == {"delta": ["missing model m-cheap"]}
    assert "delta" not in m["datasets"]
    assert m["datasets"]["alpha"]["dropped"]["missing_model"] == 1
    assert m["dropped_prompts"] == [
        {"dataset": "alpha", "prompt_id": "alpha:3", "reason": "missing_model"}
    ]
    assert m["datasets"]["alpha"]["prompts_kept"] == 19
    assert m["non_pool_rows"] == 1
    assert sum(m["splits"]["counts"].values()) == 19 + 20 + 20


def test_prepare_is_reproducible(tmp_path):
    first = build_synthetic_project(tmp_path / "a")
    second = build_synthetic_project(tmp_path / "b")
    assert first.manifest.read_bytes() == second.manifest.read_bytes()
    for name in SPLIT_NAMES:
        assert split_path(first.splits_dir, name).read_bytes() == (
            split_path(second.splits_dir, name).read_bytes()
        )


def test_pivot_shapes_and_order():
    rows = [
        row("a:2", "m2", "two", success=False, cost_usd=0.25, completion_tokens=3),
        row("a:1", "m1", "one", cost_usd=0.5, completion_tokens=7),
        row("a:2", "m1", "two", cost_usd=0.75, completion_tokens=9),
        row("a:1", "m2", "one", success=False, cost_usd=1.5, completion_tokens=11),
    ]
    t = pivot(rows, POOL)
    assert len(t) == 2 and t.pool == POOL
    assert t.prompt_ids == ("a:1", "a:2")
    assert t.success.shape == (2, 2) and t.success.dtype == bool
    assert t.success.tolist() == [[True, False], [True, False]]
    assert t.cost.dtype == np.float64 and t.cost.tolist() == [[0.5, 1.5], [0.75, 0.25]]
    assert t.completion_tokens.dtype == np.int64
    assert t.completion_tokens.tolist() == [[7, 11], [9, 3]]
    assert t.texts == ("one", "two")
    assert t.datasets.tolist() == ["a", "a"] and t.families.tolist() == ["qa", "qa"]
    assert t.prompt_bytes.dtype == np.int64 and t.prompt_bytes.tolist() == [3, 3]


def test_pivot_counts_utf8_bytes():
    t = pivot([row("a:1", m, "h\u00e9llo") for m in POOL], POOL)
    assert t.prompt_bytes.tolist() == [6]


def test_pivot_rejects_missing_duplicate_and_unknown_rows():
    with pytest.raises(ValueError, match="a:1"):
        pivot([row("a:1", "m1")], POOL)
    with pytest.raises(ValueError, match="a:1"):
        pivot([*full("a:1"), row("a:1", "m1")], POOL)
    with pytest.raises(ValueError, match="other"):
        pivot([*full("a:1"), row("a:1", "other")], POOL)


def test_table_subset_selects_rows():
    rows = [r for pid in ("a:1", "a:2", "a:3") for r in full(pid, f"text {pid}")]
    t = pivot(rows, POOL)
    sub = t.subset(np.array([True, False, True]))
    assert len(sub) == 2 and sub.prompt_ids == ("a:1", "a:3")
    assert sub.texts == ("text a:1", "text a:3") and sub.pool == POOL
    assert sub.success.shape == (2, 2) and sub.cost.shape == (2, 2)
    assert sub.datasets.shape == (2,) and sub.prompt_bytes.shape == (2,)
    assert len(t) == 3
    with pytest.raises(ValueError):
        t.subset(np.array([True, False]))
    with pytest.raises(ValueError):
        t.subset(np.array([1, 0, 1]))


def test_pivot_of_synthetic_rows():
    rows = synthetic_rows(n_per_dataset=10)
    t = pivot(rows, SYNTH_POOL)
    assert len(t) == 40 and t.success.shape == (40, 3)
    assert list(t.prompt_ids) == sorted(t.prompt_ids)


def test_utc_iso_formats_and_converts():
    assert utc_iso(datetime(2026, 10, 26, 1, 2, 3, tzinfo=UTC)) == "2026-10-26T01:02:03Z"
    assert utc_iso(datetime(2026, 10, 26, 1, 2, 3, tzinfo=timezone(timedelta(hours=-4)))) == (
        "2026-10-26T05:02:03Z"
    )
    with pytest.raises(ValueError):
        utc_iso(datetime(2026, 10, 26, 1, 2, 3))


def test_environment_reports_versions():
    env = environment()
    assert set(env) == {"python", "platform", "numpy", "scipy", "sklearn", "matplotlib"}
    assert env["python"].startswith("3.14")
    assert env["numpy"] == "2.5.4" and env["sklearn"] == "1.9.1"


def git(repo, *args):
    subprocess.run(
        [git_executable(), "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def test_current_git_sha_clean_dirty_and_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert current_git_sha(tmp_path) == "unknown"
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "f.txt").write_text("one\n", encoding="utf-8")
    git(repo, "add", "f.txt")
    git(repo, "commit", "-q", "-m", "init", "--no-gpg-sign")
    sha = current_git_sha(repo)
    assert len(sha) == 40 and set(sha) <= set("0123456789abcdef")
    (repo / "untracked.txt").write_text("x\n", encoding="utf-8")
    assert current_git_sha(repo) == sha
    (repo / "f.txt").write_text("two\n", encoding="utf-8")
    assert current_git_sha(repo) == sha + "-dirty"


def test_git_executable_prefers_system_git():
    expected = "/usr/bin/git" if os.path.exists("/usr/bin/git") else "git"
    assert git_executable() == expected
