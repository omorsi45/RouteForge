import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from routeforge_train import download
from routeforge_train.cli import main
from routeforge_train.datacmd import run_data
from routeforge_train.errors import TrainerError
from routeforge_train.features import prompt_sha256
from routeforge_train.llmrouterbench import POOL, SOURCE
from routeforge_train.outcomes import read_outcomes
from routeforge_train.paths import Paths
from tests.archive import make_tar_gz, opener_for, remote_for
from tests.lrb_fixture import record, write_member

REPO_ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 27, 5, 0, 0, tzinfo=UTC)
N_PROMPTS = 40
TEMPLATES = {
    "aime": "bench-release/aime/hybrid/{m}/aime-hybrid-{m}-1.json",
    "gpqa": "bench-release/gpqa/test/{m}/gpqa-test-{m}-1.json",
}


def member_names():
    return [t.format(m=model) for t in TEMPLATES.values() for model in POOL]


def build_archive(tmp_path, *, corrupt=None):
    """tar.gz of 2 datasets x 6 pool models plus one file that is not a wanted member."""
    stage = tmp_path / "stage"
    files = {}
    for dataset, template in TEMPLATES.items():
        for m, model in enumerate(POOL):
            name = template.format(m=model)
            records = [
                record(
                    i,
                    f"{dataset} question {i}",
                    1.0 if (i + m) % 3 else 0.0,
                    prompt_tokens=10 + i,
                    completion_tokens=5 + m,
                    cost=0.001 * (m + 1),
                    raw_output=f"{model} answers {i}",
                )
                for i in range(N_PROMPTS)
            ]
            if corrupt == (dataset, model):
                records[0]["score"] = "bad"
            write_member(stage, name, records)
            files[name] = (stage / name).read_bytes()
    files["bench-release/README.md"] = b"not a member\n"
    return make_tar_gz(files)


def project(tmp_path):
    root = tmp_path / "project"
    (root / "deploy").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "deploy/prices.yaml", root / "deploy/prices.yaml")
    return root


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(url):
        raise AssertionError(f"unexpected download of {url}")

    monkeypatch.setattr(download, "open_url", refuse)


def test_run_data_end_to_end_on_synthetic_archive(tmp_path):
    data = build_archive(tmp_path)
    root = project(tmp_path)
    paths = Paths(root)
    manifest = run_data(
        paths,
        archive=remote_for(data),
        members=member_names(),
        opener=opener_for(data),
        free_bytes=lambda p: 10**12,
        git_sha="t",
        now=NOW,
    )
    assert manifest == json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["git_sha"] == "t" and manifest["created_at"] == "2026-10-27T05:00:00Z"
    assert manifest["pool"] == list(POOL)
    assert manifest["source"]["members"] == 12
    assert manifest["source"]["url"] == remote_for(data).url
    assert manifest["source"]["archive_bytes"] == len(data)
    assert manifest["source"]["archive_sha256"] == remote_for(data).sha256
    assert manifest["source"]["dataset"] == SOURCE["dataset"]
    assert manifest["source"]["revision"] == SOURCE["revision"]
    counts = manifest["splits"]["counts"]
    assert counts == {"train": 40, "calibration": 16, "certification": 12, "test": 12}
    assert sum(counts.values()) == 2 * N_PROMPTS
    assert manifest["files"]["outcomes"]["rows"] == 2 * N_PROMPTS * 6
    assert sum(manifest["files"][n]["rows"] for n in counts) == 2 * N_PROMPTS * 6
    assert set(manifest["conversion"]) == {"records", "usage_clamped"}
    assert len(manifest["conversion"]["records"]) == 12
    assert manifest["dropped_datasets"] == {} and manifest["non_pool_rows"] == 0
    for name in counts:
        assert (paths.splits_dir / f"{name}.jsonl").is_file()
    assert (paths.raw_lrb / member_names()[0]).is_file()
    assert not (paths.raw_lrb / "bench-release/README.md").exists()
    assert len(list(read_outcomes(paths.outcomes))) == 2 * N_PROMPTS * 6

    test_prompts = {r.prompt_id for r in read_outcomes(paths.splits_dir / "test.jsonl")}
    assert len(test_prompts) == counts["test"]
    replay = [
        json.loads(line)
        for line in (paths.replay_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(replay) == counts["test"] * 6
    assert {r["prompt_id"] for r in replay} == test_prompts
    assert {r["model"] for r in replay} == set(POOL)
    for r in replay:
        index = r["prompt_id"].split(":")[1]
        assert r["text"] == f"{r['model']} answers {index}"
        assert r["prompt_sha256"] == prompt_sha256(f"{r['dataset']} question {index}")
    prompts_file = [
        json.loads(line)
        for line in (paths.replay_dir / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {p["prompt_id"] for p in prompts_file} == test_prompts


def test_run_data_second_run_reuses_download(tmp_path):
    data = build_archive(tmp_path)
    root = project(tmp_path)
    opener = opener_for(data)
    kwargs = {
        "archive": remote_for(data),
        "members": member_names(),
        "opener": opener,
        "free_bytes": lambda p: 10**12,
        "git_sha": "t",
        "now": NOW,
    }
    first = run_data(Paths(root), **kwargs)
    second = run_data(Paths(root), **kwargs)
    assert opener.calls == 1 and second == first


def test_run_data_wraps_converter_errors_naming_the_member(tmp_path):
    data = build_archive(tmp_path, corrupt=("gpqa", "gpt-5"))
    root = project(tmp_path)
    with pytest.raises(TrainerError, match="gpqa-test-gpt-5-1.json"):
        run_data(
            Paths(root),
            archive=remote_for(data),
            members=member_names(),
            opener=opener_for(data),
            free_bytes=lambda p: 10**12,
            git_sha="t",
            now=NOW,
        )


def test_run_data_missing_prices_is_trainer_error(tmp_path):
    data = build_archive(tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(TrainerError, match="prices.yaml"):
        run_data(
            Paths(root),
            archive=remote_for(data),
            members=member_names(),
            opener=opener_for(data),
            free_bytes=lambda p: 10**12,
            git_sha="t",
            now=NOW,
        )


def test_cli_data_reports_insufficient_space(tmp_path, monkeypatch, capsys):
    root = project(tmp_path)
    monkeypatch.setattr(download, "disk_free", lambda p: 0)
    assert main(["--root", str(root), "data"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "GiB" in err
    assert not (root / "data/raw/llmrouterbench").exists()


def test_cli_data_missing_prices_reports_error(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "data"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "prices.yaml" in err


def test_cli_data_prints_summary(tmp_path, monkeypatch, capsys):
    data = build_archive(tmp_path)
    root = project(tmp_path)
    monkeypatch.setattr(download, "disk_free", lambda p: 10**12)
    monkeypatch.setattr(download, "open_url", opener_for(data))
    monkeypatch.setattr("routeforge_train.datacmd.LRB_ARCHIVE", remote_for(data))
    monkeypatch.setattr("routeforge_train.datacmd.load_members", member_names)
    assert main(["--root", str(root), "data"]) == 0
    out = capsys.readouterr().out
    assert out == (
        "data OK: 80 prompts, 480 rows, splits train=40 calibration=16 certification=12 test=12\n"
    )
