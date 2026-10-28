"""Guards on the committed derived artifacts.

They read only files tracked in the repository, never data/raw or data/work, so they run in CI
without the dataset. A failure means a committed file no longer matches the policy it was
generated with: regenerate the artifacts together, never edit one by hand.
"""

import json
import subprocess
from pathlib import Path

import pytest

from routeforge_train.llmrouterbench import POOL
from routeforge_train.parity import PARITY_CASES, write_parity_fixtures
from routeforge_train.policy import load_policy
from routeforge_train.report_md import render_evaluation_markdown, render_training_markdown

REPO = Path(__file__).resolve().parents[2]
POLICY = REPO / "trainer/artifacts/policy.json"
PRICES = REPO / "deploy/prices.yaml"
PARITY = REPO / "testdata/parity/features.jsonl"
MANIFEST = REPO / "data/manifest.json"
LIMITS = REPO / "bench/limits.json"
REPORTS = REPO / "trainer/artifacts/reports"

MAX_POLICY_BYTES = 5_000_000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
NOT_TRACKED = ("data/raw/", "data/work/", "trainer/artifacts/replay/")

# Fields every platform must reproduce exactly, then the float fields and their tolerances.
EXACT_FIELDS = (
    "name",
    "messages",
    "text",
    "prompt_sha256",
    "hashed_indices",
    "static_family",
    "policy_version",
    "decisions",
)
FEATURE_VECTORS = ("hashed_values", "dense_raw", "dense")
FEATURE_TOLERANCE = 1e-12
SCORE_FIELDS = ("raw", "p", "lcb")
SCORE_TOLERANCE = 1e-9


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_exact(path: Path) -> str:
    """The file as written, with no newline translation that could hide CRLF drift."""
    return path.read_bytes().decode("utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in read_text_exact(path).splitlines()]


def assert_parity_row_matches(committed: dict, fresh: dict) -> None:
    """Exact on text, indices, blocks, model names and decisions; tolerant on float values.

    Byte equality is not required: libm and BLAS may differ in the last digit across platforms.
    """
    case = fresh["name"]
    assert committed.keys() == fresh.keys(), case
    for key in EXACT_FIELDS:
        assert committed[key] == fresh[key], (case, key)
    for key in FEATURE_VECTORS:
        assert committed[key] == pytest.approx(fresh[key], rel=0, abs=FEATURE_TOLERANCE), (
            case,
            key,
        )
    assert len(committed["scores"]) == len(fresh["scores"]), case
    for got, want in zip(committed["scores"], fresh["scores"], strict=True):
        assert got["model"] == want["model"], case
        assert got["block"] == want["block"], (case, got["model"])
        for key in SCORE_FIELDS:
            assert got[key] == pytest.approx(want[key], rel=0, abs=SCORE_TOLERANCE), (
                case,
                got["model"],
                key,
            )


def test_committed_policy_validates_and_is_under_cap():
    policy = load_policy(POLICY)
    assert POLICY.stat().st_size < MAX_POLICY_BYTES
    assert [entry.name for entry in policy.models] == list(POOL)


def test_committed_parity_fixtures_regenerate_from_policy(tmp_path):
    out = tmp_path / "features.jsonl"
    count = write_parity_fixtures(POLICY, PRICES, out)
    committed, fresh = read_jsonl(PARITY), read_jsonl(out)
    assert count == len(PARITY_CASES) == len(committed) == len(fresh)
    assert [row["name"] for row in committed] == [name for name, _ in PARITY_CASES]
    for committed_row, fresh_row in zip(committed, fresh, strict=True):
        assert_parity_row_matches(committed_row, fresh_row)


def test_committed_reports_and_limits_reference_policy_version():
    version = load_policy(POLICY).version
    assert read_json(POLICY)["version"] == version
    assert read_json(REPORTS / "training.json")["policy_version"] == version
    assert read_json(REPORTS / "evaluation.json")["policy_version"] == version
    assert read_json(LIMITS)["policy_version"] == version
    assert {row["policy_version"] for row in read_jsonl(PARITY)} == {version}


def test_committed_bench_limit_matches_manifest_max_completion_tokens():
    limits = read_json(LIMITS)
    overall = read_json(MANIFEST)["max_completion_tokens"]["overall"]
    assert limits["max_output_tokens"] == overall


def test_committed_markdown_regenerates_from_json():
    training = read_json(REPORTS / "training.json")
    evaluation = read_json(REPORTS / "evaluation.json")
    assert render_training_markdown(training) == read_text_exact(REPORTS / "training.md")
    assert render_evaluation_markdown(evaluation) == read_text_exact(REPORTS / "evaluation.md")


@pytest.mark.parametrize("kind", ["lr", "gbm", "knn"])
def test_committed_pngs_exist(kind):
    png = REPORTS / f"reliability-{kind}.png"
    with png.open("rb") as f:
        assert f.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE


def test_no_raw_data_tracked():
    git = "/usr/bin/git" if Path("/usr/bin/git").exists() else "git"
    out = subprocess.run(
        [git, "-C", str(REPO), "ls-files", "-z", "data", "trainer/artifacts", "testdata", "bench"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tracked = [path for path in out.split("\0") if path]
    assert all(not path.startswith(NOT_TRACKED) for path in tracked)
