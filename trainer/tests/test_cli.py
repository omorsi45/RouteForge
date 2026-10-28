import re
import sys
from pathlib import Path

import routeforge_train
from routeforge_train.cli import main
from routeforge_train.paths import Paths

TRAINER_DIR = Path(__file__).resolve().parent.parent
PIN = re.compile(r"^[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9.]*$")


def test_python_is_314():
    assert sys.version_info[:2] == (3, 14)


def test_version():
    assert routeforge_train.__version__ == "0.1.0"


def test_main_without_command_prints_usage_and_exits_2(capsys):
    assert main(["--root", "."]) == 2
    assert "usage:" in capsys.readouterr().err


def test_main_unknown_command_exits_2():
    assert main(["bogus"]) == 2


def test_main_help_exits_0():
    assert main(["--help"]) == 0


def test_paths_from_root(tmp_path):
    p = Paths(tmp_path)
    assert p.raw_lrb == tmp_path / "data/raw/llmrouterbench"
    assert p.raw_text_json == tmp_path / "data/raw/text-json/dataset.jsonl"
    assert p.work == tmp_path / "data/work"
    assert p.outcomes == tmp_path / "data/work/outcomes.jsonl"
    assert p.splits_dir == tmp_path / "data/work/splits"
    assert p.candidates_dir == tmp_path / "data/work/candidates"
    assert p.extraction_dir == tmp_path / "data/work/extraction"
    assert p.manifest == tmp_path / "data/manifest.json"
    assert p.prices == tmp_path / "deploy/prices.yaml"
    assert p.latency_profiles == tmp_path / "bench/latency_profiles.yaml"
    assert p.bench_limits == tmp_path / "bench/limits.json"
    assert p.artifacts == tmp_path / "trainer/artifacts"
    assert p.policy == tmp_path / "trainer/artifacts/policy.json"
    assert p.reports == tmp_path / "trainer/artifacts/reports"
    assert p.replay_dir == tmp_path / "trainer/artifacts/replay"
    assert p.parity_fixtures == tmp_path / "testdata/parity/features.jsonl"


def _pins(name):
    lines = (TRAINER_DIR / name).read_text().splitlines()
    return [line for line in lines if line.strip()]


def test_requirements_are_exact_pins():
    runtime = _pins("requirements.txt")
    dev = _pins("requirements-dev.txt")
    for line in runtime + dev:
        assert PIN.match(line), line
    for pin in (
        "numpy==2.5.4",
        "scipy==1.18.1",
        "scikit-learn==1.9.1",
        "matplotlib==3.11.2",
        "PyYAML==6.0.3",
    ):
        assert pin in runtime
    for pin in ("pytest==9.1.1", "ruff==0.17.3"):
        assert pin in dev
