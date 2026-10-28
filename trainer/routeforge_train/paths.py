from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Every repository path the trainer reads or writes, derived from one root."""

    root: Path

    @property
    def raw_lrb(self) -> Path:
        return self.root / "data/raw/llmrouterbench"

    @property
    def raw_text_json(self) -> Path:
        return self.root / "data/raw/text-json/dataset.jsonl"

    @property
    def work(self) -> Path:
        return self.root / "data/work"

    @property
    def outcomes(self) -> Path:
        return self.work / "outcomes.jsonl"

    @property
    def splits_dir(self) -> Path:
        return self.work / "splits"

    @property
    def candidates_dir(self) -> Path:
        return self.work / "candidates"

    @property
    def extraction_dir(self) -> Path:
        return self.work / "extraction"

    @property
    def manifest(self) -> Path:
        return self.root / "data/manifest.json"

    @property
    def prices(self) -> Path:
        return self.root / "deploy/prices.yaml"

    @property
    def latency_profiles(self) -> Path:
        return self.root / "bench/latency_profiles.yaml"

    @property
    def bench_limits(self) -> Path:
        return self.root / "bench/limits.json"

    @property
    def artifacts(self) -> Path:
        return self.root / "trainer/artifacts"

    @property
    def policy(self) -> Path:
        return self.artifacts / "policy.json"

    @property
    def reports(self) -> Path:
        return self.artifacts / "reports"

    @property
    def replay_dir(self) -> Path:
        return self.artifacts / "replay"

    @property
    def parity_fixtures(self) -> Path:
        return self.root / "testdata/parity/features.jsonl"
