import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from routeforge_train.errors import TrainerError
from routeforge_train.jsonio import FileInfo, file_info, sha256_file
from routeforge_train.outcomes import OutcomeRow, read_outcomes, write_outcomes
from routeforge_train.paths import Paths

SEED = 20261026
SPLIT_NAMES = ("train", "calibration", "certification", "test")
SPLIT_PERCENT = {"train": 50, "calibration": 20, "certification": 15, "test": 15}
_HELD_OUT = "test"


class HeldOutSplitAccessError(TrainerError):
    """Something other than the final evaluation tried to read the test split."""


class HeldOutSplitChangedError(TrainerError):
    """The held-out test split on disk differs from what was expected."""


class SplitChecksumError(TrainerError):
    """A development split file differs from the checksum the manifest recorded."""


def assign_splits(dataset_of: Mapping[str, str], seed: int = SEED) -> dict[str, str]:
    """Stratified split of every prompt, per dataset, independent of input order."""
    rng = np.random.default_rng(seed)
    out: dict[str, str] = {}
    for dataset in sorted(set(dataset_of.values())):
        ids = sorted(p for p, d in dataset_of.items() if d == dataset)
        n = len(ids)
        train = n * SPLIT_PERCENT["train"] // 100
        calibration = train + n * SPLIT_PERCENT["calibration"] // 100
        certification = calibration + n * SPLIT_PERCENT["certification"] // 100
        bounds = (train, calibration, certification)
        for rank, j in enumerate(rng.permutation(n)):
            out[ids[j]] = SPLIT_NAMES[sum(rank >= b for b in bounds)]
    return out


def split_path(splits_dir: Path, name: str) -> Path:
    return Path(splits_dir) / f"{name}.jsonl"


def write_splits(
    rows: Sequence[OutcomeRow],
    assignment: Mapping[str, str],
    splits_dir: Path,
    root: Path,
) -> dict[str, FileInfo]:
    """Write one file per split, each ordered by prompt_id.

    The held-out test split is checked first and never rewritten: identical bytes are left
    untouched and different bytes raise before any other file changes.
    """
    buckets: dict[str, list[OutcomeRow]] = {name: [] for name in SPLIT_NAMES}
    for row in sorted(rows, key=lambda r: r.prompt_id):
        name = assignment.get(row.prompt_id)
        if name not in buckets:
            raise ValueError(f"{row.prompt_id}: no valid split assignment ({name!r})")
        buckets[name].append(row)

    splits_dir = Path(splits_dir)
    _write_held_out(buckets[_HELD_OUT], splits_dir, root)
    for name in SPLIT_NAMES:
        if name != _HELD_OUT:
            write_outcomes(buckets[name], split_path(splits_dir, name), root)
    return {
        name: file_info(split_path(splits_dir, name), root, len(buckets[name]))
        for name in SPLIT_NAMES
    }


def _write_held_out(rows: Sequence[OutcomeRow], splits_dir: Path, root: Path) -> None:
    target = split_path(splits_dir, _HELD_OUT)
    candidate = target.with_name(target.name + ".new")
    write_outcomes(rows, candidate, root)
    try:
        if not target.exists():
            os.replace(candidate, target)
        elif candidate.read_bytes() != target.read_bytes():
            raise HeldOutSplitChangedError(
                f"{target} exists and differs from the rebuilt test split; "
                "delete data/work to rebuild the held-out split"
            )
    finally:
        candidate.unlink(missing_ok=True)


def read_split(splits_dir: Path, name: str) -> Iterator[OutcomeRow]:
    """Rows of a development split. The held-out test split is refused."""
    if name == _HELD_OUT:
        raise HeldOutSplitAccessError(
            "the test split is held out; only the final evaluation may read it"
        )
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}")
    return read_outcomes(split_path(splits_dir, name))


def read_verified_split(
    paths: Paths, manifest: Mapping[str, Any], name: str
) -> Iterator[OutcomeRow]:
    """Rows of a development split, after checking the file against the manifest checksum."""
    rows = read_split(paths.splits_dir, name)
    path = split_path(paths.splits_dir, name)
    entry = manifest.get("files", {}).get(name)
    expected = entry.get("sha256") if isinstance(entry, Mapping) else None
    if not isinstance(expected, str):
        raise SplitChecksumError(f"{path}: the manifest records no sha256 for the {name} split")
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise SplitChecksumError(f"{path}: cannot read the {name} split: {exc}") from exc
    if actual != expected:
        raise SplitChecksumError(
            f"{path}: {name} split sha256 {actual} does not match the manifest {expected}; "
            "rerun the data command"
        )
    return rows


def read_test_split(splits_dir: Path, expected_sha256: str) -> Iterator[OutcomeRow]:
    """Rows of the test split, after checking the file against the recorded checksum.

    Only evaluate.py may call this.
    """
    path = split_path(splits_dir, _HELD_OUT)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise HeldOutSplitChangedError(
            f"{path}: sha256 {actual} does not match the recorded {expected_sha256}"
        )
    return read_outcomes(path)
