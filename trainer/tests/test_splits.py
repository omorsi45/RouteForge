import ast
from collections import Counter
from pathlib import Path

import pytest

from routeforge_train.errors import TrainerError
from routeforge_train.jsonio import sha256_file
from routeforge_train.outcomes import read_outcomes
from routeforge_train.paths import Paths
from routeforge_train.splits import (
    SEED,
    SPLIT_NAMES,
    SPLIT_PERCENT,
    HeldOutSplitAccessError,
    HeldOutSplitChangedError,
    SplitChecksumError,
    assign_splits,
    read_split,
    read_test_split,
    read_verified_split,
    split_path,
    write_splits,
)
from tests.synth import synthetic_rows

PACKAGE = Path(__file__).resolve().parents[1] / "routeforge_train"


def counts(assignment, prompts):
    c = Counter(assignment[p] for p in prompts)
    return [c[name] for name in SPLIT_NAMES]


def small_rows():
    rows = synthetic_rows(n_per_dataset=20)
    assignment = assign_splits({r.prompt_id: r.dataset for r in rows})
    return rows, assignment


def test_constants():
    assert SEED == 20261026
    assert SPLIT_NAMES == ("train", "calibration", "certification", "test")
    assert SPLIT_PERCENT == {"train": 50, "calibration": 20, "certification": 15, "test": 15}
    assert split_path(Path("d"), "test") == Path("d/test.jsonl")


def test_assign_splits_counts_per_dataset():
    a = [f"a:{i}" for i in range(60)]
    b = [f"b:{i}" for i in range(7)]
    out = assign_splits({p: "a" for p in a} | {p: "b" for p in b})
    assert counts(out, a) == [30, 12, 9, 9]
    assert counts(out, b) == [3, 1, 1, 2]
    assert set(out) == set(a) | set(b)


def test_assign_splits_deterministic_order_independent_and_seed_sensitive():
    mapping = {f"a:{i}": "a" for i in range(60)} | {f"b:{i}": "b" for i in range(40)}
    forward = assign_splits(mapping)
    reverse = assign_splits(dict(reversed(list(mapping.items()))))
    assert forward == reverse
    assert assign_splits(mapping) == forward
    assert assign_splits(mapping, seed=1) != forward


def test_read_split_refuses_test(tmp_path):
    rows, assignment = small_rows()
    write_splits(rows, assignment, tmp_path / "splits", tmp_path)
    with pytest.raises(HeldOutSplitAccessError):
        list(read_split(tmp_path / "splits", "test"))
    with pytest.raises(HeldOutSplitAccessError):
        read_split(tmp_path / "splits", "test")


def test_read_split_rejects_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="bogus"):
        read_split(tmp_path, "bogus")


def test_read_split_returns_only_that_split(tmp_path):
    rows, assignment = small_rows()
    write_splits(rows, assignment, tmp_path / "splits", tmp_path)
    got = list(read_split(tmp_path / "splits", "calibration"))
    assert got and {assignment[r.prompt_id] for r in got} == {"calibration"}
    expected = [r for r in rows if assignment[r.prompt_id] == "calibration"]
    assert sorted(got, key=lambda r: (r.prompt_id, r.model)) == sorted(
        expected, key=lambda r: (r.prompt_id, r.model)
    )


def test_read_test_split_verifies_sha(tmp_path):
    rows, assignment = small_rows()
    write_splits(rows, assignment, tmp_path / "splits", tmp_path)
    sha = sha256_file(split_path(tmp_path / "splits", "test"))
    with pytest.raises(HeldOutSplitChangedError):
        read_test_split(tmp_path / "splits", "0" * 64)
    got = list(read_test_split(tmp_path / "splits", sha))
    assert got and {assignment[r.prompt_id] for r in got} == {"test"}


def test_write_splits_describes_every_file(tmp_path):
    rows, assignment = small_rows()
    infos = write_splits(rows, assignment, tmp_path / "splits", tmp_path)
    assert list(infos) == list(SPLIT_NAMES)
    assert sum(i.rows for i in infos.values()) == len(rows)
    for name, info in infos.items():
        path = split_path(tmp_path / "splits", name)
        assert info.path == path.relative_to(tmp_path).as_posix()
        assert info.sha256 == sha256_file(path)
        assert info.rows == sum(1 for _ in read_outcomes(path))
        ids = [r.prompt_id for r in read_outcomes(path)]
        assert ids == sorted(ids)


def test_write_splits_rejects_row_without_assignment(tmp_path):
    rows, assignment = small_rows()
    del assignment[rows[0].prompt_id]
    with pytest.raises(ValueError, match=rows[0].prompt_id):
        write_splits(rows, assignment, tmp_path / "splits", tmp_path)


def test_write_splits_is_idempotent_for_identical_test_split(tmp_path):
    rows, assignment = small_rows()
    first = write_splits(rows, assignment, tmp_path / "splits", tmp_path)
    test_file = split_path(tmp_path / "splits", "test")
    before = test_file.stat().st_mtime_ns
    second = write_splits(rows, assignment, tmp_path / "splits", tmp_path)
    assert test_file.stat().st_mtime_ns == before
    assert second == first
    assert sorted(p.name for p in test_file.parent.iterdir()) == sorted(
        f"{n}.jsonl" for n in SPLIT_NAMES
    )


def test_write_splits_refuses_to_change_existing_test_split(tmp_path):
    rows, assignment = small_rows()
    write_splits(rows, assignment, tmp_path / "splits", tmp_path)
    test_file = split_path(tmp_path / "splits", "test")
    train_file = split_path(tmp_path / "splits", "train")
    before, train_before = test_file.read_bytes(), train_file.read_bytes()
    moved = next(p for p, s in assignment.items() if s == "test")
    changed = {**assignment, moved: "train"}
    with pytest.raises(HeldOutSplitChangedError, match="delete data/work"):
        write_splits(rows, changed, tmp_path / "splits", tmp_path)
    assert test_file.read_bytes() == before
    assert train_file.read_bytes() == train_before
    assert not list(test_file.parent.glob("*.tmp")) and len(list(test_file.parent.iterdir())) == 4


def test_held_out_errors_are_trainer_errors():
    assert issubclass(HeldOutSplitAccessError, TrainerError)
    assert issubclass(HeldOutSplitChangedError, TrainerError)
    assert issubclass(SplitChecksumError, TrainerError)


def _verified_project(tmp_path):
    rows, assignment = small_rows()
    paths = Paths(tmp_path)
    infos = write_splits(rows, assignment, paths.splits_dir, tmp_path)
    manifest = {"files": {name: info.to_json() for name, info in infos.items()}}
    return paths, manifest


def test_read_verified_split_returns_rows_when_checksum_matches(tmp_path):
    paths, manifest = _verified_project(tmp_path)
    got = list(read_verified_split(paths, manifest, "calibration"))
    assert got == list(read_split(paths.splits_dir, "calibration"))
    assert len(got) == manifest["files"]["calibration"]["rows"]


def test_read_verified_split_rejects_changed_file_and_names_the_split(tmp_path):
    paths, manifest = _verified_project(tmp_path)
    with open(split_path(paths.splits_dir, "calibration"), "ab") as f:
        f.write(b"\n")
    with pytest.raises(SplitChecksumError, match="calibration"):
        read_verified_split(paths, manifest, "calibration")
    assert list(read_verified_split(paths, manifest, "train"))


def test_read_verified_split_rejects_missing_file_and_missing_manifest_entry(tmp_path):
    paths, manifest = _verified_project(tmp_path)
    split_path(paths.splits_dir, "train").unlink()
    with pytest.raises(SplitChecksumError, match="train"):
        read_verified_split(paths, manifest, "train")
    del manifest["files"]["certification"]
    with pytest.raises(SplitChecksumError, match="certification"):
        read_verified_split(paths, manifest, "certification")


def test_read_verified_split_still_refuses_the_test_split(tmp_path):
    paths, manifest = _verified_project(tmp_path)
    with pytest.raises(HeldOutSplitAccessError):
        read_verified_split(paths, manifest, "test")


def test_only_evaluate_reads_test_split():
    callers = set()
    for path in sorted(PACKAGE.glob("*.py")):
        if path.stem == "splits":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            named = (
                (isinstance(node, ast.Name) and node.id == "read_test_split")
                or (isinstance(node, ast.Attribute) and node.attr == "read_test_split")
                or (
                    isinstance(node, ast.ImportFrom)
                    and any(a.name == "read_test_split" for a in node.names)
                )
            )
            if named:
                callers.add(path.stem)
    assert callers == {"evaluate"}
