import hashlib

import pytest

from routeforge_train.jsonio import (
    FileInfo,
    atomic_writer,
    canonical_bytes,
    dumps_line,
    file_info,
    write_json,
)


def test_canonical_bytes_sorted_compact_utf8():
    assert canonical_bytes({"b": 1, "a": "é"}) == '{"a":"é","b":1}'.encode()


def test_canonical_bytes_rejects_nan():
    with pytest.raises(ValueError):
        canonical_bytes({"x": float("nan")})


def test_dumps_line_is_one_compact_line():
    assert dumps_line({"a": [1, 2]}) == '{"a":[1,2]}\n'


def test_dumps_line_rejects_nan():
    with pytest.raises(ValueError):
        dumps_line({"x": float("inf")})


def test_atomic_writer_replaces_on_success(tmp_path):
    target = tmp_path / "sub" / "out.txt"
    target.parent.mkdir()
    target.write_text("old")
    with atomic_writer(target) as f:
        f.write("new")
    assert target.read_text() == "new"
    assert list(tmp_path.rglob("*.tmp")) == []


def test_atomic_writer_creates_parents_and_supports_binary(tmp_path):
    target = tmp_path / "a" / "b" / "out.bin"
    with atomic_writer(target, "wb") as f:
        f.write(b"\x00\x01")
    assert target.read_bytes() == b"\x00\x01"


def test_atomic_writer_removes_tmp_on_error(tmp_path):
    target = tmp_path / "out.txt"
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_writer(target) as f:
            f.write("partial")
            raise RuntimeError("boom")
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_json_sorted_indented_trailing_newline(tmp_path):
    p = tmp_path / "x.json"
    write_json(p, {"b": 1, "a": 2})
    assert p.read_text() == '{\n  "a": 2,\n  "b": 1\n}\n'


def test_write_json_rejects_nan_and_leaves_no_file(tmp_path):
    p = tmp_path / "x.json"
    with pytest.raises(ValueError):
        write_json(p, {"a": float("nan")})
    assert not p.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_file_info_relative_path_rows_bytes_sha(tmp_path):
    f = tmp_path / "x" / "y.jsonl"
    f.parent.mkdir()
    f.write_bytes(b"ab\n")
    info = file_info(f, tmp_path, 1)
    assert info == FileInfo("x/y.jsonl", 1, 3, hashlib.sha256(b"ab\n").hexdigest())
    assert info.to_json() == {
        "path": "x/y.jsonl",
        "rows": 1,
        "bytes": 3,
        "sha256": hashlib.sha256(b"ab\n").hexdigest(),
    }
