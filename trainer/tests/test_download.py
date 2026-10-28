import hashlib
import io
import itertools
import urllib.error
from email.message import Message
from pathlib import Path
from typing import BinaryIO

import pytest

from routeforge_train import download
from routeforge_train.download import (
    MIN_FREE_BYTES,
    ChecksumMismatch,
    DownloadError,
    InsufficientSpace,
    RemoteFile,
    disk_free,
    fetch_archive_members,
    fetch_file,
)
from routeforge_train.errors import TrainerError
from routeforge_train.llmrouterbench import (
    DATASET_FAMILY,
    LRB_ARCHIVE,
    POOL,
    SOURCE,
    load_members,
)
from tests.archive import make_tar_gz, opener_for, remote_for


def plenty(_path: Path) -> int:
    return 10**12


def no_network(url: str) -> BinaryIO:
    raise AssertionError(f"unexpected network access to {url}")


def leftovers(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in tmp_path.iterdir())


def test_constants_match_the_verified_facts():
    assert MIN_FREE_BYTES == 2_147_483_648
    assert LRB_ARCHIVE.size == 1_283_503_080
    assert LRB_ARCHIVE.sha256 == "b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1"
    assert SOURCE["revision"] == "0e5af1b84bf73437a01a1849c0f1d2468baa93fc"
    assert SOURCE["revision"] in LRB_ARCHIVE.url
    assert POOL == (
        "gpt-5",
        "claude-sonnet-4",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "deepseek-v3.1-terminus",
        "qwen3-235b-a22b-2507",
    )
    assert list(DATASET_FAMILY) == [
        "aime",
        "livemathbench",
        "gpqa",
        "mmlupro",
        "livecodebench",
        "simpleqa",
        "hle",
        "arenahard",
    ]
    assert DATASET_FAMILY["livecodebench"] == "code"
    assert DATASET_FAMILY["arenahard"] == "instruction"


def test_members_file_lists_48_pool_files():
    m = load_members()
    assert len(m) == 48 and len(set(m)) == 48 and m == sorted(m)
    assert all(x.startswith("bench-release/") and not set("*?[") & set(x) for x in m)
    pairs = [(x.split("/")[1], x.split("/")[-2]) for x in m]
    assert sorted(pairs) == sorted(itertools.product(DATASET_FAMILY, POOL))


def test_trainer_error_hierarchy():
    assert issubclass(DownloadError, TrainerError)
    assert issubclass(ChecksumMismatch, DownloadError)
    assert issubclass(InsufficientSpace, DownloadError)


def test_fetch_extracts_only_listed_members(tmp_path):
    data = make_tar_gz({"r/a.json": b"A", "r/b.json": b"B", "r/c.json": b"C"})
    out = fetch_archive_members(
        remote_for(data),
        ["r/a.json", "r/c.json"],
        tmp_path / "x",
        opener=opener_for(data),
        free_bytes=plenty,
    )
    found = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert found == [".verified", "r/a.json", "r/c.json"]
    assert (out / "r/a.json").read_bytes() == b"A"
    assert leftovers(tmp_path) == ["x"]


def test_fetch_writes_verified_record(tmp_path):
    import json

    data = make_tar_gz({"r/a.json": b"A", "r/c.json": b"CCC"})
    remote = remote_for(data)
    out = fetch_archive_members(
        remote,
        ["r/a.json", "r/c.json"],
        tmp_path / "x",
        opener=opener_for(data),
        free_bytes=plenty,
    )
    record = json.loads((out / ".verified").read_text(encoding="utf-8"))
    assert record == {
        "url": remote.url,
        "sha256": remote.sha256,
        "size": remote.size,
        "members": {"r/a.json": 1, "r/c.json": 3},
    }


def test_fetch_checksum_mismatch_removes_partial_output(tmp_path):
    data = make_tar_gz({"r/a.json": b"A"})
    remote = RemoteFile("https://example.invalid/a.tar.gz", len(data), "0" * 64)
    with pytest.raises(ChecksumMismatch) as info:
        fetch_archive_members(
            remote, ["r/a.json"], tmp_path / "x", opener=opener_for(data), free_bytes=plenty
        )
    assert "0" * 64 in str(info.value)
    assert hashlib.sha256(data).hexdigest() in str(info.value)
    assert leftovers(tmp_path) == []


def test_fetch_size_mismatch_removes_partial_output(tmp_path):
    data = make_tar_gz({"r/a.json": b"A"})
    good = remote_for(data)
    remote = RemoteFile(good.url, good.size + 1, good.sha256)
    with pytest.raises(ChecksumMismatch):
        fetch_archive_members(
            remote, ["r/a.json"], tmp_path / "x", opener=opener_for(data), free_bytes=plenty
        )
    assert leftovers(tmp_path) == []


def test_fetch_refuses_below_2_gib_free(tmp_path):
    data = make_tar_gz({"r/a.json": b"A"})
    opener = opener_for(data)
    with pytest.raises(InsufficientSpace) as info:
        fetch_archive_members(
            remote_for(data),
            ["r/a.json"],
            tmp_path / "x",
            opener=opener,
            free_bytes=lambda p: 2**31 - 1,
        )
    assert "2.0 GiB" in str(info.value)
    assert opener.calls == 0
    assert leftovers(tmp_path) == []
    out = fetch_archive_members(
        remote_for(data),
        ["r/a.json"],
        tmp_path / "x",
        opener=opener,
        free_bytes=lambda p: 2**31 + 1,  # the floor plus room for the 1-byte member
    )
    assert (out / "r/a.json").read_bytes() == b"A"
    assert opener.calls == 1


def test_fetch_missing_member_fails(tmp_path):
    data = make_tar_gz({"r/a.json": b"A"})
    with pytest.raises(DownloadError) as info:
        fetch_archive_members(
            remote_for(data),
            ["r/a.json", "r/zzz.json"],
            tmp_path / "x",
            opener=opener_for(data),
            free_bytes=plenty,
        )
    assert "r/zzz.json" in str(info.value)
    assert leftovers(tmp_path) == []


def test_fetch_rejects_listed_member_that_is_not_a_regular_file(tmp_path):
    data = make_tar_gz({"r/a.json": b"A"}, symlinks={"r/link.json": "a.json"})
    with pytest.raises(DownloadError):
        fetch_archive_members(
            remote_for(data),
            ["r/a.json", "r/link.json"],
            tmp_path / "x",
            opener=opener_for(data),
            free_bytes=plenty,
        )
    assert leftovers(tmp_path) == []


def test_fetch_rejects_listed_hard_link(tmp_path):
    data = make_tar_gz({"r/a.json": b"A"}, hardlinks={"r/hard.json": "r/a.json"})
    with pytest.raises(DownloadError):
        fetch_archive_members(
            remote_for(data),
            ["r/a.json", "r/hard.json"],
            tmp_path / "x",
            opener=opener_for(data),
            free_bytes=plenty,
        )
    assert leftovers(tmp_path) == []


@pytest.mark.parametrize(
    "name", ["../evil.json", "r/../../evil.json", "/abs/evil.json", "r/./a.json", ""]
)
def test_fetch_rejects_malicious_member_names(tmp_path, name):
    data = make_tar_gz({"r/a.json": b"A", name or "empty": b"EVIL"})
    opener = opener_for(data)
    dest = tmp_path / "sub" / "x"
    with pytest.raises(DownloadError):
        fetch_archive_members(
            remote_for(data), ["r/a.json", name], dest, opener=opener, free_bytes=plenty
        )
    assert opener.calls == 0
    assert not (tmp_path / "evil.json").exists()
    assert not (tmp_path / "sub").exists()


def test_fetch_never_writes_unlisted_traversal_members(tmp_path):
    data = make_tar_gz({"r/a.json": b"A", "../evil.json": b"EVIL"})
    out = fetch_archive_members(
        remote_for(data),
        ["r/a.json"],
        tmp_path / "sub" / "x",
        opener=opener_for(data),
        free_bytes=plenty,
    )
    assert not (tmp_path / "sub" / "evil.json").exists()
    assert not (tmp_path / "evil.json").exists()
    assert (out / "r/a.json").read_bytes() == b"A"


def test_fetch_reuses_verified_extraction_without_network(tmp_path):
    data = make_tar_gz({"r/a.json": b"A", "r/b.json": b"B"})
    remote = remote_for(data)
    first = fetch_archive_members(
        remote, ["r/a.json"], tmp_path / "x", opener=opener_for(data), free_bytes=plenty
    )
    second = fetch_archive_members(
        remote, ["r/a.json"], tmp_path / "x", opener=no_network, free_bytes=lambda p: 0
    )
    assert second == first == tmp_path / "x"


def test_fetch_redownloads_when_verified_member_changed(tmp_path):
    data = make_tar_gz({"r/a.json": b"A"})
    opener = opener_for(data)
    out = fetch_archive_members(
        remote_for(data), ["r/a.json"], tmp_path / "x", opener=opener, free_bytes=plenty
    )
    (out / "r/a.json").write_bytes(b"")
    again = fetch_archive_members(
        remote_for(data), ["r/a.json"], tmp_path / "x", opener=opener, free_bytes=plenty
    )
    assert opener.calls == 2
    assert (again / "r/a.json").read_bytes() == b"A"


def test_fetch_redownloads_when_verified_member_is_missing(tmp_path):
    data = make_tar_gz({"r/a.json": b"A"})
    opener = opener_for(data)
    out = fetch_archive_members(
        remote_for(data), ["r/a.json"], tmp_path / "x", opener=opener, free_bytes=plenty
    )
    (out / "r/a.json").unlink()
    fetch_archive_members(
        remote_for(data), ["r/a.json"], tmp_path / "x", opener=opener, free_bytes=plenty
    )
    assert opener.calls == 2


def test_fetch_redownloads_when_verified_record_is_for_another_archive(tmp_path):
    first = make_tar_gz({"r/a.json": b"A"})
    second = make_tar_gz({"r/a.json": b"Z"})
    opener = opener_for(second)
    fetch_archive_members(
        remote_for(first), ["r/a.json"], tmp_path / "x", opener=opener_for(first), free_bytes=plenty
    )
    out = fetch_archive_members(
        remote_for(second), ["r/a.json"], tmp_path / "x", opener=opener, free_bytes=plenty
    )
    assert opener.calls == 1
    assert (out / "r/a.json").read_bytes() == b"Z"


def test_fetch_redownloads_when_verified_record_is_corrupt(tmp_path):
    data = make_tar_gz({"r/a.json": b"A"})
    opener = opener_for(data)
    out = fetch_archive_members(
        remote_for(data), ["r/a.json"], tmp_path / "x", opener=opener, free_bytes=plenty
    )
    (out / ".verified").write_text("{not json", encoding="utf-8")
    fetch_archive_members(
        remote_for(data), ["r/a.json"], tmp_path / "x", opener=opener, free_bytes=plenty
    )
    assert opener.calls == 2


def test_fetch_clears_stale_partial_directory(tmp_path):
    stale = tmp_path / "x.partial"
    stale.mkdir()
    (stale / "junk.txt").write_text("stale", encoding="utf-8")
    data = make_tar_gz({"r/a.json": b"A"})
    fetch_archive_members(
        remote_for(data), ["r/a.json"], tmp_path / "x", opener=opener_for(data), free_bytes=plenty
    )
    assert not stale.exists()
    assert not list(tmp_path.rglob("junk.txt"))


def test_fetch_replaces_an_unverified_existing_destination(tmp_path):
    dest = tmp_path / "x"
    dest.mkdir()
    (dest / "old.txt").write_text("old", encoding="utf-8")
    data = make_tar_gz({"r/a.json": b"A"})
    fetch_archive_members(
        remote_for(data), ["r/a.json"], dest, opener=opener_for(data), free_bytes=plenty
    )
    assert not (dest / "old.txt").exists()
    assert (dest / "r/a.json").read_bytes() == b"A"


def test_fetch_non_gzip_body_removes_partial(tmp_path):
    body = b"<html>rate limited</html>"
    remote = RemoteFile(
        "https://example.invalid/a.tar.gz", len(body), hashlib.sha256(body).hexdigest()
    )
    with pytest.raises(DownloadError) as info:
        fetch_archive_members(
            remote, ["r/a.json"], tmp_path / "x", opener=opener_for(body), free_bytes=plenty
        )
    assert remote.url in str(info.value)
    assert leftovers(tmp_path) == []


def test_fetch_truncated_stream_removes_partial(tmp_path):
    data = make_tar_gz({"r/a.json": b"A" * 100_000, "r/b.json": b"B" * 100_000})
    remote = remote_for(data)
    with pytest.raises(DownloadError) as info:
        fetch_archive_members(
            remote,
            ["r/a.json", "r/b.json"],
            tmp_path / "x",
            opener=opener_for(data[: len(data) // 2]),
            free_bytes=plenty,
        )
    assert remote.url in str(info.value)
    assert leftovers(tmp_path) == []


def test_fetch_http_error_mentions_url(tmp_path):
    url = "https://example.invalid/a.tar.gz"

    def opener(u: str) -> BinaryIO:
        raise urllib.error.HTTPError(u, 429, "Too Many Requests", Message(), io.BytesIO(b""))

    with pytest.raises(DownloadError) as info:
        fetch_archive_members(
            RemoteFile(url, 1, "0" * 64),
            ["r/a.json"],
            tmp_path / "x",
            opener=opener,
            free_bytes=plenty,
        )
    assert url in str(info.value) and "429" in str(info.value)
    assert leftovers(tmp_path) == []


def test_fetch_url_error_mentions_url(tmp_path):
    url = "https://example.invalid/a.tar.gz"

    def opener(u: str) -> BinaryIO:
        raise urllib.error.URLError("name resolution failed")

    with pytest.raises(DownloadError) as info:
        fetch_archive_members(
            RemoteFile(url, 1, "0" * 64),
            ["r/a.json"],
            tmp_path / "x",
            opener=opener,
            free_bytes=plenty,
        )
    assert url in str(info.value) and "name resolution failed" in str(info.value)
    assert leftovers(tmp_path) == []


def test_fetch_uses_module_defaults_looked_up_at_call_time(tmp_path, monkeypatch):
    data = make_tar_gz({"r/a.json": b"A"})
    opener = opener_for(data)
    monkeypatch.setattr(download, "open_url", opener)
    monkeypatch.setattr(download, "disk_free", plenty)
    out = fetch_archive_members(remote_for(data), ["r/a.json"], tmp_path / "x")
    assert opener.calls == 1
    assert (out / "r/a.json").read_bytes() == b"A"


def test_fetch_file_uses_module_defaults_looked_up_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "open_url", lambda url: io.BytesIO(b"payload"))
    monkeypatch.setattr(download, "disk_free", plenty)
    target = fetch_file(remote_for(b"payload"), tmp_path / "y.bin")
    assert target.read_bytes() == b"payload"


def test_disk_free_uses_nearest_existing_ancestor(tmp_path):
    assert disk_free(tmp_path / "does" / "not" / "exist") > 0


def test_fetch_file_verifies_and_renames(tmp_path):
    body = b"hello bytes"
    dest = tmp_path / "out.bin"
    got = fetch_file(remote_for(body), dest, opener=opener_for(body), free_bytes=plenty)
    assert got == dest and dest.read_bytes() == body
    assert leftovers(tmp_path) == ["out.bin"]


def test_fetch_file_mismatch_leaves_nothing(tmp_path):
    body = b"hello bytes"
    remote = RemoteFile("https://example.invalid/f", len(body), "0" * 64)
    with pytest.raises(ChecksumMismatch):
        fetch_file(remote, tmp_path / "out.bin", opener=opener_for(body), free_bytes=plenty)
    assert leftovers(tmp_path) == []


def test_fetch_file_size_mismatch_leaves_nothing(tmp_path):
    body = b"hello bytes"
    good = remote_for(body)
    remote = RemoteFile(good.url, good.size + 1, good.sha256)
    with pytest.raises(ChecksumMismatch):
        fetch_file(remote, tmp_path / "out.bin", opener=opener_for(body), free_bytes=plenty)
    assert leftovers(tmp_path) == []


def test_fetch_file_reuses_matching_existing_file(tmp_path):
    body = b"hello bytes"
    dest = tmp_path / "out.bin"
    dest.write_bytes(body)
    assert fetch_file(remote_for(body), dest, opener=no_network, free_bytes=lambda p: 0) == dest


def test_fetch_file_replaces_mismatching_existing_file(tmp_path):
    body = b"hello bytes"
    dest = tmp_path / "out.bin"
    dest.write_bytes(b"stale")
    opener = opener_for(body)
    fetch_file(remote_for(body), dest, opener=opener, free_bytes=plenty)
    assert opener.calls == 1 and dest.read_bytes() == body


def test_fetch_file_refuses_below_2_gib_free(tmp_path):
    body = b"hello bytes"
    opener = opener_for(body)
    with pytest.raises(InsufficientSpace) as info:
        fetch_file(remote_for(body), tmp_path / "out.bin", opener=opener, free_bytes=lambda p: 1)
    assert "2.0 GiB" in str(info.value)
    assert opener.calls == 0
    assert leftovers(tmp_path) == []


def test_fetch_file_http_error_leaves_nothing(tmp_path):
    url = "https://example.invalid/f"

    def opener(u: str) -> BinaryIO:
        raise urllib.error.HTTPError(u, 503, "Service Unavailable", Message(), io.BytesIO(b""))

    with pytest.raises(DownloadError) as info:
        fetch_file(
            RemoteFile(url, 1, "0" * 64), tmp_path / "out.bin", opener=opener, free_bytes=plenty
        )
    assert url in str(info.value) and "503" in str(info.value)
    assert leftovers(tmp_path) == []


class TrackingStream(io.BytesIO):
    """In-memory body that records how many bytes the consumer pulled from it."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.bytes_read = 0

    def read(self, size: int | None = -1) -> bytes:
        chunk = super().read(size)
        self.bytes_read += len(chunk)
        return chunk


def test_fetch_stops_reading_once_stream_exceeds_declared_size(tmp_path):
    data = make_tar_gz({"r/a.json": b"A"})
    body = data + b"\0" * (8 * 1024 * 1024)
    stream = TrackingStream(body)
    with pytest.raises(ChecksumMismatch):
        fetch_archive_members(
            remote_for(data),
            ["r/a.json"],
            tmp_path / "x",
            opener=lambda url: stream,
            free_bytes=plenty,
        )
    assert stream.bytes_read <= len(data) + download._CHUNK
    assert leftovers(tmp_path) == []


def test_fetch_stops_extracting_at_free_space_floor(tmp_path):
    data = make_tar_gz({"r/a.json": b"A" * 100})
    with pytest.raises(DownloadError) as info:
        fetch_archive_members(
            remote_for(data),
            ["r/a.json"],
            tmp_path / "x",
            opener=opener_for(data),
            free_bytes=lambda p: MIN_FREE_BYTES + 10,
        )
    assert not isinstance(info.value, ChecksumMismatch)
    assert leftovers(tmp_path) == []


def test_fetch_file_stops_reading_once_body_exceeds_declared_size(tmp_path):
    body = b"hello bytes"
    stream = TrackingStream(body + b"\0" * (8 * 1024 * 1024))
    with pytest.raises(ChecksumMismatch):
        fetch_file(
            remote_for(body),
            tmp_path / "out.bin",
            opener=lambda url: stream,
            free_bytes=plenty,
        )
    assert stream.bytes_read <= len(body) + download._CHUNK
    assert leftovers(tmp_path) == []
