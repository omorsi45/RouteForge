"""Verified streaming download and exact-member extraction.

The archive is never written to disk. Its bytes flow through a hashing reader into a
streaming tar reader, only exactly named regular members are copied out, and the result
is renamed into place only after the byte count and sha256 of the whole stream match.
"""

import contextlib
import hashlib
import json
import os
import shutil
import tarfile
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, BinaryIO, NoReturn

from routeforge_train.errors import TrainerError
from routeforge_train.jsonio import write_json

MIN_FREE_BYTES = 2 * 1024**3
_CHUNK = 1 << 20
_VERIFIED = ".verified"
_USER_AGENT = "routeforge-train/0.1.0"


@dataclass(frozen=True)
class RemoteFile:
    url: str
    size: int
    sha256: str


class DownloadError(TrainerError):
    """A download or extraction failed; no partial output is left behind."""


class ChecksumMismatch(DownloadError):
    """The streamed bytes do not match the expected size or sha256."""


class InsufficientSpace(DownloadError):
    """Less than MIN_FREE_BYTES is free at the destination."""


Opener = Callable[[str], BinaryIO]


def open_url(url: str) -> BinaryIO:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    return urllib.request.urlopen(request, timeout=60)


def disk_free(path: Path) -> int:
    """Free bytes on the filesystem holding `path`, using its nearest existing ancestor."""
    probe = Path(path).absolute()
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


class _HashingReader:
    """File-like wrapper that hashes and counts every byte read from `raw`.

    Reads past `remote.size` are refused as soon as one byte too many arrives, so a wrong
    or hostile body cannot stream unbounded data before the checksum is compared.
    """

    def __init__(self, raw: BinaryIO, remote: RemoteFile) -> None:
        self._raw = raw
        self._remote = remote
        self._digest = hashlib.sha256()
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        remaining = self._remote.size - self.count
        if size < 0 or size > remaining + 1:
            size = remaining + 1
        chunk = self._raw.read(size)
        self._digest.update(chunk)
        self.count += len(chunk)
        if self.count > self._remote.size:
            raise ChecksumMismatch(
                f"{self._remote.url}: stream exceeds the expected {self._remote.size} bytes"
            )
        return chunk

    def drain(self) -> None:
        while self.read(_CHUNK):
            pass

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def fetch_archive_members(
    remote: RemoteFile,
    members: Sequence[str],
    dest: Path,
    *,
    opener: Opener | None = None,
    free_bytes: Callable[[Path], int] | None = None,
) -> Path:
    """Stream `remote`, verify it, and extract exactly the named regular `members` to `dest`.

    Returns `dest` untouched when a previous verified extraction of the same archive is
    intact. Otherwise downloads into `<dest>.partial`, which replaces `dest` only after the
    whole stream is verified; any failure removes the partial directory.
    """
    wanted = set(members)
    for name in sorted(wanted):
        _check_member_name(remote, name)
    dest = Path(dest)
    if _is_verified(dest, remote, wanted):
        return dest
    free = _require_space(remote, dest.parent, free_bytes)
    open_stream = open_url if opener is None else opener

    staging = dest.parent / (dest.name + ".partial")
    _remove(staging)
    try:
        staging.mkdir(parents=True)
        sizes = _stream_extract(remote, wanted, staging, open_stream, free - MIN_FREE_BYTES)
        write_json(
            staging / _VERIFIED,
            {"url": remote.url, "sha256": remote.sha256, "size": remote.size, "members": sizes},
        )
        _remove(dest)
        os.replace(staging, dest)
    except BaseException as exc:
        _remove(staging)
        _reraise(remote, exc)
    return dest


def fetch_file(
    remote: RemoteFile,
    dest: Path,
    *,
    opener: Opener | None = None,
    free_bytes: Callable[[Path], int] | None = None,
) -> Path:
    """Stream `remote` into `dest`, verifying size and sha256 before the final rename."""
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size == remote.size and _sha256_of(dest) == remote.sha256:
        return dest
    _require_space(remote, dest.parent, free_bytes)
    open_stream = open_url if opener is None else opener

    partial = dest.with_name(dest.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(open_stream(remote.url)) as response, open(partial, "wb") as out:
            reader = _HashingReader(response, remote)
            for chunk in iter(lambda: reader.read(_CHUNK), b""):
                out.write(chunk)
        _verify_stream(remote, reader)
        os.replace(partial, dest)
    except BaseException as exc:
        partial.unlink(missing_ok=True)
        _reraise(remote, exc)
    return dest


def _check_member_name(remote: RemoteFile, name: str) -> None:
    parts = name.split("/")
    if not name or "\\" in name or "\x00" in name or any(p in ("", ".", "..") for p in parts):
        raise DownloadError(f"{remote.url}: refusing unsafe member name {name!r}")


def _require_space(
    remote: RemoteFile, parent: Path, free_bytes: Callable[[Path], int] | None
) -> int:
    """Return the free bytes at `parent`, or raise when fewer than MIN_FREE_BYTES are free."""
    free = (disk_free if free_bytes is None else free_bytes)(parent)
    if free < MIN_FREE_BYTES:
        raise InsufficientSpace(
            f"{remote.url}: {free / 1024**3:.2f} GiB free at {parent}, "
            f"{MIN_FREE_BYTES / 1024**3:.1f} GiB required"
        )
    return free


def _is_verified(dest: Path, remote: RemoteFile, wanted: set[str]) -> bool:
    try:
        record = json.loads((dest / _VERIFIED).read_text(encoding="utf-8"))
        if record["sha256"] != remote.sha256:
            return False
        sizes = record["members"]
        for name in wanted:
            path = dest / name
            if not path.is_file() or path.stat().st_size != sizes[name]:
                return False
    except OSError, ValueError, KeyError, TypeError:
        return False
    return True


def _stream_extract(
    remote: RemoteFile, wanted: set[str], staging: Path, open_stream: Opener, budget: int
) -> dict[str, int]:
    """Extract `wanted` into `staging`, writing at most `budget` bytes in total."""
    sizes: dict[str, int] = {}
    extracted = 0
    with contextlib.closing(open_stream(remote.url)) as response:
        reader = _HashingReader(response, remote)
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for member in tar:
                if member.name not in wanted:
                    continue
                if not member.isreg():
                    raise DownloadError(
                        f"{remote.url}: member {member.name!r} is not a regular file"
                    )
                source = tar.extractfile(member)
                if source is None:
                    raise DownloadError(f"{remote.url}: cannot read member {member.name!r}")
                target = staging / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, open(target, "wb") as out:
                    sizes[member.name] = _copy(source, out, budget - extracted, remote)
                extracted += sizes[member.name]
        reader.drain()
    _verify_stream(remote, reader)
    missing = sorted(wanted - sizes.keys())
    if missing:
        raise DownloadError(f"{remote.url}: archive lacks members: {', '.join(missing)}")
    return sizes


def _copy(source: IO[bytes], out: IO[bytes], budget: int, remote: RemoteFile) -> int:
    written = 0
    while chunk := source.read(_CHUNK):
        if written + len(chunk) > budget:
            raise InsufficientSpace(
                f"{remote.url}: extracting would leave less than "
                f"{MIN_FREE_BYTES / 1024**3:.1f} GiB free"
            )
        out.write(chunk)
        written += len(chunk)
    return written


def _verify_stream(remote: RemoteFile, reader: _HashingReader) -> None:
    actual = reader.hexdigest()
    if reader.count != remote.size or actual != remote.sha256:
        raise ChecksumMismatch(
            f"{remote.url}: expected {remote.size} bytes sha256 {remote.sha256}, "
            f"got {reader.count} bytes sha256 {actual}"
        )


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _reraise(remote: RemoteFile, exc: BaseException) -> NoReturn:
    """Re-raise DownloadError and interrupts unchanged; wrap anything else with the URL."""
    if isinstance(exc, DownloadError) or not isinstance(exc, Exception):
        raise exc
    wrapped = DownloadError(f"{remote.url}: {str(exc) or type(exc).__name__}")
    if isinstance(exc, urllib.error.HTTPError):
        exc.close()  # an HTTPError owns the error response's connection
    raise wrapped from exc
