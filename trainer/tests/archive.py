"""In-memory archive builders for the download tests. No network, no real data."""

import hashlib
import io
import tarfile
from typing import BinaryIO

from routeforge_train.download import RemoteFile


def make_tar_gz(
    files: dict[str, bytes],
    symlinks: dict[str, str] | None = None,
    hardlinks: dict[str, str] | None = None,
) -> bytes:
    """Build a gzip-compressed tar with regular files, then symlinks, then hard links."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        for name, target in (symlinks or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
        for name, target in (hardlinks or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.LNKTYPE
            info.linkname = target
            tar.addfile(info)
    return buf.getvalue()


def remote_for(data: bytes, url: str = "https://example.invalid/a.tar.gz") -> RemoteFile:
    return RemoteFile(url=url, size=len(data), sha256=hashlib.sha256(data).hexdigest())


class CountingOpener:
    """Serves `data` from memory and counts how often it was called."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.calls = 0

    def __call__(self, url: str) -> BinaryIO:
        self.calls += 1
        return io.BytesIO(self.data)


def opener_for(data: bytes) -> CountingOpener:
    return CountingOpener(data)
