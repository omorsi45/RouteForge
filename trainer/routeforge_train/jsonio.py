import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any


def dumps_line(obj: Any) -> str:
    """Compact single-line JSON with a trailing newline."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"


def canonical_bytes(obj: Any) -> bytes:
    """Canonical JSON: sorted keys, compact separators, UTF-8, no NaN."""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


@contextmanager
def atomic_writer(path: Path, mode: str = "w") -> Iterator[IO]:
    """Write to `<name>.tmp` beside `path`, then replace `path` on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    kwargs = {} if "b" in mode else {"encoding": "utf-8", "newline": "\n"}
    try:
        with open(tmp, mode, **kwargs) as f:
            yield f
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_json(path: Path, obj: Any) -> None:
    """Atomically write indented, key-sorted JSON with a trailing newline."""
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    with atomic_writer(path) as f:
        f.write(text + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileInfo:
    path: str
    rows: int
    bytes: int
    sha256: str

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "rows": self.rows, "bytes": self.bytes, "sha256": self.sha256}


def file_info(path: Path, root: Path, rows: int) -> FileInfo:
    """Describe `path` relative to `root` with POSIX separators."""
    path = Path(path)
    return FileInfo(
        path=path.relative_to(root).as_posix(),
        rows=rows,
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )
