import os
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

_SYSTEM_GIT = "/usr/bin/git"
_GIT_TIMEOUT_S = 30
_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
}


def git_executable() -> str:
    return _SYSTEM_GIT if os.path.exists(_SYSTEM_GIT) else "git"


def _git(root: Path, *args: str) -> str | None:
    """Standard output of a git command run in `root`, or None when it fails."""
    try:
        done = subprocess.run(
            [git_executable(), "-C", str(root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    return done.stdout if done.returncode == 0 else None


def current_git_sha(root: Path) -> str:
    """HEAD of the repository at `root`, suffixed `-dirty` when tracked files differ."""
    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if head is None or status is None or not head.strip():
        return "unknown"
    return head.strip() + ("-dirty" if status.strip() else "")


def utc_iso(now: datetime) -> str:
    if now.tzinfo is None:
        raise ValueError("utc_iso needs a timezone-aware datetime")
    return now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def environment() -> dict[str, str]:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for key, package in _PACKAGES.items():
        env[key] = metadata.version(package)
    return env
