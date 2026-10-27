"""The `data` command: download, convert, split and describe the LLMRouterBench pool."""

import argparse
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from routeforge_train.download import Opener, RemoteFile, fetch_archive_members
from routeforge_train.errors import TrainerError
from routeforge_train.llmrouterbench import LRB_ARCHIVE, POOL, SOURCE, load_members
from routeforge_train.lrb_convert import convert_llmrouterbench, export_replay
from routeforge_train.paths import Paths
from routeforge_train.prepare import prepare
from routeforge_train.prices import load_prices
from routeforge_train.provenance import current_git_sha
from routeforge_train.splits import SPLIT_NAMES


def run_data(
    paths: Paths,
    *,
    archive: RemoteFile | None = None,
    members: Sequence[str] | None = None,
    opener: Opener | None = None,
    free_bytes: Callable[[Path], int] | None = None,
    git_sha: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch the pool members, build outcomes, splits, replay files and the manifest.

    `opener` and `free_bytes` default to the download module's own, looked up at call time.
    Returns the manifest.
    """
    archive = LRB_ARCHIVE if archive is None else archive
    members = load_members() if members is None else list(members)
    prices = load_prices(paths.prices)
    now = datetime.now(UTC) if now is None else now
    git_sha = current_git_sha(paths.root) if git_sha is None else git_sha

    fetch_archive_members(archive, members, paths.raw_lrb, opener=opener, free_bytes=free_bytes)
    try:
        stats = convert_llmrouterbench(paths.raw_lrb, members, paths.outcomes, paths.root)
        converted = stats.to_json()
        source = {
            **SOURCE,
            "url": archive.url,
            "archive_bytes": archive.size,
            "archive_sha256": archive.sha256,
            "members": len(members),
        }
        result = prepare(
            paths,
            pool=POOL,
            source=source,
            conversion={
                "records": converted["records"],
                "usage_clamped": converted["usage_clamped"],
            },
            prices=prices,
            git_sha=git_sha,
            now=now,
        )
        replay = export_replay(
            paths.raw_lrb,
            members,
            result.test_ids,
            result.rules,
            paths.replay_dir / "test.jsonl",
            paths.root,
        )
    except (ValueError, OSError) as exc:
        raise TrainerError(f"cannot build the dataset from {paths.raw_lrb}: {exc}") from exc
    expected = len(result.test_ids) * len(POOL)
    if replay.rows != expected:
        raise TrainerError(
            f"{replay.path}: {replay.rows} replay rows, expected {expected} "
            f"({len(result.test_ids)} test prompts x {len(POOL)} models)"
        )
    return result.manifest


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "data", help="download LLMRouterBench, build outcomes, splits and the manifest"
    )
    parser.set_defaults(func=_run)


def _run(paths: Paths, _args: argparse.Namespace) -> int:
    manifest = run_data(paths)
    counts = manifest["splits"]["counts"]
    rows = sum(manifest["files"][name]["rows"] for name in SPLIT_NAMES)
    summary = " ".join(f"{name}={counts[name]}" for name in SPLIT_NAMES)
    print(f"data OK: {sum(counts.values())} prompts, {rows} rows, splits {summary}")
    return 0
