import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from routeforge_train import datacmd, train
from routeforge_train.errors import TrainerError
from routeforge_train.paths import Paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="routeforge_train",
        description="Offline trainer, calibrator and evaluator for RouteForge routing policies.",
    )
    parser.add_argument("--root", default=".", help="repository root (default: .)")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    datacmd.register(subparsers)
    train.register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    func = getattr(args, "func", None)
    if func is None:
        parser.print_usage(sys.stderr)
        return 2
    try:
        return func(Paths(Path(args.root)), args)
    except TrainerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
