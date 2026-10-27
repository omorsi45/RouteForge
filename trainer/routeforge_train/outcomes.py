import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from routeforge_train.jsonio import FileInfo, atomic_writer, dumps_line, file_info

FIELDS = (
    "prompt_id",
    "dataset",
    "family",
    "prompt",
    "model",
    "score",
    "success",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
)

_NON_EMPTY_STRINGS = ("prompt_id", "dataset", "family", "model")
_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class OutcomeRow:
    """One model's recorded result on one prompt."""

    prompt_id: str
    dataset: str
    family: str
    prompt: str
    model: str
    score: float
    success: bool
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

    def to_json(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in FIELDS}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Self:
        keys = set(d)
        if keys != set(FIELDS):
            missing = sorted(set(FIELDS) - keys)
            extra = sorted(keys - set(FIELDS))
            raise ValueError(f"outcome row keys differ: missing {missing}, unexpected {extra}")
        for name in _NON_EMPTY_STRINGS:
            if not isinstance(d[name], str) or not d[name]:
                raise ValueError(f"outcome field {name} must be a non-empty string")
        if not isinstance(d["prompt"], str):
            raise ValueError("outcome field prompt must be a string")
        score = d["score"]
        if not _is_number(score) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("outcome field score must be a finite number in [0, 1]")
        if not isinstance(d["success"], bool):
            raise ValueError("outcome field success must be a bool")
        for name in _TOKEN_FIELDS:
            value = d[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"outcome field {name} must be an integer >= 0")
        cost = d["cost_usd"]
        if not _is_number(cost) or not math.isfinite(cost) or cost < 0:
            raise ValueError("outcome field cost_usd must be a finite number >= 0")
        return cls(
            prompt_id=d["prompt_id"],
            dataset=d["dataset"],
            family=d["family"],
            prompt=d["prompt"],
            model=d["model"],
            score=float(score),
            success=d["success"],
            prompt_tokens=d["prompt_tokens"],
            completion_tokens=d["completion_tokens"],
            cost_usd=float(cost),
        )


def write_outcomes(rows: Iterable[OutcomeRow], path: Path, root: Path) -> FileInfo:
    """Atomically write one JSON line per row and describe the file relative to `root`."""
    count = 0
    with atomic_writer(path) as f:
        for row in rows:
            f.write(dumps_line(row.to_json()))
            count += 1
    return file_info(path, root, count)


def read_outcomes(path: Path) -> Iterator[OutcomeRow]:
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            try:
                yield OutcomeRow.from_json(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
