import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from routeforge_train.errors import TrainerError
from routeforge_train.features import prompt_sha256
from routeforge_train.jsonio import FileInfo, atomic_writer, dumps_line, file_info
from routeforge_train.labels import LabelRule, label_rule
from routeforge_train.llmrouterbench import DATASET_FAMILY, POOL
from routeforge_train.outcomes import OutcomeRow, write_outcomes

_PREFIX = "bench-release"
# Model output and grader fields; none of them is ever kept except raw_output for replay.
_DISCARDED = ("prediction", "origin_query", "ground_truth")


def member_dataset_model(member: str) -> tuple[str, str]:
    """Dataset directory and model of an archive member such as `bench-release/aime/...`."""
    parts = member.split("/")
    if len(parts) < 4 or parts[0] != _PREFIX:
        raise ValueError(f"not a bench-release member: {member}")
    dataset = parts[1]
    if dataset not in DATASET_FAMILY:
        raise ValueError(f"unknown dataset {dataset!r} in member {member}")
    return dataset, parts[-2]


@dataclass(frozen=True)
class ConvertStats:
    rows: int
    records: dict[tuple[str, str], int]
    clamped: dict[tuple[str, str], int]
    label_rules: dict[str, LabelRule]

    def to_json(self) -> dict[str, Any]:
        def ordered(table: Mapping[tuple[str, str], int]) -> list[tuple[str, str, int]]:
            keys = sorted(table, key=lambda k: (list(DATASET_FAMILY).index(k[0]), POOL.index(k[1])))
            return [(dataset, model, table[(dataset, model)]) for dataset, model in keys]

        return {
            "rows": self.rows,
            "records": [
                {"dataset": dataset, "model": model, "records": count}
                for dataset, model, count in ordered(self.records)
            ],
            "usage_clamped": [
                {"dataset": dataset, "model": model, "completion_tokens_clamped": count}
                for dataset, model, count in ordered(self.clamped)
            ],
            "label_rules": {
                dataset: {"rule": rule.describe(), "score_values": list(rule.score_values)}
                for dataset, rule in sorted(
                    self.label_rules.items(), key=lambda kv: list(DATASET_FAMILY).index(kv[0])
                )
            },
        }


@dataclass(frozen=True, slots=True)
class _Record:
    index: int
    prompt: str
    score: float
    prompt_tokens: int
    completion_tokens: int
    clamped: bool
    cost: float
    raw_output: str | None


def convert_llmrouterbench(
    raw_dir: Path, members: Sequence[str], out: Path, root: Path
) -> ConvertStats:
    """Reduce the member files to the outcomes format, one file in memory at a time."""
    grouped = _group_members(members)
    records: dict[tuple[str, str], int] = {}
    clamped: dict[tuple[str, str], int] = {}
    rules: dict[str, LabelRule] = {}

    def rows() -> Iterator[OutcomeRow]:
        for dataset, by_model in grouped:
            loaded: list[tuple[str, list[_Record]]] = []
            for model, member in by_model:
                recs = _read_member(raw_dir, member, dataset, model, _never)
                records[(dataset, model)] = len(recs)
                count = sum(1 for rec in recs if rec.clamped)
                if count:
                    clamped[(dataset, model)] = count
                loaded.append((model, recs))
            rule = label_rule(rec.score for _, recs in loaded for rec in recs)
            rules[dataset] = rule
            for model, recs in loaded:
                for rec in recs:
                    yield OutcomeRow(
                        prompt_id=f"{dataset}:{rec.index}",
                        dataset=dataset,
                        family=DATASET_FAMILY[dataset],
                        prompt=rec.prompt,
                        model=model,
                        score=rec.score,
                        success=rule.is_success(rec.score),
                        prompt_tokens=rec.prompt_tokens,
                        completion_tokens=rec.completion_tokens,
                        cost_usd=rec.cost,
                    )

    info = write_outcomes(rows(), out, root)
    return ConvertStats(info.rows, records, clamped, rules)


def export_replay(
    raw_dir: Path,
    members: Sequence[str],
    test_ids: AbstractSet[str],
    rules: Mapping[str, LabelRule],
    out: Path,
    root: Path,
) -> FileInfo:
    """Write the recorded raw output of every test-split prompt for the offline replay."""
    grouped = _group_members(members)
    count = 0
    with atomic_writer(out) as f:
        for dataset, by_model in grouped:
            if dataset not in rules:
                raise ValueError(f"no label rule for dataset {dataset}")
            rule = rules[dataset]

            def wanted(index: object, dataset: str = dataset) -> bool:
                return _is_int(index) and f"{dataset}:{index}" in test_ids

            for model, member in by_model:
                for rec in _read_member(raw_dir, member, dataset, model, wanted):
                    if rec.raw_output is None:
                        continue
                    f.write(
                        dumps_line(
                            {
                                "prompt_sha256": prompt_sha256(rec.prompt),
                                "model": model,
                                "prompt_id": f"{dataset}:{rec.index}",
                                "dataset": dataset,
                                "text": rec.raw_output,
                                "prompt_tokens": rec.prompt_tokens,
                                "completion_tokens": rec.completion_tokens,
                                "cost_usd": rec.cost,
                                "score": rec.score,
                                "success": rule.is_success(rec.score),
                            }
                        )
                    )
                    count += 1
    return file_info(out, root, count)


def _never(_index: object) -> bool:
    return False


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _group_members(members: Sequence[str]) -> list[tuple[str, list[tuple[str, str]]]]:
    """Members grouped by dataset in table order, models in pool order."""
    table: dict[tuple[str, str], str] = {}
    for member in members:
        dataset, model = member_dataset_model(member)
        if model not in POOL:
            raise ValueError(f"model {model!r} of member {member} is not in the pool")
        if (dataset, model) in table:
            raise ValueError(f"duplicate member for {dataset}/{model}: {member}")
        table[(dataset, model)] = member
    grouped = []
    for dataset in DATASET_FAMILY:
        by_model = [(model, table[(dataset, model)]) for model in POOL if (dataset, model) in table]
        if by_model:
            grouped.append((dataset, by_model))
    return grouped


def _reducing_hook(keep_raw: Callable[[object], bool]) -> Callable[[dict[str, Any]], Any]:
    """Drop model output and grader fields from each record while the file is parsed."""

    def hook(obj: dict[str, Any]) -> Any:
        if "raw_output" in obj:
            raw = obj.pop("raw_output")
            for name in _DISCARDED:
                obj.pop(name, None)
            if keep_raw(obj.get("index")):
                obj["raw_output"] = raw
        return obj

    return hook


def _read_member(
    raw_dir: Path,
    member: str,
    dataset: str,
    model: str,
    keep_raw: Callable[[object], bool],
) -> list[_Record]:
    try:
        with open(raw_dir / member, encoding="utf-8") as fp:
            doc = json.load(fp, object_hook=_reducing_hook(keep_raw))
    except ValueError as exc:
        raise ValueError(f"{member}: cannot parse: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("records"), list):
        raise ValueError(f"{member}: expected an object with a records list")
    for key, expected in (("model_name", model), ("dataset_name", dataset)):
        if doc.get(key) != expected:
            raise TrainerError(
                f"{member}: {key} is {doc.get(key)!r} but the path says {expected!r}"
            )
    seen: set[int] = set()
    recs = []
    for position, raw in enumerate(doc["records"]):
        rec = _parse_record(member, position, raw, keep_raw)
        if rec.index in seen:
            raise ValueError(f"{member}: duplicate index {rec.index}")
        seen.add(rec.index)
        recs.append(rec)
    recs.sort(key=lambda rec: rec.index)
    return recs


def _parse_record(
    member: str, position: int, raw: object, keep_raw: Callable[[object], bool]
) -> _Record:
    if not isinstance(raw, dict):
        raise ValueError(f"{member} record #{position}: not an object")
    index = raw.get("index")
    if not _is_int(index):
        raise ValueError(f"{member} record #{position}: index must be an integer")

    def fail(problem: str) -> ValueError:
        return ValueError(f"{member} index {index}: {problem}")

    prompt = raw.get("prompt")
    if not isinstance(prompt, str):
        raise fail("prompt must be a string")
    try:
        prompt.encode("utf-8")
    except UnicodeEncodeError:
        raise fail("prompt is not valid UTF-8 text") from None
    score = raw.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, int | float)
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
    ):
        raise fail("score must be a finite number in [0, 1]")
    prompt_tokens = raw.get("prompt_tokens")
    if not _is_int(prompt_tokens) or prompt_tokens < 0:
        raise fail("prompt_tokens must be an integer >= 0")
    completion_tokens = raw.get("completion_tokens")
    if not _is_int(completion_tokens):
        raise fail("completion_tokens must be an integer")
    cost = raw.get("cost")
    if (
        isinstance(cost, bool)
        or not isinstance(cost, int | float)
        or not math.isfinite(cost)
        or cost < 0
    ):
        raise fail("cost must be a finite number >= 0")
    raw_output = None
    if keep_raw(index):
        raw_output = raw.get("raw_output")
        if not isinstance(raw_output, str):
            raise fail("raw_output must be a string")
    return _Record(
        index=index,
        prompt=prompt,
        score=float(score),
        prompt_tokens=prompt_tokens,
        completion_tokens=max(completion_tokens, 0),
        clamped=completion_tokens < 0,
        cost=float(cost),
        raw_output=raw_output,
    )
