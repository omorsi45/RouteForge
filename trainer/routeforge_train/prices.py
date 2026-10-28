import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from routeforge_train.errors import TrainerError

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class Price:
    """List price in USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float


@dataclass(frozen=True)
class PriceTable:
    effective_date: str
    source: str
    models: dict[str, Price]

    def price(self, model: str) -> Price:
        try:
            return self.models[model]
        except KeyError:
            raise TrainerError(f"no list price for model {model!r}") from None


def list_cost_usd(price: Price, prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * price.input_per_mtok + completion_tokens * price.output_per_mtok) / 1e6


def load_prices(path: Path) -> PriceTable:
    """Read and validate a price table; every problem is a TrainerError naming the file."""
    path = Path(path)
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise TrainerError(f"{path}: cannot read prices: {exc}") from exc

    def fail(problem: str) -> TrainerError:
        return TrainerError(f"{path}: {problem}")

    if not isinstance(doc, dict):
        raise fail("expected a mapping with effective_date, source and models")
    effective_date = doc.get("effective_date")
    if not isinstance(effective_date, str) or not _DATE.fullmatch(effective_date):
        raise fail("effective_date must be a quoted YYYY-MM-DD string")
    try:
        date.fromisoformat(effective_date)
    except ValueError:
        raise fail(f"effective_date {effective_date!r} is not a calendar date") from None
    source = doc.get("source")
    if not isinstance(source, str) or not source:
        raise fail("source must be a non-empty string")
    entries = doc.get("models")
    if not isinstance(entries, list) or not entries:
        raise fail("models must be a non-empty list")
    models: dict[str, Price] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise fail(f"models[{position}] must be a mapping")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise fail(f"models[{position}].name must be a non-empty string")
        if name in models:
            raise fail(f"duplicate model {name!r}")
        models[name] = Price(
            _rate(entry, "input_per_mtok", name, fail),
            _rate(entry, "output_per_mtok", name, fail),
        )
    return PriceTable(effective_date, source, models)


def _rate(entry: dict[str, Any], key: str, name: str, fail: Any) -> float:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise fail(f"{name}.{key} must be a number")
    if not math.isfinite(value) or value < 0:
        raise fail(f"{name}.{key} must be finite and >= 0")
    return float(value)
