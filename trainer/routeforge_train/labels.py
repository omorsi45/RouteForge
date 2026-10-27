import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class LabelRule:
    """Per-dataset success rule: a record succeeds when its score reaches `threshold`."""

    threshold: float
    score_values: tuple[float, ...]

    def describe(self) -> str:
        return f"score >= {self.threshold}"

    def is_success(self, score: float) -> bool:
        return score >= self.threshold


def label_rule(scores: Iterable[float]) -> LabelRule:
    """Binary scores need a full score; graded scores count from one half."""
    values = sorted(set(scores))
    if not values:
        raise ValueError("label_rule needs at least one score")
    for value in values:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"score {value!r} is outside [0, 1]")
    threshold = 1.0 if set(values) <= {0.0, 1.0} else 0.5
    return LabelRule(threshold=threshold, score_values=tuple(values))
