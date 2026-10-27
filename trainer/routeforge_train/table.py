from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from routeforge_train.outcomes import OutcomeRow


@dataclass(frozen=True, eq=False)
class OutcomeTable:
    """Per-prompt outcomes of every pool model, one row per prompt, sorted by prompt_id."""

    pool: tuple[str, ...]
    prompt_ids: tuple[str, ...]
    datasets: np.ndarray  # (n,) str
    families: np.ndarray  # (n,) str
    texts: tuple[str, ...]
    prompt_bytes: np.ndarray  # (n,) int64, UTF-8 length of the prompt
    success: np.ndarray  # (n, M) bool
    cost: np.ndarray  # (n, M) float64, recorded USD
    completion_tokens: np.ndarray  # (n, M) int64

    def __len__(self) -> int:
        return len(self.prompt_ids)

    def subset(self, mask: np.ndarray) -> OutcomeTable:
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != (len(self),):
            raise ValueError(f"mask must be a bool array of shape ({len(self)},)")
        return OutcomeTable(
            pool=self.pool,
            prompt_ids=tuple(p for p, keep in zip(self.prompt_ids, mask, strict=True) if keep),
            datasets=self.datasets[mask],
            families=self.families[mask],
            texts=tuple(t for t, keep in zip(self.texts, mask, strict=True) if keep),
            prompt_bytes=self.prompt_bytes[mask],
            success=self.success[mask],
            cost=self.cost[mask],
            completion_tokens=self.completion_tokens[mask],
        )


def pivot(rows: Iterable[OutcomeRow], pool: Sequence[str]) -> OutcomeTable:
    """One table row per prompt; every prompt needs exactly one row per pool model."""
    pool = tuple(pool)
    column = {model: j for j, model in enumerate(pool)}
    by_prompt: dict[str, list[OutcomeRow | None]] = {}
    for row in rows:
        if row.model not in column:
            raise ValueError(f"{row.prompt_id}: model {row.model!r} is not in the pool")
        slots = by_prompt.setdefault(row.prompt_id, [None] * len(pool))
        if slots[column[row.model]] is not None:
            raise ValueError(f"{row.prompt_id}: more than one row for model {row.model!r}")
        slots[column[row.model]] = row

    prompt_ids = tuple(sorted(by_prompt))
    n, m = len(prompt_ids), len(pool)
    success = np.zeros((n, m), dtype=np.bool_)
    cost = np.zeros((n, m), dtype=np.float64)
    completion = np.zeros((n, m), dtype=np.int64)
    datasets, families, texts = [], [], []
    for i, prompt_id in enumerate(prompt_ids):
        filled: list[OutcomeRow] = []
        for j, slot in enumerate(by_prompt[prompt_id]):
            if slot is None:
                raise ValueError(f"{prompt_id}: no row for model {pool[j]!r}")
            success[i, j] = slot.success
            cost[i, j] = slot.cost_usd
            completion[i, j] = slot.completion_tokens
            filled.append(slot)
        first = filled[0]
        datasets.append(first.dataset)
        families.append(first.family)
        texts.append(first.prompt)
    return OutcomeTable(
        pool=pool,
        prompt_ids=prompt_ids,
        datasets=np.array(datasets, dtype=np.str_),
        families=np.array(families, dtype=np.str_),
        texts=tuple(texts),
        prompt_bytes=np.array([len(t.encode("utf-8")) for t in texts], dtype=np.int64),
        success=success,
        cost=cost,
        completion_tokens=completion,
    )
