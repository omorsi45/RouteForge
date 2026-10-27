"""Parity fixtures for the Go gateway (spec 8.1).

Each line pins one synthetic request: its feature text, hashed and dense features, the exported
policy's per-model scores and the routing decision of every profile. Scores always come from
the exported artifact reloaded from disk, never from in-memory models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from routeforge_train.features import (
    dense_raw,
    feature_text,
    featurize,
    hashed_features,
    prompt_sha256,
)
from routeforge_train.jsonio import atomic_writer
from routeforge_train.policy import Policy, load_policy, score_policy
from routeforge_train.prices import PriceTable, load_prices
from routeforge_train.routing import (
    REASON_ESCALATED,
    REASON_PASSING,
    expected_cost_usd,
    route_quality,
)


def _user(content: object) -> dict[str, object]:
    return {"role": "user", "content": content}


PARITY_CASES: tuple[tuple[str, tuple[dict, ...]], ...] = (
    ("empty", (_user(""),)),
    ("empty_content_list", (_user([]),)),
    ("single_word", (_user("Hello"),)),
    ("mixed_case_digits", (_user("What is 2+2? Answer in ONE word."),)),
    ("repeated_tokens", (_user("the the the cat the"),)),
    ("code_fence", (_user("```python\nprint('hi')\n```\nWhy does this fail?"),)),
    ("multiline", (_user("line one\nline two\n\nline four"),)),
    ("crlf", (_user("first\r\nsecond\r\n"),)),
    ("unicode", (_user("Caf\u00e9 na\u00efve \u6771\u4eac emoji \U0001f680 r\u00e9sum\u00e9"),)),
    ("lone_surrogate", (_user("bad \ud800 text"),)),
    ("punctuation_only", (_user("?!?... ---"),)),
    ("long_token", (_user("a" * 300 + " b"),)),
    (
        "array_content",
        (_user([{"type": "text", "text": "part one"}, {"type": "text", "text": "Part Two?"}]),),
    ),
    (
        "multi_message",
        (
            {"role": "system", "content": "You are terse."},
            _user("Summarize this."),
            {"role": "assistant", "content": "Sure."},
            _user("Now in French."),
        ),
    ),
    ("numbers", (_user("3.14159 2026-10-26 1e10 0x1F"),)),
    ("math", (_user("Let $x^2 + 3x - 4 = 0$. Find all real roots and explain each step."),)),
    (
        "code_task",
        (
            _user(
                "Write a Python function that returns the n-th Fibonacci number.\n"
                "```\ndef fib(n):\n    pass\n```"
            ),
        ),
    ),
    ("qa", (_user("Who wrote the novel that introduced the detective Hercule Poirot?"),)),
    (
        "instruction",
        (_user("Draft a polite email to reschedule our meeting to Thursday at 3 PM."),),
    ),
    ("uppercase_heavy", (_user("URGENT: READ THIS NOW"),)),
)


def parity_rows(policy: Policy, prices: PriceTable) -> list[dict[str, Any]]:
    """One row per case, scored by `policy`; scores are computed one prompt at a time."""
    names = [entry.name for entry in policy.models]
    expected = {entry.name: entry.expected_output_tokens for entry in policy.models}
    strongest = names.index(policy.strongest_model)
    rows: list[dict[str, Any]] = []
    for name, messages in PARITY_CASES:
        text = feature_text(list(messages))
        indices, values = hashed_features(text)
        raw = dense_raw(text)
        scores = score_policy(policy, featurize([text]))
        exp_cost = expected_cost_usd(
            np.array([len(text.encode("utf-8"))], dtype=np.int64), names, prices, expected
        )
        decisions = {}
        for profile in policy.profiles:
            chosen, escalated = route_quality(scores.lcb, exp_cost, strongest, profile.lambda_)
            decisions[profile.name] = {
                "chosen": names[int(chosen[0])],
                "reason": REASON_ESCALATED if escalated[0] else REASON_PASSING,
            }
        rows.append(
            {
                "name": name,
                "messages": [dict(message) for message in messages],
                "text": text,
                "prompt_sha256": prompt_sha256(text),
                "hashed_indices": indices,
                "hashed_values": values,
                "dense_raw": raw,
                "dense": policy.scaler.transform(np.array([raw]))[0].tolist(),
                "static_family": policy.static_rules.classify(text),
                "policy_version": policy.version,
                "scores": [
                    {
                        "model": model,
                        "raw": float(scores.raw[0, j]),
                        "p": float(scores.p[0, j]),
                        "lcb": float(scores.lcb[0, j]),
                        "block": int(scores.block[0, j]),
                    }
                    for j, model in enumerate(names)
                ],
                "decisions": decisions,
            }
        )
    return rows


def write_parity_fixtures(policy_path: Path, prices_path: Path, out: Path) -> int:
    """Reload the exported policy and write one ASCII JSON line per parity case."""
    policy = load_policy(policy_path)
    rows = parity_rows(policy, load_prices(prices_path))
    with atomic_writer(out) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
    return len(rows)
