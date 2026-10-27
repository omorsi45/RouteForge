import math

import pytest

from routeforge_train.jsonio import sha256_file
from routeforge_train.labels import label_rule
from routeforge_train.outcomes import FIELDS, OutcomeRow, read_outcomes, write_outcomes


def make_row(**change) -> dict:
    row = {
        "prompt_id": "aime:1",
        "dataset": "aime",
        "family": "reasoning",
        "prompt": "What is 2+2?",
        "model": "gpt-5",
        "score": 1.0,
        "success": True,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost_usd": 0.001,
    }
    row.update(change)
    return row


def test_outcome_row_round_trip():
    row = OutcomeRow.from_json(make_row())
    assert OutcomeRow.from_json(row.to_json()) == row
    assert list(row.to_json()) == list(FIELDS)


def test_outcome_row_cost_int_is_stored_as_float():
    row = OutcomeRow.from_json(make_row(cost_usd=0))
    assert isinstance(row.cost_usd, float)


def test_outcome_row_allows_empty_prompt():
    assert OutcomeRow.from_json(make_row(prompt="")).prompt == ""


@pytest.mark.parametrize(
    "change",
    [
        {"extra": 1},
        {"score": float("nan")},
        {"score": 1.5},
        {"prompt_tokens": True},
        {"completion_tokens": -1},
        {"cost_usd": -0.1},
        {"success": 1},
        {"model": ""},
    ],
)
def test_outcome_row_strict_parsing(change):
    with pytest.raises(ValueError):
        OutcomeRow.from_json(make_row(**change))


def test_outcome_row_missing_field_names_it():
    d = make_row()
    del d["family"]
    with pytest.raises(ValueError):
        OutcomeRow.from_json(d)


def test_outcome_row_bad_field_is_named():
    with pytest.raises(ValueError, match="score"):
        OutcomeRow.from_json(make_row(score=math.inf))


def test_write_and_read_outcomes(tmp_path):
    rows = [
        OutcomeRow.from_json(make_row()),
        OutcomeRow.from_json(
            make_row(prompt_id="aime:2", model="claude-sonnet-4", score=0.0, success=False)
        ),
    ]
    path = tmp_path / "work" / "outcomes.jsonl"
    info = write_outcomes(rows, path, tmp_path)
    assert info.rows == 2
    assert info.path == "work/outcomes.jsonl"
    assert info.sha256 == sha256_file(path)
    assert list(read_outcomes(path)) == rows


def test_label_rule_binary_half_and_invalid():
    assert label_rule([0.0, 1.0, 1.0]).describe() == "score >= 1.0"
    assert label_rule([0.0]).threshold == 1.0
    r = label_rule([0.0, 0.5, 1.0])
    assert r.describe() == "score >= 0.5"
    assert r.is_success(0.5)
    assert not r.is_success(0.4)
    assert r.score_values == (0.0, 0.5, 1.0)
    assert label_rule([0.3]).threshold == 0.5
    with pytest.raises(ValueError):
        label_rule([])
    with pytest.raises(ValueError):
        label_rule([1.2])
    with pytest.raises(ValueError):
        label_rule([float("nan")])


def test_label_rule_binary_requires_full_score():
    r = label_rule([0.0, 1.0])
    assert r.is_success(1.0)
    assert not r.is_success(0.5)
