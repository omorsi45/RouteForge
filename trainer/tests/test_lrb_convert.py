import json

import pytest

from routeforge_train.errors import TrainerError
from routeforge_train.features import prompt_sha256
from routeforge_train.labels import LabelRule, label_rule
from routeforge_train.lrb_convert import (
    convert_llmrouterbench,
    export_replay,
    member_dataset_model,
)
from routeforge_train.outcomes import read_outcomes
from tests.lrb_fixture import record, write_member

AIME = "bench-release/aime/hybrid/{}/aime-hybrid-{}-1.json"
ARENA = "bench-release/arenahard/{}/arenahard-test-{}-1.json"
HLE = "bench-release/hle/test/{}/hle-test-{}-1.json"


def member(template: str, model: str) -> str:
    return template.format(model, model)


def test_member_dataset_model():
    assert member_dataset_model("bench-release/aime/hybrid/gpt-5/aime-hybrid-gpt-5-1.json") == (
        "aime",
        "gpt-5",
    )
    assert member_dataset_model("bench-release/arenahard/gpt-5/arenahard-test-gpt-5-1.json") == (
        "arenahard",
        "gpt-5",
    )
    with pytest.raises(ValueError):
        member_dataset_model("bench-release/tau2/test/gpt-5/x.json")
    with pytest.raises(ValueError):
        member_dataset_model("other/aime/hybrid/gpt-5/x.json")


def test_convert_writes_schema_rows_in_order(tmp_path):
    raw = tmp_path / "raw"
    gpt_member = member(AIME, "gpt-5")
    qwen = member(AIME, "qwen3-235b-a22b-2507")
    write_member(raw, gpt_member, [record(2, "p2", 1.0, cost=1), record(1, "p1", 0.0)])
    write_member(raw, qwen, [record(2, "p2", 0.0), record(1, "p1", 1.0)])
    out = tmp_path / "work" / "outcomes.jsonl"
    # members passed in reverse pool order; output must follow pool order
    stats = convert_llmrouterbench(raw, [qwen, gpt_member], out, tmp_path)
    rows = list(read_outcomes(out))
    assert [(r.model, r.prompt_id) for r in rows] == [
        ("gpt-5", "aime:1"),
        ("gpt-5", "aime:2"),
        ("qwen3-235b-a22b-2507", "aime:1"),
        ("qwen3-235b-a22b-2507", "aime:2"),
    ]
    assert {r.family for r in rows} == {"reasoning"}
    assert {r.dataset for r in rows} == {"aime"}
    assert all(isinstance(r.cost_usd, float) for r in rows)
    assert rows[1].prompt == "p2"
    assert stats.rows == 4
    assert stats.records == {("aime", "gpt-5"): 2, ("aime", "qwen3-235b-a22b-2507"): 2}


def test_convert_orders_datasets_by_family_table(tmp_path):
    raw = tmp_path / "raw"
    arena = member(ARENA, "gpt-5")
    aime = member(AIME, "gpt-5")
    write_member(raw, arena, [record(1, "q", 0.5)])
    write_member(raw, aime, [record(1, "p", 1.0)])
    out = tmp_path / "outcomes.jsonl"
    convert_llmrouterbench(raw, [arena, aime], out, tmp_path)
    assert [r.dataset for r in read_outcomes(out)] == ["aime", "arenahard"]


def test_convert_label_rule_binary_and_half(tmp_path):
    raw = tmp_path / "raw"
    aime = member(AIME, "gpt-5")
    arena = member(ARENA, "gpt-5")
    write_member(raw, aime, [record(1, "p1", 0.0), record(2, "p2", 1.0)])
    write_member(raw, arena, [record(1, "q1", 0.0), record(2, "q2", 0.5), record(3, "q3", 1.0)])
    out = tmp_path / "outcomes.jsonl"
    stats = convert_llmrouterbench(raw, [aime, arena], out, tmp_path)
    rows = {r.prompt_id: r for r in read_outcomes(out)}
    assert [rows[f"aime:{i}"].success for i in (1, 2)] == [False, True]
    assert [rows[f"arenahard:{i}"].success for i in (1, 2, 3)] == [False, True, True]
    assert stats.label_rules["aime"].describe() == "score >= 1.0"
    assert stats.label_rules["arenahard"].describe() == "score >= 0.5"


def test_convert_clamps_negative_completion_tokens(tmp_path):
    raw = tmp_path / "raw"
    hle = member(HLE, "gpt-5")
    write_member(
        raw,
        hle,
        [
            record(1, "p1", 0.0, prompt_tokens=0, completion_tokens=-52, cost=0),
            record(2, "p2", 1.0),
        ],
    )
    out = tmp_path / "outcomes.jsonl"
    stats = convert_llmrouterbench(raw, [hle], out, tmp_path)
    rows = list(read_outcomes(out))
    assert [r.completion_tokens for r in rows] == [0, 5]
    assert stats.clamped == {("hle", "gpt-5"): 1}
    assert stats.to_json()["usage_clamped"] == [
        {"dataset": "hle", "model": "gpt-5", "completion_tokens_clamped": 1}
    ]


def test_convert_stats_to_json(tmp_path):
    raw = tmp_path / "raw"
    gpt_member = member(AIME, "gpt-5")
    sonnet_member = member(AIME, "claude-sonnet-4")
    write_member(raw, gpt_member, [record(1, "p", 1.0)])
    write_member(raw, sonnet_member, [record(1, "p", 0.0)])
    stats = convert_llmrouterbench(raw, [sonnet_member, gpt_member], tmp_path / "o.jsonl", tmp_path)
    assert stats.to_json() == {
        "rows": 2,
        "records": [
            {"dataset": "aime", "model": "gpt-5", "records": 1},
            {"dataset": "aime", "model": "claude-sonnet-4", "records": 1},
        ],
        "usage_clamped": [],
        "label_rules": {"aime": {"rule": "score >= 1.0", "score_values": [0.0, 1.0]}},
    }


def test_convert_rejects_model_name_mismatch(tmp_path):
    raw = tmp_path / "raw"
    gpt_member = member(AIME, "gpt-5")
    write_member(raw, gpt_member, [record(1, "p", 1.0)], model="gpt-4")
    with pytest.raises(TrainerError, match="aime-hybrid-gpt-5-1.json") as excinfo:
        convert_llmrouterbench(raw, [gpt_member], tmp_path / "o.jsonl", tmp_path)
    assert "gpt-4" in str(excinfo.value)


def test_convert_rejects_dataset_name_mismatch(tmp_path):
    raw = tmp_path / "raw"
    gpt_member = member(AIME, "gpt-5")
    write_member(raw, gpt_member, [record(1, "p", 1.0)], dataset="gpqa")
    with pytest.raises(TrainerError, match="aime-hybrid-gpt-5-1.json") as excinfo:
        convert_llmrouterbench(raw, [gpt_member], tmp_path / "o.jsonl", tmp_path)
    assert "gpqa" in str(excinfo.value)


@pytest.mark.parametrize(
    "bad",
    [{"score": 1.5}, {"cost": -0.01}, {"prompt": 7}, {"prompt_tokens": -3}],
)
def test_convert_rejects_bad_records(tmp_path, bad):
    raw = tmp_path / "raw"
    gpt_member = member(AIME, "gpt-5")
    good = record(1, "p1", 1.0)
    broken = {**record(2, "p2", 1.0), **bad}
    write_member(raw, gpt_member, [good, broken])
    with pytest.raises(ValueError, match=r"aime-hybrid-gpt-5-1\.json.*index 2"):
        convert_llmrouterbench(raw, [gpt_member], tmp_path / "o.jsonl", tmp_path)


def test_convert_rejects_lone_surrogate_prompt(tmp_path):
    raw = tmp_path / "raw"
    gpt_member = member(AIME, "gpt-5")
    write_member(raw, gpt_member, [record(1, "bad \ud800 prompt", 1.0)])
    with pytest.raises(ValueError, match="index 1"):
        convert_llmrouterbench(raw, [gpt_member], tmp_path / "o.jsonl", tmp_path)


def test_convert_rejects_duplicate_index(tmp_path):
    raw = tmp_path / "raw"
    gpt_member = member(AIME, "gpt-5")
    write_member(raw, gpt_member, [record(1, "p", 1.0), record(1, "p", 0.0)])
    with pytest.raises(ValueError, match="duplicate"):
        convert_llmrouterbench(raw, [gpt_member], tmp_path / "o.jsonl", tmp_path)


def test_convert_output_has_no_model_output_text(tmp_path):
    raw = tmp_path / "raw"
    gpt_member = member(AIME, "gpt-5")
    arena = member(ARENA, "gpt-5")
    write_member(
        raw,
        gpt_member,
        [record(1, "p", 1.0, prediction="SECRET-PREDICTION", raw_output="SECRET-RAW")],
    )
    write_member(
        raw,
        arena,
        [
            record(
                1,
                "q",
                0.5,
                prediction=[{"role": "assistant", "content": "SECRET-PREDICTION"}],
                raw_output="SECRET-RAW",
            )
        ],
    )
    out = tmp_path / "outcomes.jsonl"
    convert_llmrouterbench(raw, [gpt_member, arena], out, tmp_path)
    data = out.read_bytes()
    assert b"SECRET-PREDICTION" not in data
    assert b"SECRET-RAW" not in data


def test_export_replay_writes_raw_output_for_test_prompts_only(tmp_path):
    raw = tmp_path / "raw"
    members = []
    for model in ("gpt-5", "qwen3-235b-a22b-2507"):
        m = member(AIME, model)
        members.append(m)
        write_member(
            raw,
            m,
            [
                record(1, "prompt one", 0.0, raw_output=f"{model}-raw-1"),
                record(2, "prompt two", 1.0, raw_output=f"{model}-raw-2", completion_tokens=-4),
            ],
        )
    arena_members = []
    for model in ("gpt-5", "qwen3-235b-a22b-2507"):
        m = member(ARENA, model)
        arena_members.append(m)
        write_member(
            raw,
            m,
            [
                record(
                    1,
                    "arena q",
                    0.5,
                    prediction=[
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "x"},
                    ],
                    raw_output=f"{model}-arena-raw",
                )
            ],
        )
    rules = {
        "aime": label_rule([0.0, 1.0]),
        "arenahard": LabelRule(0.5, (0.0, 0.5, 1.0)),
    }
    out = tmp_path / "replay" / "replay.jsonl"
    info = export_replay(
        raw, [*members, *arena_members], {"aime:2", "arenahard:1"}, rules, out, tmp_path
    )
    assert info.rows == 4
    assert info.path == "replay/replay.jsonl"
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [(r["model"], r["prompt_id"]) for r in rows] == [
        ("gpt-5", "aime:2"),
        ("qwen3-235b-a22b-2507", "aime:2"),
        ("gpt-5", "arenahard:1"),
        ("qwen3-235b-a22b-2507", "arenahard:1"),
    ]
    assert rows[0]["text"] == "gpt-5-raw-2"
    assert rows[0]["prompt_sha256"] == prompt_sha256("prompt two")
    assert rows[0]["completion_tokens"] == 0
    assert rows[0]["success"] is True
    assert rows[2]["text"] == "gpt-5-arena-raw"
    assert rows[2]["success"] is True
    assert rows[2]["dataset"] == "arenahard"
    assert set(rows[0]) == {
        "prompt_sha256",
        "model",
        "prompt_id",
        "dataset",
        "text",
        "prompt_tokens",
        "completion_tokens",
        "cost_usd",
        "score",
        "success",
    }
    assert b"raw-1" not in out.read_bytes()


def test_export_replay_rejects_non_string_raw_output(tmp_path):
    raw = tmp_path / "raw"
    gpt_member = member(AIME, "gpt-5")
    write_member(raw, gpt_member, [record(1, "p", 1.0, raw_output=None)])
    with pytest.raises(ValueError, match="raw_output"):
        export_replay(
            raw,
            [gpt_member],
            {"aime:1"},
            {"aime": label_rule([1.0])},
            tmp_path / "r.jsonl",
            tmp_path,
        )


def test_export_replay_ignores_bad_raw_output_of_non_test_prompts(tmp_path):
    raw = tmp_path / "raw"
    gpt_member = member(AIME, "gpt-5")
    write_member(raw, gpt_member, [record(1, "p1", 1.0, raw_output=None), record(2, "p2", 1.0)])
    info = export_replay(
        raw, [gpt_member], {"aime:2"}, {"aime": label_rule([1.0])}, tmp_path / "r.jsonl", tmp_path
    )
    assert info.rows == 1
