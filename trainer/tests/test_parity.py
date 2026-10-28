import dataclasses
import json

import numpy as np

from routeforge_train import parity as parity_module
from routeforge_train.features import (
    dense_raw,
    feature_text,
    featurize,
    hashed_features,
    prompt_sha256,
    tokenize,
)
from routeforge_train.parity import PARITY_CASES, parity_rows, write_parity_fixtures
from routeforge_train.policy import export_policy, load_policy, score_policy
from routeforge_train.prices import Price, PriceTable, load_prices
from routeforge_train.routing import (
    REASON_ESCALATED,
    REASON_PASSING,
    expected_cost_usd,
    route_quality,
)
from tests.synth import SYNTH_POOL

ROW_KEYS = {
    "name",
    "messages",
    "text",
    "prompt_sha256",
    "hashed_indices",
    "hashed_values",
    "dense_raw",
    "dense",
    "static_family",
    "policy_version",
    "scores",
    "decisions",
}


def fixture_lines(paths):
    return paths.parity_fixtures.read_text(encoding="utf-8").splitlines()


def test_parity_cases_cover_edge_inputs():
    names = [n for n, _ in PARITY_CASES]
    assert len(names) == len(set(names)) == 20
    assert {
        "empty",
        "crlf",
        "unicode",
        "lone_surrogate",
        "array_content",
        "multi_message",
        "code_fence",
        "long_token",
    } <= set(names)
    assert {"empty_content_list", "punctuation_only", "numbers", "single_word"} <= set(names)


def test_parity_cases_produce_the_expected_feature_texts():
    texts = {name: feature_text(list(messages)) for name, messages in PARITY_CASES}
    assert texts["empty"] == "" and texts["empty_content_list"] == ""
    assert texts["crlf"] == "first\r\nsecond\r\n"
    assert texts["lone_surrogate"] == "bad \ufffd text"
    assert texts["array_content"] == "part one\nPart Two?"
    assert texts["multi_message"] == "You are terse.\nSummarize this.\nSure.\nNow in French."
    assert texts["long_token"] == "a" * 300 + " b"
    assert texts["unicode"] == "Caf\u00e9 na\u00efve \u6771\u4eac emoji \U0001f680 r\u00e9sum\u00e9"
    assert tokenize(texts["punctuation_only"].encode("utf-8")) == []
    assert texts["code_fence"].startswith("```python\n")
    assert texts["uppercase_heavy"] == "URGENT: READ THIS NOW"


def test_parity_cases_are_plain_json_messages():
    for name, messages in PARITY_CASES:
        assert isinstance(messages, tuple) and messages, name
        assert all(set(m) == {"role", "content"} for m in messages), name
        json.dumps(messages, ensure_ascii=True)


def test_parity_rows_match_features_and_reloaded_policy(trained_paths):
    policy = load_policy(trained_paths.policy)
    lines = fixture_lines(trained_paths)
    assert len(lines) == len(PARITY_CASES)
    assert trained_paths.parity_fixtures.read_bytes().isascii()
    for line, (name, messages) in zip(lines, PARITY_CASES, strict=True):
        row = json.loads(line)
        assert set(row) == ROW_KEYS and row["name"] == name
        assert line == json.dumps(
            row, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
        )
        assert row["messages"] == json.loads(json.dumps(messages))
        text = feature_text(list(messages))
        assert row["text"] == text
        assert row["prompt_sha256"] == prompt_sha256(text)
        indices, values = hashed_features(text)
        assert row["hashed_indices"] == indices and row["hashed_values"] == values
        assert row["dense_raw"] == dense_raw(text)
        assert row["dense"] == policy.scaler.transform(np.array([dense_raw(text)]))[0].tolist()
        assert row["static_family"] == policy.static_rules.classify(text)
        assert row["policy_version"] == policy.version
        scores = score_policy(policy, featurize([text]))
        assert [s["model"] for s in row["scores"]] == list(SYNTH_POOL)
        for j, got in enumerate(row["scores"]):
            assert got["raw"] == float(scores.raw[0, j])
            assert got["p"] == float(scores.p[0, j])
            assert got["lcb"] == float(scores.lcb[0, j])
            assert got["block"] == int(scores.block[0, j])
            assert set(got) == {"model", "raw", "p", "lcb", "block"}


def test_parity_lone_surrogate_row_is_ascii_escaped(trained_paths):
    lines = {json.loads(line)["name"]: line for line in fixture_lines(trained_paths)}
    row = json.loads(lines["lone_surrogate"])
    assert row["text"] == "bad \ufffd text"
    assert row["messages"][0]["content"] == "bad \ud800 text"
    assert "\\ud800" in lines["lone_surrogate"] and "\\ufffd" in lines["lone_surrogate"]


def test_parity_decisions_follow_route_quality(trained_paths):
    policy = load_policy(trained_paths.policy)
    prices = load_prices(trained_paths.prices)
    pool = [m.name for m in policy.models]
    eot = {m.name: m.expected_output_tokens for m in policy.models}
    strongest = pool.index(policy.strongest_model)
    seen_reasons = set()
    for line in fixture_lines(trained_paths):
        row = json.loads(line)
        lcb = np.array([[s["lcb"] for s in row["scores"]]])
        cost = expected_cost_usd(np.array([len(row["text"].encode("utf-8"))]), pool, prices, eot)
        assert list(row["decisions"]) == sorted(p.name for p in policy.profiles)
        for profile in policy.profiles:
            chosen, escalated = route_quality(lcb, cost, strongest, profile.lambda_)
            want = {
                "chosen": pool[int(chosen[0])],
                "reason": REASON_ESCALATED if escalated[0] else REASON_PASSING,
            }
            assert row["decisions"][profile.name] == want
            seen_reasons.add(want["reason"])
    assert REASON_ESCALATED in seen_reasons


def test_parity_decisions_break_cost_ties_by_pool_order(lr_policy):
    flat = PriceTable("2026-10-26", "tie", {m: Price(1.0, 1.0) for m in SYNTH_POOL})
    models = tuple(dataclasses.replace(e, expected_output_tokens=100.0) for e in lr_policy.models)
    permissive = tuple(dataclasses.replace(p, lambda_=0.0) for p in lr_policy.profiles)
    policy = dataclasses.replace(lr_policy, models=models, profiles=permissive)
    rows = parity_rows(policy, flat)
    assert len(rows) == len(PARITY_CASES)
    for row in rows:
        for decision in row["decisions"].values():
            assert decision == {"chosen": "m-strong", "reason": REASON_PASSING}


def test_parity_rows_escalate_to_the_strongest_for_the_sentinel_profile(lr_policy, synth_tables):
    sentinel = tuple(dataclasses.replace(p, lambda_=1.01) for p in lr_policy.profiles)
    policy = dataclasses.replace(lr_policy, profiles=sentinel)
    for row in parity_rows(policy, synth_tables.prices):
        for decision in row["decisions"].values():
            assert decision == {"chosen": "m-strong", "reason": REASON_ESCALATED}


def test_write_parity_fixtures_scores_the_reloaded_artifact(
    lr_policy, trained_paths, tmp_path, monkeypatch
):
    policy_path = tmp_path / "policy.json"
    export_policy(lr_policy, policy_path)
    calls = []
    real_load = parity_module.load_policy

    def recording_load(path):
        calls.append(path)
        return real_load(path)

    monkeypatch.setattr(parity_module, "load_policy", recording_load)
    out = tmp_path / "nested" / "features.jsonl"
    assert write_parity_fixtures(policy_path, trained_paths.prices, out) == len(PARITY_CASES)
    assert calls == [policy_path]
    first = out.read_bytes()
    assert first.count(b"\n") == len(PARITY_CASES) and first.endswith(b"\n")
    assert write_parity_fixtures(policy_path, trained_paths.prices, out) == len(PARITY_CASES)
    assert out.read_bytes() == first
    assert [p.name for p in out.parent.iterdir()] == ["features.jsonl"]
