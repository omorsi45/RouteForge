import io
import json
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest

from routeforge_train import download, extraction
from routeforge_train.cli import main
from routeforge_train.errors import TrainerError
from routeforge_train.extraction import (
    DATASET,
    FAMILY,
    PROMPT_TEMPLATE,
    TEXT_JSON,
    BuildStats,
    ChatResult,
    ExtractionRequest,
    build_requests,
    grade,
    json_equal,
    label_requests,
    ollama_chat,
    run_label_extraction,
)
from routeforge_train.outcomes import read_outcomes
from routeforge_train.paths import Paths
from routeforge_train.report_md import GUARANTEE_NOTE
from tests.archive import opener_for, remote_for

TRAINER_DIR = Path(__file__).resolve().parent.parent
OLLAMA_URL = "http://127.0.0.1:11434"
EXTRACTION_STATUS = (
    "Extraction quality is unmeasured until `make label-extraction` has been run against local "
    "Ollama models; no extraction results are reported."
)


def line(text_input, schema, output):
    body = f"### Input: {text_input}\n### Schema: {schema}\n### Output: {output}"
    return json.dumps({"text": body}).encode()


GOOD = [
    line("Paid 5 dollars", '{"amount": "number"}', '{"amount": 5}'),
    line("Ann is 31", '{"name": "string", "age": "number"}', '{"name": "Ann", "age": 31}'),
    line("Open on Mon", '{"days": ["string"]}', '{"days": ["Mon"]}'),
]


def plenty(_path):
    return 10**12


def write_source(tmp_path, lines, *, trailing_newline=True):
    path = tmp_path / "dataset.jsonl"
    path.write_bytes(b"\n".join(lines) + (b"\n" if trailing_newline else b""))
    return path


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(url, *args, **kwargs):
        raise AssertionError(f"unexpected network access to {url}")

    monkeypatch.setattr(download, "open_url", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def test_constants_match_the_verified_facts():
    assert TEXT_JSON.url == (
        "https://huggingface.co/datasets/suneeldk/text-json/resolve/"
        "d0474144541e250edffa3d25676798028e8f6bb3/dataset.jsonl"
    )
    assert TEXT_JSON.size == 1_128_379
    assert TEXT_JSON.sha256 == "b439561eb8438dfaaa654fc835d3e69946360207396f25e2cc8e9b0922aac121"
    assert (DATASET, FAMILY) == ("text-json", "extraction")
    assert PROMPT_TEMPLATE == (
        "Extract the information from the text into a JSON object that follows the schema. "
        "Reply with the JSON object only.\n\nSchema: {schema}\n\nText: {text}"
    )


def test_build_requests_parses_sections_and_skips_invalid(tmp_path):
    bad_utf8 = line("Paid 5 POUND", '{"amount": "number"}', '{"amount": 5}')
    bad_utf8 = bad_utf8.replace(b"POUND", b"\xa3")  # a cp1252 pound sign, not valid UTF-8
    no_output = json.dumps({"text": "### Input: x\n### Schema: {}"}).encode()
    src = write_source(tmp_path, [GOOD[0], GOOD[1], bad_utf8, no_output, GOOD[2]])
    requests, stats = build_requests(src)
    assert stats == BuildStats(rows=3, skipped_invalid_utf8=1, skipped_malformed=1)
    assert [r.prompt_id for r in requests] == ["text-json:1", "text-json:2", "text-json:5"]
    assert [r.gold for r in requests] == [
        {"amount": 5},
        {"name": "Ann", "age": 31},
        {"days": ["Mon"]},
    ]
    assert all(isinstance(r, ExtractionRequest) for r in requests)


def test_build_requests_skips_empty_lines_and_keeps_physical_line_numbers(tmp_path):
    src = write_source(tmp_path, [GOOD[0], b"", b"   ", GOOD[1]], trailing_newline=False)
    requests, stats = build_requests(src)
    assert stats == BuildStats(rows=2, skipped_invalid_utf8=0, skipped_malformed=0)
    assert [r.prompt_id for r in requests] == ["text-json:1", "text-json:4"]


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(b"not json at all", id="not-json"),
        pytest.param(b'["### Input: a"]', id="line-not-an-object"),
        pytest.param(b'{"other": 1}', id="no-text-field"),
        pytest.param(b'{"text": 5}', id="text-not-a-string"),
        pytest.param(line("a", "{}", "not json"), id="output-not-json"),
        pytest.param(line("a", "{}", "[1, 2]"), id="output-not-an-object"),
        pytest.param(
            json.dumps({"text": "### Schema: {}\n### Output: {}"}).encode(), id="no-input-marker"
        ),
        pytest.param(
            json.dumps({"text": "### Input: a\n### Output: {}"}).encode(), id="no-schema-marker"
        ),
        pytest.param(
            json.dumps({"text": "### Input: a\n### Schema: {}\n### Output:{}"}).encode(),
            id="output-marker-without-space",
        ),
    ],
)
def test_build_requests_counts_malformed_lines(tmp_path, bad):
    requests, stats = build_requests(write_source(tmp_path, [bad, GOOD[0]]))
    assert stats == BuildStats(rows=1, skipped_invalid_utf8=0, skipped_malformed=1)
    assert [r.prompt_id for r in requests] == ["text-json:2"]


def test_build_requests_keeps_multiline_input_text(tmp_path):
    src = write_source(tmp_path, [line("first\nsecond", '{"k": "string"}', '{"k": "v"}')])
    (request,), _ = build_requests(src)
    assert request.prompt == PROMPT_TEMPLATE.format(schema='{"k": "string"}', text="first\nsecond")


def test_prompt_template_exact(tmp_path):
    src = write_source(tmp_path, [GOOD[0]])
    (request,), _ = build_requests(src)
    assert request.prompt == PROMPT_TEMPLATE.format(
        schema='{"amount": "number"}', text="Paid 5 dollars"
    )


@pytest.mark.parametrize(
    "pred,score",
    [
        ('{"a": 1, "b": "x"}', 1.0),
        ('{"a": 1.0, "b": "x"}', 1.0),
        ('{"a": 1, "b": "y"}', 0.0),
        ('{"a": 1}', 0.0),
        ('{"a": 1, "b": "x", "c": 3}', 1.0),
        ('```json\n{"a": 1, "b": "x"}\n```', 1.0),
        ('```\n{"a": 1, "b": "x"}\n```', 1.0),
        ('  \n```json\n{"a": 1, "b": "x"}\n```  \n', 1.0),
        ('```json\n{"a": 1, "b": "x"}', 0.0),
        ("not json", 0.0),
        ("", 0.0),
        ("[1, 2]", 0.0),
        ('"x"', 0.0),
        ('{"a": true, "b": "x"}', 0.0),
        ('{"a": 1, "b": null}', 0.0),
        ('{"a": "1", "b": "x"}', 0.0),
        ("[" * 100_000, 0.0),
    ],
)
def test_grade_exact_fields(pred, score):
    assert grade(pred, {"a": 1, "b": "x"}) == score


def test_grade_requires_every_gold_key_even_when_null():
    assert grade('{"a": null}', {"a": None}) == 1.0
    assert grade("{}", {"a": None}) == 0.0


def test_grade_nested_values_are_exact_and_ordered():
    gold = {"x": [1, {"y": None}], "z": {"p": True}}
    assert grade('{"z": {"p": true}, "x": [1, {"y": null}]}', gold) == 1.0
    assert grade('{"x": [{"y": null}, 1], "z": {"p": true}}', gold) == 0.0
    assert grade('{"x": [1, {"y": null}], "z": {"p": 1}}', gold) == 0.0


def test_json_equal_lists_are_ordered():
    assert not json_equal([1, 2], [2, 1]) and json_equal(
        {"x": [1, {"y": None}]}, {"x": [1, {"y": None}]}
    )


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 1.0, True),
        (2, 2.5, False),
        (True, 1, False),
        (1, True, False),
        (False, 0, False),
        (0.0, False, False),
        (True, True, True),
        (True, False, False),
        (None, None, True),
        (None, 0, False),
        (None, False, False),
        ("1", 1, False),
        ("a", "a", True),
        ("a", "b", False),
        ([True], [1], False),
        ([1, 2], [1, 2, 3], False),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
        ({"a": 1}, {"a": 1, "b": 2}, False),
        ({"a": 1, "b": 2}, {"a": 1}, False),
        ({"a": 1}, {"b": 1}, False),
        ({"a": 1}, [1], False),
        ([], {}, False),
    ],
)
def test_json_equal_cases(a, b, expected):
    assert json_equal(a, b) is expected
    assert json_equal(b, a) is expected


def test_label_requests_with_fake_chat():
    requests = [
        ExtractionRequest("text-json:1", "p1", {"a": 1, "b": "x"}),
        ExtractionRequest("text-json:2", "p2", {"a": 2}),
    ]
    calls = []

    def chat(model, prompt):
        calls.append((model, prompt))
        return ChatResult('{"a": 1, "b": "x"}', 12, 7)

    rows = label_requests(requests, ["m1", "m2"], chat)
    # one model at a time, so a local server loads each model once
    assert calls == [("m1", "p1"), ("m1", "p2"), ("m2", "p1"), ("m2", "p2")]
    assert [(r.model, r.prompt_id, r.score, r.success) for r in rows] == [
        ("m1", "text-json:1", 1.0, True),
        ("m1", "text-json:2", 0.0, False),
        ("m2", "text-json:1", 1.0, True),
        ("m2", "text-json:2", 0.0, False),
    ]
    assert [r.prompt for r in rows] == ["p1", "p2", "p1", "p2"]
    assert all(r.dataset == DATASET and r.family == FAMILY for r in rows)
    assert all(r.prompt_tokens == 12 and r.completion_tokens == 7 for r in rows)
    assert all(r.cost_usd == 0.0 for r in rows)


class FakeResponse(io.BytesIO):
    """In-memory HTTP body usable as a context manager."""


def ollama_reply(**fields):
    body = {
        "model": "m",
        "message": {"content": '{"a": 1}'},
        "done": True,
        "prompt_eval_count": 12,
        "eval_count": 7,
    }
    body.update(fields)
    return FakeResponse(json.dumps(body).encode())


def test_ollama_chat_request_shape(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return ollama_reply()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = ollama_chat()("llama3.2:3b", "the prompt")
    assert result == ChatResult('{"a": 1}', 12, 7)
    request = captured["request"]
    assert request.full_url == f"{OLLAMA_URL}/api/chat"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data) == {
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": "the prompt"}],
        "stream": False,
        "options": {"temperature": 0, "seed": 20261026},
    }
    assert captured["timeout"] == 300.0


def test_ollama_chat_uses_given_base_url_and_timeout(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return ollama_reply()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ollama_chat("http://gpu-box:9999/", timeout=5.0)("m", "p")
    assert captured == {"url": "http://gpu-box:9999/api/chat", "timeout": 5.0}


def test_ollama_chat_treats_absent_token_counts_as_zero(monkeypatch):
    # Ollama omits prompt_eval_count when the whole prompt was served from its cache.
    reply = {"message": {"content": "{}"}, "done": True}
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda r, timeout=None: FakeResponse(json.dumps(reply).encode())
    )
    assert ollama_chat()("m", "p") == ChatResult("{}", 0, 0)


def test_ollama_unreachable_is_a_clear_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(TrainerError) as info:
        ollama_chat()("m", "p")
    message = str(info.value)
    assert f"cannot reach Ollama at {OLLAMA_URL}" in message
    assert "Connection refused" in message
    assert message.endswith("start it with: ollama serve")


def test_ollama_missing_model_is_a_clear_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        body = io.BytesIO(b'{"error":"model \'nope:1b\' not found"}')
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", Message(), body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(TrainerError) as info:
        ollama_chat()("nope:1b", "p")
    message = str(info.value)
    assert message.startswith("Ollama 404 for model nope:1b: ")
    assert "model 'nope:1b' not found" in message


def test_ollama_timeout_is_a_clear_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(TrainerError) as info:
        ollama_chat(timeout=2.0)("some-model", "p")
    assert str(info.value) == "Ollama timed out after 2 s for model some-model"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"<html>bad gateway</html>", id="not-json"),
        pytest.param(b"[]", id="not-an-object"),
        pytest.param(b'{"done": true}', id="no-message"),
        pytest.param(b'{"message": {"done": true}}', id="no-content"),
        pytest.param(b'{"message": {"content": 5}}', id="content-not-a-string"),
        pytest.param(b'{"message": {"content": "x"}, "eval_count": "7"}', id="count-not-an-int"),
        pytest.param(b'{"message": {"content": "x"}, "eval_count": -1}', id="negative-count"),
    ],
)
def test_ollama_unexpected_response_is_a_clear_error(monkeypatch, body):
    monkeypatch.setattr(urllib.request, "urlopen", lambda r, timeout=None: FakeResponse(body))
    with pytest.raises(TrainerError, match="unexpected response for model some-model"):
        ollama_chat()("some-model", "p")


def serve(lines=GOOD):
    data = b"\n".join(lines) + b"\n"
    return data, remote_for(data, "https://example.invalid/dataset.jsonl")


def test_run_label_extraction_writes_requests_and_outcomes(tmp_path):
    data, remote = serve()
    paths = Paths(tmp_path / "project")
    opener = opener_for(data)
    calls = []

    def chat(model, prompt):
        calls.append(model)
        return ChatResult('{"amount": 5}', 3, 4)

    rows = run_label_extraction(
        paths, ["m1", "m2"], chat=chat, opener=opener, free_bytes=plenty, remote=remote
    )
    assert rows == 6 and calls == ["m1"] * 3 + ["m2"] * 3
    assert paths.raw_text_json.read_bytes() == data
    outcomes = list(read_outcomes(paths.extraction_dir / "outcomes.jsonl"))
    assert [(r.model, r.prompt_id, r.success) for r in outcomes] == [
        (model, f"text-json:{n}", n == 1) for model in ("m1", "m2") for n in (1, 2, 3)
    ]
    assert all(r.dataset == DATASET and r.family == FAMILY for r in outcomes)
    assert all(r.prompt_tokens == 3 and r.completion_tokens == 4 for r in outcomes)
    stored = [
        json.loads(text)
        for text in (paths.extraction_dir / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [s["prompt_id"] for s in stored] == ["text-json:1", "text-json:2", "text-json:3"]
    assert stored[0] == {
        "prompt_id": "text-json:1",
        "prompt": PROMPT_TEMPLATE.format(schema='{"amount": "number"}', text="Paid 5 dollars"),
        "gold": {"amount": 5},
    }
    assert stored[2]["gold"] == {"days": ["Mon"]}
    assert opener.calls == 1


def test_run_label_extraction_reuses_the_verified_download(tmp_path):
    data, remote = serve()
    paths = Paths(tmp_path / "project")
    opener = opener_for(data)
    kwargs = {"chat": lambda m, p: ChatResult("{}", 1, 1), "free_bytes": plenty, "remote": remote}
    run_label_extraction(paths, ["m1"], opener=opener, **kwargs)
    run_label_extraction(paths, ["m1"], opener=opener, **kwargs)
    assert opener.calls == 1


def test_run_label_extraction_limit_keeps_requests_and_outcomes_consistent(tmp_path):
    data, remote = serve()
    paths = Paths(tmp_path / "project")
    rows = run_label_extraction(
        paths,
        ["m1", "m2"],
        chat=lambda m, p: ChatResult("{}", 1, 1),
        opener=opener_for(data),
        free_bytes=plenty,
        limit=2,
        remote=remote,
    )
    assert rows == 4
    requests_text = (paths.extraction_dir / "requests.jsonl").read_text(encoding="utf-8")
    assert [json.loads(t)["prompt_id"] for t in requests_text.splitlines()] == [
        "text-json:1",
        "text-json:2",
    ]
    assert len(list(read_outcomes(paths.extraction_dir / "outcomes.jsonl"))) == 4


def test_run_label_extraction_resolves_download_defaults_at_call_time(tmp_path, monkeypatch):
    data, remote = serve()
    opener = opener_for(data)
    monkeypatch.setattr(download, "open_url", opener)
    monkeypatch.setattr(download, "disk_free", plenty)
    paths = Paths(tmp_path / "project")
    run_label_extraction(paths, ["m1"], chat=lambda m, p: ChatResult("{}", 1, 1), remote=remote)
    assert opener.calls == 1
    assert paths.raw_text_json.read_bytes() == data


def test_run_label_extraction_writes_nothing_on_failure(tmp_path):
    data, remote = serve()
    paths = Paths(tmp_path / "project")
    calls = []

    def chat(model, prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise TrainerError("Ollama went away")
        return ChatResult('{"amount": 5}', 1, 1)

    with pytest.raises(TrainerError, match="Ollama went away"):
        run_label_extraction(
            paths,
            ["m1", "m2"],
            chat=chat,
            opener=opener_for(data),
            free_bytes=plenty,
            remote=remote,
        )
    assert len(calls) == 2
    assert not (paths.extraction_dir / "outcomes.jsonl").exists()
    assert not (paths.extraction_dir / "requests.jsonl").exists()


def test_run_label_extraction_failed_rerun_keeps_earlier_results(tmp_path):
    data, remote = serve()
    paths = Paths(tmp_path / "project")
    kwargs = {"opener": opener_for(data), "free_bytes": plenty, "remote": remote}
    run_label_extraction(paths, ["m1"], chat=lambda m, p: ChatResult("{}", 1, 1), limit=1, **kwargs)
    before = {
        name: (paths.extraction_dir / name).read_bytes()
        for name in ("requests.jsonl", "outcomes.jsonl")
    }

    def broken(model, prompt):
        raise TrainerError("Ollama went away")

    with pytest.raises(TrainerError):
        run_label_extraction(paths, ["m1"], chat=broken, **kwargs)
    after = {name: (paths.extraction_dir / name).read_bytes() for name in before}
    assert after == before


def test_cli_label_extraction_requires_models():
    assert main(["label-extraction", "--models", ""]) == 2


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--models", " , "], id="blank-names"),
        pytest.param(["--models", "a,,b"], id="empty-item"),
        pytest.param(["--models", "a,b,a"], id="duplicate"),
        pytest.param(["--models", "a", "--limit", "0"], id="zero-limit"),
        pytest.param(["--models", "a", "--limit", "many"], id="limit-not-a-number"),
        pytest.param([], id="no-models-option"),
    ],
)
def test_cli_label_extraction_rejects_bad_arguments(args, monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("must not run")

    monkeypatch.setattr(extraction, "run_label_extraction", unexpected)
    assert main(["label-extraction", *args]) == 2


def test_cli_label_extraction_runs_and_prints_summary(tmp_path, monkeypatch, capsys):
    calls = []
    built = []

    def fake_chat(model, prompt):
        raise AssertionError("the spy never labels")

    def fake_ollama_chat(base_url):
        built.append(base_url)
        return fake_chat

    def fake_run(paths, models, **kwargs):
        calls.append((paths, models, kwargs))
        return 12

    monkeypatch.setattr(extraction, "ollama_chat", fake_ollama_chat)
    monkeypatch.setattr(extraction, "run_label_extraction", fake_run)
    argv = ["--root", str(tmp_path), "label-extraction", "--models", "a:1b,b", "--limit", "3"]
    assert main([*argv, "--ollama-url", "http://gpu-box:9999"]) == 0
    assert capsys.readouterr().out == "label-extraction OK: 12 rows for 2 models\n"
    assert built == ["http://gpu-box:9999"]
    assert calls == [(Paths(tmp_path), ("a:1b", "b"), {"chat": fake_chat, "limit": 3})]


def test_cli_label_extraction_default_url_and_no_limit(tmp_path, monkeypatch):
    built = []
    calls = []
    monkeypatch.setattr(extraction, "ollama_chat", lambda base_url: built.append(base_url))
    monkeypatch.setattr(
        extraction, "run_label_extraction", lambda paths, models, **kw: calls.append(kw) or 1
    )
    assert main(["--root", str(tmp_path), "label-extraction", "--models", "a"]) == 0
    assert built == [OLLAMA_URL]
    assert calls[0]["limit"] is None


def test_cli_label_extraction_reports_trainer_errors(tmp_path, monkeypatch, capsys):
    def fail(paths, models, **kwargs):
        raise TrainerError("cannot reach Ollama at x")

    monkeypatch.setattr(extraction, "ollama_chat", lambda base_url: None)
    monkeypatch.setattr(extraction, "run_label_extraction", fail)
    assert main(["--root", str(tmp_path), "label-extraction", "--models", "a"]) == 1
    assert capsys.readouterr().err == "error: cannot reach Ollama at x\n"


def test_readme_states_the_guarantee_and_the_extraction_status():
    text = (TRAINER_DIR / "README.md").read_text(encoding="utf-8")
    assert GUARANTEE_NOTE in text
    assert EXTRACTION_STATUS in text
    assert len(text.splitlines()) < 80
    assert "\u2014" not in text


def test_readme_lists_the_committed_and_ignored_paths():
    text = (TRAINER_DIR / "README.md").read_text(encoding="utf-8")
    committed = (
        "data/manifest.json",
        "deploy/prices.yaml",
        "bench/latency_profiles.yaml",
        "bench/limits.json",
        "testdata/parity/features.jsonl",
        "trainer/artifacts/policy.json",
        "trainer/artifacts/reports/",
    )
    ignored = ("data/raw/", "data/work/", "trainer/artifacts/replay/")
    for path in (*committed, *ignored):
        assert f"`{path}`" in text, path
    for command in ("make py-setup", "make data", "make train", "make evaluate"):
        assert command in text, command
    assert "make label-extraction EXTRACTION_MODELS=" in text
