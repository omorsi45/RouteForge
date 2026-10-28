"""Structured-extraction workload: request builder, exact-field grader and Ollama labeler.

Only `ollama_chat` talks to a model server, and it runs from `make label-extraction`, never
from the test suite. Prompt text and model replies are never printed or put in error messages.
"""

import argparse
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from routeforge_train.download import Opener, RemoteFile, fetch_file
from routeforge_train.errors import TrainerError
from routeforge_train.jsonio import atomic_writer, dumps_line
from routeforge_train.outcomes import OutcomeRow, write_outcomes
from routeforge_train.paths import Paths

# Apache-2.0, 2,000 rows. Four rows contain cp1252 bytes and are not valid UTF-8.
TEXT_JSON = RemoteFile(
    "https://huggingface.co/datasets/suneeldk/text-json/resolve/"
    "d0474144541e250edffa3d25676798028e8f6bb3/dataset.jsonl",
    1_128_379,
    "b439561eb8438dfaaa654fc835d3e69946360207396f25e2cc8e9b0922aac121",
)
DATASET = "text-json"
FAMILY = "extraction"
PROMPT_TEMPLATE = (
    "Extract the information from the text into a JSON object that follows the schema. "
    "Reply with the JSON object only.\n\nSchema: {schema}\n\nText: {text}"
)
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_SEED = 20261026

_INPUT = "### Input: "
_SCHEMA = "\n### Schema: "
_OUTPUT = "\n### Output: "
_FENCE = "```"


@dataclass(frozen=True)
class ExtractionRequest:
    prompt_id: str
    prompt: str
    gold: dict


@dataclass(frozen=True)
class BuildStats:
    rows: int
    skipped_invalid_utf8: int
    skipped_malformed: int


def build_requests(src: Path) -> tuple[list[ExtractionRequest], BuildStats]:
    """Parse the text-json source into requests, counting the lines that cannot be used.

    A line that is not strict UTF-8 or whose `text` is not
    `### Input: <text>\\n### Schema: <schema>\\n### Output: <json object>` is skipped. Prompt ids
    use the 1-based physical line number, so they stay stable when lines are skipped.
    """
    requests: list[ExtractionRequest] = []
    skipped_invalid_utf8 = 0
    skipped_malformed = 0
    for number, raw in enumerate(Path(src).read_bytes().split(b"\n"), start=1):
        if not raw.strip():
            continue
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped_invalid_utf8 += 1
            continue
        parsed = _parse_line(line)
        if parsed is None:
            skipped_malformed += 1
            continue
        text, schema, gold = parsed
        prompt = PROMPT_TEMPLATE.format(schema=schema, text=text)
        requests.append(ExtractionRequest(f"{DATASET}:{number}", prompt, gold))
    stats = BuildStats(len(requests), skipped_invalid_utf8, skipped_malformed)
    return requests, stats


def _parse_line(line: str) -> tuple[str, str, dict] | None:
    """The input text, schema and gold object of one source line, or None when malformed."""
    try:
        record = json.loads(line)
    except ValueError:
        return None
    body = record.get("text") if isinstance(record, dict) else None
    if not isinstance(body, str) or not body.startswith(_INPUT):
        return None
    head, found, rest = body.partition(_SCHEMA)
    if not found:
        return None
    schema, found, output = rest.partition(_OUTPUT)
    if not found:
        return None
    try:
        gold = json.loads(output)
    except ValueError:
        return None
    if not isinstance(gold, dict):
        return None
    return head.removeprefix(_INPUT), schema, gold


def json_equal(a: object, b: object) -> bool:
    """Exact JSON equality.

    A bool never equals a number; an int equals a float of the same value; lists compare
    elementwise in order; objects need identical key sets and equal values.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(json_equal(a[key], b[key]) for key in a)
    return a is None and b is None


def grade(prediction: str, gold: Mapping[str, object]) -> float:
    """1.0 when the reply is a JSON object holding every gold field exactly, else 0.0.

    One surrounding Markdown code fence is removed first. Extra fields in the reply are ignored.
    """
    try:
        reply = json.loads(_unfence(prediction.strip()))
    except ValueError, RecursionError:  # RecursionError: a reply of thousands of "["
        return 0.0
    if not isinstance(reply, dict):
        return 0.0
    exact = all(key in reply and json_equal(reply[key], value) for key, value in gold.items())
    return 1.0 if exact else 0.0


def _unfence(text: str) -> str:
    if len(text) >= 2 * len(_FENCE) and text.startswith(_FENCE) and text.endswith(_FENCE):
        return text[len(_FENCE) : -len(_FENCE)].removeprefix("json").strip()
    return text


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


ChatFn = Callable[[str, str], ChatResult]  # (model, prompt)


def ollama_chat(base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 300.0) -> ChatFn:
    """A chat function backed by Ollama's `/api/chat`, greedy and seeded for repeatable runs."""
    endpoint = base_url.rstrip("/") + "/api/chat"

    def chat(model: str, prompt: str) -> ChatResult:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "seed": OLLAMA_SEED},
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:  # before URLError, its base class
            raise TrainerError(f"Ollama {exc.code} for model {model}: {_error_body(exc)}") from exc
        except urllib.error.URLError as exc:
            raise TrainerError(
                f"cannot reach Ollama at {base_url}: {exc.reason}; start it with: ollama serve"
            ) from exc
        except TimeoutError as exc:
            raise TrainerError(f"Ollama timed out after {timeout:g} s for model {model}") from exc
        return _parse_reply(model, payload)

    return chat


def _error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace").strip() or str(exc.reason)
    finally:
        exc.close()


def _parse_reply(model: str, payload: bytes) -> ChatResult:
    try:
        reply = json.loads(payload)
    except ValueError as exc:
        raise _unexpected(model, "the body is not JSON") from exc
    message = reply.get("message") if isinstance(reply, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise _unexpected(model, "there is no message.content string")
    counts = []
    # Ollama leaves prompt_eval_count out when the whole prompt came from its cache.
    for name in ("prompt_eval_count", "eval_count"):
        value = reply.get(name, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise _unexpected(model, f"{name} is not a non-negative integer")
        counts.append(value)
    return ChatResult(content, counts[0], counts[1])


def _unexpected(model: str, reason: str) -> TrainerError:
    return TrainerError(f"Ollama sent an unexpected response for model {model}: {reason}")


def label_requests(
    requests: Sequence[ExtractionRequest], models: Sequence[str], chat: ChatFn
) -> list[OutcomeRow]:
    """Ask every model every request, one model at a time so a server loads each only once."""
    rows = []
    for model in models:
        for request in requests:
            result = chat(model, request.prompt)
            score = grade(result.text, request.gold)
            rows.append(
                OutcomeRow(
                    prompt_id=request.prompt_id,
                    dataset=DATASET,
                    family=FAMILY,
                    prompt=request.prompt,
                    model=model,
                    score=score,
                    success=score >= 1.0,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    cost_usd=0.0,
                )
            )
    return rows


def run_label_extraction(
    paths: Paths,
    models: Sequence[str],
    *,
    chat: ChatFn | None = None,
    opener: Opener | None = None,
    free_bytes: Callable[[Path], int] | None = None,
    limit: int | None = None,
    remote: RemoteFile = TEXT_JSON,
) -> int:
    """Download and parse text-json, label it with every model and write the results.

    `opener` and `free_bytes` default to the download module's own, looked up at call time.
    `limit` keeps only the first N requests. Requests and outcomes are written only after
    every call succeeded, so a failed run leaves any earlier pair of files untouched.
    Returns the number of outcome rows written.
    """
    fetch_file(remote, paths.raw_text_json, opener=opener, free_bytes=free_bytes)
    requests, _stats = build_requests(paths.raw_text_json)
    if limit is not None:
        requests = requests[:limit]
    rows = label_requests(requests, models, ollama_chat() if chat is None else chat)
    with atomic_writer(paths.extraction_dir / "requests.jsonl") as f:
        for request in requests:
            f.write(
                dumps_line(
                    {"prompt_id": request.prompt_id, "prompt": request.prompt, "gold": request.gold}
                )
            )
    write_outcomes(rows, paths.extraction_dir / "outcomes.jsonl", paths.root)
    return len(rows)


def _model_list(value: str) -> tuple[str, ...]:
    models = tuple(item.strip() for item in value.split(","))
    if not all(models):
        raise argparse.ArgumentTypeError(
            "expected comma-separated model names, for example llama3.2:3b,qwen2.5:7b"
        )
    if len(set(models)) != len(models):
        raise argparse.ArgumentTypeError("model names must be unique")
    return models


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError("expected a number of at least 1")
    return number


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "label-extraction", help="label the text-json extraction workload with local Ollama models"
    )
    parser.add_argument(
        "--models",
        required=True,
        type=_model_list,
        help="comma-separated Ollama model tags, for example llama3.2:3b,qwen2.5:7b",
    )
    parser.add_argument(
        "--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama base URL (default: %(default)s)"
    )
    parser.add_argument(
        "--limit", type=_positive_int, default=None, help="label only the first N requests"
    )
    parser.set_defaults(func=_run)


def _run(paths: Paths, args: argparse.Namespace) -> int:
    rows = run_label_extraction(
        paths, args.models, chat=ollama_chat(args.ollama_url), limit=args.limit
    )
    print(f"label-extraction OK: {rows} rows for {len(args.models)} models")
    return 0
