"""Synthetic LLMRouterBench member files for the converter tests. No real data."""

import json
from pathlib import Path


def record(
    index: int,
    prompt: str,
    score: float,
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    cost: float = 0.001,
    prediction: object = "P",
    raw_output: object = "R",
) -> dict:
    return {
        "completion_tokens": completion_tokens,
        "cost": cost,
        "ground_truth": "G",
        "index": index,
        "origin_query": "Q",
        "prediction": prediction,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "raw_output": raw_output,
        "score": score,
    }


def write_member(
    raw_dir: Path,
    member: str,
    records: list[dict],
    *,
    model: str | None = None,
    dataset: str | None = None,
) -> None:
    """Write a member file under `raw_dir`; names default to the ones the path implies."""
    parts = member.split("/")
    path = raw_dir / member
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "completion_tokens": 0,
        "cost": 0.0,
        "counts": len(records),
        "data_fingerprint": "fp",
        "dataset_name": dataset if dataset is not None else parts[1],
        "demo": False,
        "model_name": model if model is not None else parts[-2],
        "performance": 0.0,
        "prompt_tokens": 0,
        "split": "test",
        "time_taken": 0.0,
        "records": records,
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
