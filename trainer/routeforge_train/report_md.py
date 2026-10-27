"""Markdown rendering of the training report. Every number is formatted from the JSON report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

GUARANTEE_NOTE = (
    "With probability at least 1 - delta over the draw of the certification set, the expected "
    "regret rate on future traffic exchangeable with it is at most alpha, when all pool models "
    "are eligible. It says nothing about any single request."
)
COST_NOTE = (
    "Offline costs are the dataset-recorded cost field (provider usage times list price per "
    "LLMRouterBench). The gateway price table (deploy/prices.yaml) uses list prices. Files whose "
    "recorded total differs from the list-price recomputation by more than 1% are listed below."
)
TEMPLATE_NOTE = (
    "Benchmark prompt templates let a classifier recognize the dataset, which inflates "
    "in-distribution gains; leave-one-dataset-out results are reported next to them."
)
LATENCY_NOTE = (
    "The latency-aware router uses SIMULATED latency profiles (bench/latency_profiles.yaml) and "
    "is a reference only, not part of the H1 test."
)
EXTRACTION_NOTE = "Extraction quality is unmeasured: make label-extraction has not been run."

_PREDICTOR_ORDER = ("lr", "gbm", "knn")
_COST_RATIO_TOLERANCE = 0.01


def fmt_usd(x: float) -> str:
    return f"${x:,.4f}"


def fmt_rate(x: float) -> str:
    return f"{x:.3f}"


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _lambda(x: float) -> str:
    return f"{x:.2f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _optional(x: float | None, fmt: Any) -> str:
    return "n/a" if x is None else fmt(x)


def _provenance(report: Mapping[str, Any]) -> list[str]:
    source = report["source"]
    environment = ", ".join(
        f"{key} {value}" for key, value in sorted(report["environment"].items())
    )
    return [
        "## Provenance",
        "",
        f"- Dataset: {source['dataset']}",
        f"- Revision: {source['revision']}",
        f"- Archive sha256: {source['archive_sha256']}",
        f"- Git SHA: {report['git_sha']}",
        f"- Policy version: {report['policy_version']}",
        f"- Created at: {report['created_at']}",
        f"- Environment: {environment}",
    ]


def _data(report: Mapping[str, Any]) -> list[str]:
    data = report["data"]
    datasets = data["datasets"]
    reasons = sorted({reason for stats in datasets.values() for reason in stats["dropped"]})
    split_names = ("train", "calibration", "certification", "test")
    header = ["Dataset", "Family", "Prompts in", "Kept", *reasons, "Label rule", *split_names]
    rows = [
        [
            dataset,
            stats["family"],
            str(stats["prompts_in"]),
            str(stats["prompts_kept"]),
            *(str(stats["dropped"].get(reason, 0)) for reason in reasons),
            stats["label_rule"],
            *(str(stats["split_counts"][name]) for name in split_names),
        ]
        for dataset, stats in sorted(datasets.items())
    ]
    totals = ", ".join(f"{name} {data['split_counts'][name]}" for name in split_names)
    return [
        "## Data",
        "",
        f"Pool, in tie-break order: {', '.join(data['pool'])}.",
        "",
        *_table(header, rows),
        "",
        f"Prompts per split: {totals}.",
    ]


def _predictors(report: Mapping[str, Any]) -> list[str]:
    features = report["features"]
    lr = report["lr"]
    params = ", ".join(f"{key} {value}" for key, value in sorted(report["gbm"]["params"].items()))
    rows = []
    for model in report["data"]["pool"]:
        c = lr["c_by_model"][model]
        loss = lr["cv_log_loss_by_model"].get(model, {}).get(str(c))
        rows.append(
            [
                model,
                str(c),
                _optional(loss, fmt_rate),
                f"{report['expected_output_tokens'][model]:g}",
            ]
        )
    return [
        "## Predictors",
        "",
        f"- Hashed features: {features['hash_dim']} buckets; vocabulary of "
        f"{features['vocab_size']} indices (document frequency at least "
        f"{features['vocab_min_df']}, at most {features['vocab_max']}).",
        f"- Logistic regression weights are rounded to {features['lr_weight_sig_digits']} "
        "significant digits.",
        f"- Gradient boosting parameters: {params}.",
        f"- kNN: k = {report['knn']['k']}, offline only.",
        f"- Strongest model: {report['strongest_model']}. Cheapest model: "
        f"{report['cheapest_model']}.",
        "",
        *_table(["Model", "LR C", "LR CV log loss", "Expected output tokens"], rows),
    ]


def _selection(report: Mapping[str, Any]) -> list[str]:
    selection = report["selection"]
    candidates = selection["candidates"]
    rows = [
        [
            kind,
            _lambda(candidates[kind]["lambda"]),
            fmt_pct(candidates[kind]["accuracy"]),
            fmt_usd(candidates[kind]["total_cost_usd"]),
            _optional(candidates[kind]["cost_per_success_usd"], fmt_usd),
            fmt_pct(candidates[kind]["regret_rate"]),
        ]
        for kind in _PREDICTOR_ORDER
        if kind in candidates
    ]
    return [
        "## Selection (calibration split, balanced)",
        "",
        f"Calibration prompts: {selection['calibration_n']}. Profile: {selection['profile']}.",
        "",
        *_table(
            ["Predictor", "Lambda", "Accuracy", "Total cost", "Cost per success", "Regret rate"],
            rows,
        ),
        "",
        f"Winner: {selection['winner']}",
    ]


def _profile_table(profiles: Sequence[Mapping[str, Any]]) -> list[str]:
    rows = [
        [
            profile["name"],
            f"{profile['alpha']:.2f}",
            _lambda(profile["lambda"]),
            str(profile["cert_n"]),
            str(profile["cert_regret"]),
            fmt_rate(profile["regret_ucb"]),
        ]
        for profile in profiles
    ]
    return _table(["Profile", "Alpha", "Lambda", "n", "Regret count", "Regret UCB"], rows)


def _certification(report: Mapping[str, Any]) -> list[str]:
    certification = report["certification"]
    winner = report["selection"]["winner"]
    lines = [
        "## Certification",
        "",
        f"### Served ({winner})",
        "",
        *_profile_table(certification["served"]),
    ]
    sentinel_notes = [note for note in report["notes"] if note != GUARANTEE_NOTE]
    if sentinel_notes:
        lines += ["", *(f"- {note}" for note in sentinel_notes)]
    offline = certification["offline"]
    for kind in _PREDICTOR_ORDER:
        if kind in offline:
            lines += ["", f"### Offline comparison ({kind}, never exported)", ""]
            lines += _profile_table(offline[kind])
    return lines


def _costs(report: Mapping[str, Any]) -> list[str]:
    costs = report["costs"]
    rows = [
        [
            model,
            fmt_usd(costs["by_model"][model]["recorded_usd_all_rows"]),
            fmt_usd(costs["by_model"][model]["list_usd_all_rows"]),
            fmt_usd(costs["by_model"][model]["recorded_usd_kept"]),
            fmt_usd(costs["by_model"][model]["list_usd_kept"]),
        ]
        for model in report["data"]["pool"]
    ]
    outside = [
        cell
        for cell in costs["by_dataset_model"]
        if cell["ratio"] is not None and abs(cell["ratio"] - 1) > _COST_RATIO_TOLERANCE
    ]
    lines = [
        "## Costs",
        "",
        COST_NOTE,
        "",
        *_table(
            [
                "Model",
                "Recorded (all rows)",
                "List (all rows)",
                "Recorded (kept rows)",
                "List (kept rows)",
            ],
            rows,
        ),
        "",
    ]
    if outside:
        lines += [
            "Pairs whose recorded total differs from the list-price total by more than 1%:",
            "",
        ]
        lines += _table(
            ["Dataset", "Model", "Recorded", "List", "Ratio"],
            [
                [
                    cell["dataset"],
                    cell["model"],
                    fmt_usd(cell["recorded_usd"]),
                    fmt_usd(cell["list_usd"]),
                    fmt_rate(cell["ratio"]),
                ]
                for cell in outside
            ],
        )
    else:
        lines.append("No dataset and model pair differs by more than 1%.")
    return lines


def render_training_markdown(report: Mapping[str, Any]) -> str:
    """The training report as Markdown, ending in exactly one newline."""
    sections = [
        _provenance(report),
        _data(report),
        _predictors(report),
        _selection(report),
        _certification(report),
        _costs(report),
        ["## Guarantee", "", GUARANTEE_NOTE],
    ]
    lines = ["# RouteForge training report"]
    for section in sections:
        lines += ["", *section]
    return "\n".join(lines) + "\n"
