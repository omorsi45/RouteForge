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
EXTRACTION_PRESENT_NOTE = (
    "Extraction labels exist under data/work/extraction but this evaluation does not use them: "
    "extraction quality is unmeasured."
)

_PREDICTOR_ORDER = ("lr", "gbm", "knn")
_PROFILE_ORDER = ("strict", "balanced", "economy")
_BASELINE_ORDER = (
    "always_strongest",
    "always_cheapest",
    "static_keyword",
    "static_oracle_family",
    "latency",
)
_H1_PROFILE = "balanced"
_STRATEGY_HEADER = [
    "Strategy",
    "Accuracy",
    "Total cost",
    "Cost per success",
    "Regret rate",
    "Escalation rate",
]
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
    outside_rows = _cost_discrepancies(costs["by_dataset_model"])
    return [
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
        *outside_rows,
    ]


def _cost_discrepancies(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    """The dataset and model pairs whose recorded total is more than 1% off the list total."""
    outside = [
        cell
        for cell in cells
        if cell["ratio"] is not None and abs(cell["ratio"] - 1) > _COST_RATIO_TOLERANCE
    ]
    if not outside:
        return ["No dataset and model pair differs by more than 1%."]
    return [
        "Pairs whose recorded total differs from the list-price total by more than 1%:",
        "",
        *_table(
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
        ),
    ]


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


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def _p(x: float | None) -> str:
    return _optional(x, lambda value: f"{value:.4f}")


def _signed_usd(x: float) -> str:
    rounded = round(x, 4) + 0.0  # a value that prints as zero is not negative
    return f"-{fmt_usd(-rounded)}" if rounded < 0 else fmt_usd(rounded)


def _pct(x: float) -> str:
    return fmt_pct(round(x, 3) + 0.0)


def _signed_points(x: float) -> str:
    return f"{round(100 * x, 1) + 0.0:+.1f} pp"


def _interval(entry: Mapping[str, Any], fmt: Any) -> str:
    """`estimate [low, high]`; any undefined part reads n/a."""
    return (
        f"{_optional(entry['estimate'], fmt)} "
        f"[{_optional(entry['ci_low'], fmt)}, {_optional(entry['ci_high'], fmt)}]"
    )


def _strategy_names(report: Mapping[str, Any], profile: str) -> list[str]:
    names = [
        *_BASELINE_ORDER,
        *(f"quality/{kind}/{profile}" for kind in _PREDICTOR_ORDER),
        f"share_matched_random/{profile}",
        "oracle",
    ]
    return [name for name in names if name in report["strategies"]]


def _strategy_row(name: str, summary: Mapping[str, Any]) -> list[str]:
    return [
        name,
        fmt_pct(summary["accuracy"]),
        fmt_usd(summary["total_cost_usd"]),
        _optional(summary["cost_per_success_usd"], fmt_usd),
        fmt_pct(summary["regret_rate"]),
        _optional(summary["escalation_rate"], fmt_pct),
    ]


def _evaluation_provenance(report: Mapping[str, Any]) -> list[str]:
    source, test = report["source"], report["test"]
    stats = report["statistics"][_H1_PROFILE]
    per_dataset = ", ".join(f"{name} {count}" for name, count in sorted(test["by_dataset"].items()))
    return [
        "## Provenance",
        "",
        f"- Dataset: {source['dataset']}",
        f"- Revision: {source['revision']}",
        f"- Archive sha256: {source['archive_sha256']}",
        f"- Git SHA: {report['git_sha']}",
        f"- Policy version: {report['policy_version']}",
        f"- Served predictor: {report['served_predictor']}",
        f"- Created at: {report['created_at']}",
        f"- Test split: {test['n']} prompts, sha256 {test['sha256']}",
        f"- Test prompts per dataset: {per_dataset}",
        f"- Paired bootstrap: {stats['n_resamples']} resamples stratified by dataset; "
        f"share-matched random averages {stats['n_random_seeds']} seeds",
    ]


def _evaluation_h1(report: Mapping[str, Any]) -> list[str]:
    h1 = report["h1"]
    regret = h1["regret"]
    matched = h1["vs_share_matched_random"]
    oracle = h1["vs_static_oracle_family"]
    verdict = (
        f"H1: falsified ({', '.join(h1['falsified_by'])})" if h1["falsified"] else "H1: supported"
    )
    return [
        f"## H1 ({h1['profile']}, {h1['predictor']})",
        "",
        f"- Target: cost per success at least {fmt_pct(h1['target'])} below always_strongest. "
        f"Reduction: {_interval(h1['cps_reduction_vs_strongest'], _pct)} (95% CI). "
        f"Target met: {_yes_no(h1['target_met'])}.",
        f"- Regret rate: {fmt_pct(regret['rate'])} ({regret['count']} of {regret['n']}), "
        f"95% CI [{fmt_pct(regret['ci_low'])}, {fmt_pct(regret['ci_high'])}], "
        f"alpha {fmt_pct(regret['alpha'])}. "
        f"CI lower bound above alpha: {_yes_no(h1['regret_exceeds_alpha_beyond_ci'])}.",
        "- Cost per success versus share-matched random: "
        f"{_interval(matched['cost_per_success_diff'], _signed_usd)}. "
        f"Beats it: {_yes_no(matched['beats'])}.",
        "- Cost per success versus the static oracle family: "
        f"{_interval(oracle['cost_per_success_diff'], _signed_usd)}. "
        f"Beats it: {_yes_no(oracle['beats'])}.",
        "- LODO pooled cost per success reduction: "
        f"{_interval(h1['lodo_pooled_reduction'], _pct)}. "
        f"Gains disappear: {_yes_no(h1['lodo_gains_disappear'])}.",
        "",
        verdict,
    ]


def _evaluation_strategies(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Strategies",
        "",
        "Test split, recorded costs. Regret is measured against the policy's strongest model.",
    ]
    for profile in _PROFILE_ORDER:
        rows = [_strategy_row(n, report["strategies"][n]) for n in _strategy_names(report, profile)]
        lines += ["", f"### {profile}", "", *_table(_STRATEGY_HEADER, rows)]
    return lines


def _evaluation_shares(report: Mapping[str, Any]) -> list[str]:
    pool = report["test"]["pool"]
    rows = [
        [name, *(fmt_pct(report["strategies"][name]["shares"][model]) for model in pool)]
        for name in _strategy_names(report, _H1_PROFILE)
    ]
    return ["## Model shares (balanced)", "", *_table(["Strategy", *pool], rows)]


def _evaluation_per_dataset(report: Mapping[str, Any]) -> list[str]:
    names = _strategy_names(report, _H1_PROFILE)
    lines = ["## Per dataset (balanced)"]
    for dataset in sorted(report["per_dataset"]):
        rows = [_strategy_row(n, report["per_dataset"][dataset][n]) for n in names]
        lines += [
            "",
            f"### {dataset}",
            "",
            f"{report['test']['by_dataset'][dataset]} test prompts.",
            "",
            *_table(_STRATEGY_HEADER, rows),
        ]
    return lines


def _evaluation_lodo(report: Mapping[str, Any]) -> list[str]:
    lodo = report["lodo"]
    rows = []
    for dataset in sorted(lodo["datasets"]):
        entry = lodo["datasets"][dataset]
        quality, strongest = entry["quality"][_H1_PROFILE], entry["always_strongest"]
        rows.append(
            [
                dataset,
                str(entry["n"]),
                entry["strongest_model"],
                " / ".join(_lambda(entry["lambdas"][name]) for name in _PROFILE_ORDER),
                fmt_pct(quality["accuracy"]),
                fmt_pct(strongest["accuracy"]),
                _optional(quality["cost_per_success_usd"], fmt_usd),
                _optional(strongest["cost_per_success_usd"], fmt_usd),
                fmt_pct(quality["regret_rate"]),
            ]
        )
    pooled = lodo["pooled"]
    return [
        "## Leave-one-dataset-out",
        "",
        "Each row retrains the served predictor without the named dataset, certifies it on the "
        "remaining certification prompts and routes that dataset's test prompts "
        f"({_H1_PROFILE} profile). Lambdas are strict / balanced / economy.",
        "",
        *_table(
            [
                "Held-out dataset",
                "n",
                "Strongest",
                "Lambdas",
                "Quality accuracy",
                "Strongest accuracy",
                "Quality cost per success",
                "Strongest cost per success",
                "Quality regret rate",
            ],
            rows,
        ),
        "",
        f"Pooled ({pooled['profile']}, {pooled['n']} test prompts): cost per success reduction "
        f"versus always_strongest {_interval(pooled, _pct)} (95% CI).",
    ]


def _evaluation_calibration(report: Mapping[str, Any]) -> list[str]:
    pool = report["test"]["pool"]
    lines = [
        "## Calibration",
        "",
        "Isotonic blocks are the served calibration; Platt scaling, fit on the calibration "
        "split, is the offline comparison. Lower Brier and ECE are better.",
    ]
    for kind in _PREDICTOR_ORDER:
        if kind not in report["calibration"]:
            continue
        per_model = report["calibration"][kind]
        families = sorted(per_model[pool[0]]["auroc_by_family"])
        header = [
            "Model",
            "AUROC",
            *(f"AUROC {family}" for family in families),
            "Brier isotonic",
            "Brier Platt",
            "ECE isotonic",
            "ECE Platt",
        ]
        rows = []
        for model in pool:
            entry = per_model[model]
            rows.append(
                [
                    model,
                    _optional(entry["auroc_raw"], fmt_rate),
                    *(_optional(entry["auroc_by_family"][f], fmt_rate) for f in families),
                    fmt_rate(entry["brier_isotonic"]),
                    fmt_rate(entry["brier_platt"]),
                    fmt_rate(entry["ece_isotonic"]),
                    fmt_rate(entry["ece_platt"]),
                ]
            )
        lines += [
            "",
            f"### {kind}",
            "",
            f"Reliability plot: [reliability-{kind}.png](reliability-{kind}.png)",
            "",
            *_table(header, rows),
        ]
    return lines


def _evaluation_statistics(report: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Statistics",
        "",
        "Differences are quality minus baseline, from a paired bootstrap stratified by dataset "
        "(95% percentile intervals). Raw p-values are two-sided; Holm adjusts across the six "
        "baselines of a profile, separately for accuracy and for cost per success.",
    ]
    header = [
        "Baseline",
        "Accuracy diff",
        "p",
        "Holm p",
        "Cost per success diff",
        "p",
        "Holm p",
        "Cost per success reduction",
    ]
    for profile in _PROFILE_ORDER:
        stats = report["statistics"][profile]
        rows = []
        for baseline in (*_BASELINE_ORDER, f"share_matched_random/{profile}"):
            cmp = stats["baselines"][baseline]
            acc, cps = cmp["accuracy_diff"], cmp["cost_per_success_diff"]
            rows.append(
                [
                    baseline,
                    _interval(acc, _signed_points),
                    _p(acc["p_value"]),
                    _p(acc["p_holm"]),
                    _interval(cps, _signed_usd),
                    _p(cps["p_value"]),
                    _p(cps["p_holm"]),
                    _interval(cmp["relative_cps_reduction"], _pct),
                ]
            )
        mcnemar, tost, regret = (
            stats["mcnemar_vs_always_strongest"],
            stats["tost_vs_always_strongest"],
            stats["regret"],
        )
        low, high = tost["ci90"]
        lines += [
            "",
            f"### {profile}",
            "",
            f"Quality strategy: {stats['quality']}.",
            "",
            *_table(header, rows),
            "",
            f"- McNemar versus always_strongest: quality only {mcnemar['b']}, "
            f"always_strongest only {mcnemar['c']}, exact p {_p(mcnemar['p_value'])}.",
            f"- TOST accuracy versus always_strongest, margin +/-{100 * tost['margin']:.1f} pp: "
            f"difference {_signed_points(tost['estimate'])}, "
            f"90% CI [{_signed_points(low)}, {_signed_points(high)}], "
            f"non-inferior {_yes_no(tost['non_inferior'])} (p {_p(tost['p_lower'])}), "
            f"equivalent {_yes_no(tost['equivalent'])} (p {_p(tost['p_value'])}).",
            f"- Regret rate {fmt_pct(regret['rate'])} ({regret['count']} of {regret['n']}), "
            f"95% exact CI [{fmt_pct(regret['ci_low'])}, {fmt_pct(regret['ci_high'])}], "
            f"alpha {fmt_pct(regret['alpha'])}.",
        ]
    return lines


def _evaluation_costs(report: Mapping[str, Any]) -> list[str]:
    costs = report["costs"]
    rows = [
        [
            model,
            fmt_usd(costs["by_model"][model]["recorded_usd"]),
            fmt_usd(costs["by_model"][model]["list_usd"]),
        ]
        for model in report["test"]["pool"]
    ]
    return [
        "## Costs",
        "",
        COST_NOTE,
        "",
        *_table(["Model", "Recorded (test rows)", "List (test rows)"], rows),
        "",
        *_cost_discrepancies(costs["by_dataset_model"]),
    ]


def render_evaluation_markdown(report: Mapping[str, Any]) -> str:
    """The evaluation report as Markdown, ending in exactly one newline.

    The text depends only on the report, in any key order, so it can be rendered again from
    the JSON file.
    """
    sections = [
        _evaluation_provenance(report),
        _evaluation_h1(report),
        _evaluation_strategies(report),
        _evaluation_shares(report),
        _evaluation_per_dataset(report),
        _evaluation_lodo(report),
        _evaluation_calibration(report),
        _evaluation_statistics(report),
        _evaluation_costs(report),
        ["## Notes", "", f"- {GUARANTEE_NOTE}", f"- {TEMPLATE_NOTE}", f"- {LATENCY_NOTE}"],
        ["## Extraction", "", report["extraction"]["note"]],
    ]
    lines = ["# RouteForge offline evaluation"]
    for section in sections:
        lines += ["", *section]
    return "\n".join(lines) + "\n"
