import copy
import json
import re

from routeforge_train.certify import PROFILES
from routeforge_train.report_md import (
    COST_NOTE,
    EXTRACTION_NOTE,
    EXTRACTION_PRESENT_NOTE,
    GUARANTEE_NOTE,
    LATENCY_NOTE,
    TEMPLATE_NOTE,
    render_evaluation_markdown,
)
from tests.synth import SYNTH_POOL

PROFILE_NAMES = tuple(name for name, _ in PROFILES)


def test_render_evaluation_markdown_sections_and_notes(evaluated):
    md = render_evaluation_markdown(evaluated)
    for s in (
        "# RouteForge offline evaluation",
        "## H1 (balanced, ",
        "## Leave-one-dataset-out",
        "## Calibration",
        "## Extraction",
    ):
        assert s in md
    for note in (GUARANTEE_NOTE, TEMPLATE_NOTE, LATENCY_NOTE, COST_NOTE, EXTRACTION_NOTE):
        assert note in md
    assert evaluated["policy_version"] in md and "\u2014" not in md


def test_headings_are_the_documented_sections_in_order(evaluated):
    md = render_evaluation_markdown(evaluated)
    assert md.startswith("# RouteForge offline evaluation\n")
    assert re.findall(r"^## .*$", md, flags=re.MULTILINE) == [
        "## Provenance",
        f"## H1 (balanced, {evaluated['served_predictor']})",
        "## Strategies",
        "## Model shares (balanced)",
        "## Per dataset (balanced)",
        "## Leave-one-dataset-out",
        "## Calibration",
        "## Statistics",
        "## Costs",
        "## Notes",
        "## Extraction",
    ]
    assert md.endswith("\n") and not md.endswith("\n\n")


def test_markdown_is_a_pure_function_of_the_report(evaluated):
    md = render_evaluation_markdown(evaluated)
    assert render_evaluation_markdown(evaluated) == md
    assert render_evaluation_markdown(json.loads(json.dumps(evaluated))) == md


def test_evaluation_md_is_the_render_of_evaluation_json(trained_paths, evaluated):
    report = json.loads((trained_paths.reports / "evaluation.json").read_text(encoding="utf-8"))
    text = (trained_paths.reports / "evaluation.md").read_text(encoding="utf-8")
    assert render_evaluation_markdown(report) == text


def test_provenance_lists_the_run(evaluated):
    md = render_evaluation_markdown(evaluated)
    provenance = md.split("## Provenance")[1].split("\n## ")[0]
    assert evaluated["git_sha"] in provenance
    assert evaluated["created_at"] in provenance
    assert evaluated["test"]["sha256"] in provenance
    assert evaluated["served_predictor"] in provenance
    assert evaluated["source"]["archive_sha256"] in provenance
    for dataset, count in evaluated["test"]["by_dataset"].items():
        assert f"{dataset} {count}" in provenance


def test_every_profile_table_lists_every_relevant_strategy(evaluated):
    strategies = md_section(render_evaluation_markdown(evaluated), "## Strategies")
    for profile in PROFILE_NAMES:
        table = strategies.split(f"### {profile}\n")[1].split("\n### ")[0]
        names = ["always_strongest", "always_cheapest", "static_keyword", "static_oracle_family"]
        names += ["latency", "oracle", f"share_matched_random/{profile}"]
        names += [f"quality/{kind}/{profile}" for kind in ("lr", "gbm", "knn")]
        for name in names:
            assert f"| {name} |" in table, (profile, name)
        other = [p for p in PROFILE_NAMES if p != profile]
        assert not any(f"/{p} |" in table for p in other)
    assert (
        "Strategy | Accuracy | Total cost | Cost per success | Regret rate | Escalation rate"
        in (strategies)
    )


def test_model_shares_table_has_a_column_per_pool_model(evaluated):
    shares = md_section(render_evaluation_markdown(evaluated), "## Model shares (balanced)")
    header = next(line for line in shares.splitlines() if line.startswith("| Strategy"))
    for model in SYNTH_POOL:
        assert model in header
    assert "| always_strongest |" in shares and "| quality/lr/balanced |" in shares


def test_per_dataset_section_has_a_table_per_dataset(evaluated):
    section = md_section(render_evaluation_markdown(evaluated), "## Per dataset (balanced)")
    for dataset in evaluated["test"]["by_dataset"]:
        assert f"### {dataset}" in section


def test_lodo_section_lists_each_held_out_dataset_and_the_pooled_result(evaluated):
    section = md_section(render_evaluation_markdown(evaluated), "## Leave-one-dataset-out")
    for dataset in evaluated["lodo"]["datasets"]:
        assert f"| {dataset} |" in section
    assert "Pooled" in section
    assert "balanced" in section


def test_calibration_section_links_the_plots_and_lists_every_model(evaluated):
    section = md_section(render_evaluation_markdown(evaluated), "## Calibration")
    for kind in ("lr", "gbm", "knn"):
        assert f"reliability-{kind}.png" in section
        assert f"### {kind}" in section
    for model in SYNTH_POOL:
        assert f"| {model} |" in section
    assert "AUROC" in section and "Brier" in section and "ECE" in section


def test_statistics_section_reports_ci_holm_mcnemar_and_tost(evaluated):
    section = md_section(render_evaluation_markdown(evaluated), "## Statistics")
    for profile in PROFILE_NAMES:
        assert f"### {profile}" in section
    for baseline in evaluated["statistics"]["balanced"]["baselines"]:
        assert f"| {baseline} |" in section
    for word in ("Holm", "McNemar", "TOST", "Regret"):
        assert word in section


def test_verdict_line_follows_the_h1_block(evaluated):
    report = copy.deepcopy(evaluated)
    report["h1"]["falsified"] = True
    report["h1"]["falsified_by"] = ["share_matched_random", "lodo_gains_disappear"]
    md = render_evaluation_markdown(report)
    assert "H1: falsified (share_matched_random, lodo_gains_disappear)" in md
    assert "H1: supported" not in md
    report["h1"]["falsified"] = False
    report["h1"]["falsified_by"] = []
    md = render_evaluation_markdown(report)
    assert "H1: supported" in md and "H1: falsified" not in md


def test_h1_block_reports_target_estimate_regret_and_checks(evaluated):
    section = md_section(
        render_evaluation_markdown(evaluated), f"## H1 (balanced, {evaluated['served_predictor']})"
    )
    for word in ("Target", "Cost per success", "Regret", "share-matched random", "oracle", "LODO"):
        assert word in section, word


def test_h1_block_handles_undefined_intervals(evaluated):
    report = copy.deepcopy(evaluated)
    h1 = report["h1"]
    h1["cps_reduction_vs_strongest"] = {"estimate": None, "ci_low": None, "ci_high": None}
    h1["lodo_pooled_reduction"] = {"estimate": None, "ci_low": None, "ci_high": None}
    h1["vs_share_matched_random"]["cost_per_success_diff"] = {
        "estimate": None,
        "ci_low": None,
        "ci_high": None,
    }
    assert "n/a" in render_evaluation_markdown(report)


def test_values_that_round_to_zero_are_not_printed_negative(evaluated):
    report = copy.deepcopy(evaluated)
    tiny = {"estimate": -1e-9, "ci_low": -2e-9, "ci_high": 1e-9, "p_value": 1.0, "p_holm": 1.0}
    for stats in report["statistics"].values():
        for comparison in stats["baselines"].values():
            comparison["accuracy_diff"] = dict(tiny)
            comparison["cost_per_success_diff"] = dict(tiny)
            comparison["relative_cps_reduction"] = {k: tiny[k] for k in list(tiny)[:3]}
    report["h1"]["lodo_pooled_reduction"] = {k: tiny[k] for k in list(tiny)[:3]}
    md = render_evaluation_markdown(report)
    assert "-$0.0000" not in md and "-0.0%" not in md and "-0.0 pp" not in md
    assert "$0.0000 [$0.0000, $0.0000]" in md and "0.0% [0.0%, 0.0%]" in md
    assert "+0.0 pp [+0.0 pp, +0.0 pp]" in md


def test_negative_values_keep_their_sign(evaluated):
    report = copy.deepcopy(evaluated)
    cps = {"estimate": -0.0021, "ci_low": -0.003, "ci_high": -0.0012, "p_value": 0.01}
    report["h1"]["vs_share_matched_random"]["cost_per_success_diff"] = {
        k: cps[k] for k in list(cps)[:3]
    }
    report["h1"]["cps_reduction_vs_strongest"] = {"estimate": 0.31, "ci_low": 0.2, "ci_high": 0.4}
    md = render_evaluation_markdown(report)
    assert "-$0.0021 [-$0.0030, -$0.0012]" in md
    assert "31.0% [20.0%, 40.0%]" in md


def test_cost_rows_are_listed_only_beyond_one_percent(evaluated):
    report = copy.deepcopy(evaluated)
    cells = [("zz-high", 1.25), ("zz-low", 0.5), ("zz-near", 1.005), ("zz-none", None)]
    report["costs"]["by_dataset_model"] = [
        {"dataset": name, "model": SYNTH_POOL[0], "recorded_usd": 1.0, "list_usd": 1.0, "ratio": r}
        for name, r in cells
    ]
    section = md_section(render_evaluation_markdown(report), "## Costs")
    assert "zz-high" in section and "zz-low" in section
    assert "zz-near" not in section and "zz-none" not in section
    assert "1.250" in section and "0.500" in section
    report["costs"]["by_dataset_model"] = []
    assert "No dataset and model pair differs by more than 1%." in render_evaluation_markdown(
        report
    )


def test_costs_section_has_the_note_and_per_model_totals(evaluated):
    section = md_section(render_evaluation_markdown(evaluated), "## Costs")
    assert COST_NOTE in section
    for model in SYNTH_POOL:
        assert f"| {model} |" in section


def test_notes_and_extraction_sections(evaluated):
    md = render_evaluation_markdown(evaluated)
    notes = md_section(md, "## Notes")
    for note in (GUARANTEE_NOTE, TEMPLATE_NOTE, LATENCY_NOTE):
        assert note in notes
    assert md_section(md, "## Extraction").strip() == EXTRACTION_NOTE
    report = copy.deepcopy(evaluated)
    report["extraction"] = {
        "status": "labels_present_not_evaluated",
        "note": EXTRACTION_PRESENT_NOTE,
    }
    present = md_section(render_evaluation_markdown(report), "## Extraction")
    assert present.strip() == EXTRACTION_PRESENT_NOTE
    assert EXTRACTION_NOTE not in render_evaluation_markdown(report)


def md_section(md: str, heading: str) -> str:
    """The text under a level-2 heading, up to the next one."""
    return md.split(heading + "\n", 1)[1].split("\n## ", 1)[0]
