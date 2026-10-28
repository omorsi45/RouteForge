import dataclasses
import json

import numpy as np
import pytest

from routeforge_train import evaluate as evaluate_module
from routeforge_train import splits
from routeforge_train.certify import LAMBDA_GRID, PROFILES
from routeforge_train.cli import main
from routeforge_train.evaluate import (
    extraction_status,
    h1_verdict,
    oracle_choice,
    run_evaluate,
    share_matched_random,
)
from routeforge_train.features import featurize, prompt_sha256
from routeforge_train.jsonio import sha256_file, write_json
from routeforge_train.metrics import summarize
from routeforge_train.outcomes import read_outcomes
from routeforge_train.pipeline import request_texts
from routeforge_train.policy import export_policy, load_policy, score_policy
from routeforge_train.prices import load_prices
from routeforge_train.report_md import EXTRACTION_NOTE
from routeforge_train.routing import (
    REASON_ESCALATED,
    REASON_PASSING,
    expected_cost_usd,
    route_quality,
)
from routeforge_train.splits import HeldOutSplitChangedError
from routeforge_train.stats import SEED, holm
from routeforge_train.table import pivot
from routeforge_train.train import run_train
from tests.synth import FIXED_NOW, SYNTH_POOL, build_synthetic_project

SMALL = 100
PROFILE_NAMES = tuple(name for name, _ in PROFILES)
KINDS = ("lr", "gbm", "knn")
REPORT_KEYS = {
    "schema_version",
    "created_at",
    "git_sha",
    "policy_version",
    "served_predictor",
    "source",
    "test",
    "strategies",
    "per_dataset",
    "lambda_sweep",
    "calibration",
    "statistics",
    "lodo",
    "h1",
    "costs",
    "extraction",
    "notes",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_report_has_all_strategies(evaluated):
    names = set(evaluated["strategies"])
    assert {
        "always_strongest",
        "always_cheapest",
        "static_keyword",
        "static_oracle_family",
        "latency",
        "oracle",
    } <= names
    for p in PROFILE_NAMES:
        assert f"share_matched_random/{p}" in names
        assert {f"quality/{k}/{p}" for k in KINDS} <= names
    assert len(names) == 6 + 3 * 3 + 3


def test_report_top_level_keys_and_header(evaluated, trained_paths):
    manifest = read_json(trained_paths.manifest)
    policy = load_policy(trained_paths.policy)
    assert set(evaluated) == REPORT_KEYS
    assert evaluated["schema_version"] == 1
    assert evaluated["created_at"] == "2026-10-26T00:00:00Z"
    assert evaluated["git_sha"] == "synthetic"
    assert evaluated["policy_version"] == policy.version
    assert evaluated["served_predictor"] == policy.predictor_type
    assert evaluated["source"] == manifest["source"]
    test = evaluated["test"]
    assert test["n"] == manifest["files"]["test"]["rows"] // len(SYNTH_POOL)
    assert test["sha256"] == manifest["files"]["test"]["sha256"]
    assert sum(test["by_dataset"].values()) == test["n"]
    assert sorted(test["by_dataset"]) == ["alpha", "beta", "delta", "gamma"]


def test_every_strategy_covers_every_test_prompt(evaluated):
    n = evaluated["test"]["n"]
    assert all(s["n"] == n for s in evaluated["strategies"].values())
    for dataset, count in evaluated["test"]["by_dataset"].items():
        assert all(s["n"] == count for s in evaluated["per_dataset"][dataset].values())
    assert sorted(evaluated["per_dataset"]) == sorted(evaluated["test"]["by_dataset"])


def test_always_strongest_and_oracle_have_zero_regret(evaluated):
    assert evaluated["strategies"]["always_strongest"]["regret_rate"] == 0.0
    assert evaluated["strategies"]["oracle"]["regret_rate"] == 0.0
    strongest = evaluated["strategies"]["always_strongest"]["shares"]
    assert max(strongest.values()) == 1.0


def test_oracle_accuracy_is_upper_bound(evaluated):
    oracle = evaluated["strategies"]["oracle"]["accuracy"]
    assert all(s["accuracy"] <= oracle for s in evaluated["strategies"].values())
    n = evaluated["test"]["n"]
    per_dataset = sum(
        evaluated["per_dataset"][d]["oracle"]["accuracy"] * count
        for d, count in evaluated["test"]["by_dataset"].items()
    )
    assert oracle == pytest.approx(per_dataset / n)


def test_oracle_choice_known():
    success = np.array(
        [[0, 1, 1], [0, 0, 0], [1, 1, 1], [0, 1, 1]],
        dtype=bool,
    )
    cost = np.array(
        [[1.0, 3.0, 2.0], [5.0, 4.0, 4.0], [2.0, 2.0, 2.0], [9.0, 1.0, 1.0]],
    )
    chosen = oracle_choice(success, cost)
    assert chosen.tolist() == [2, 1, 0, 1] and chosen.dtype == np.int64


def test_share_matched_random_uses_exact_quality_counts():
    chosen = np.array([0, 0, 1, 2, 2, 2])
    success = np.ones((6, 3), dtype=bool)
    cost = np.tile([1.0, 100.0, 10000.0], (6, 1))
    got = share_matched_random(chosen, success, cost, 0, ("a", "b", "c"), n_seeds=7)
    assert got.shares == {"a": 2 / 6, "b": 1 / 6, "c": 3 / 6}
    # Every seed assigns exactly two rows to a, one to b and three to c.
    assert summarize(got)["total_cost_usd"] == pytest.approx(2 * 1 + 100 + 3 * 10000)
    assert summarize(got)["accuracy"] == 1.0
    assert got.escalated is None and got.chosen is None


def test_share_matched_random_assignment_is_the_seeded_permutation():
    chosen = np.array([0, 0, 1, 2, 2, 2])
    n = len(chosen)
    success = np.ones((n, 3), dtype=bool)
    cost = np.arange(n)[:, None] * 10.0 + np.arange(3)[None, :]
    got = share_matched_random(chosen, success, cost, 0, ("a", "b", "c"), n_seeds=1)
    expected = np.random.default_rng(SEED).permutation(np.repeat(np.arange(3), [2, 1, 3]))
    assert (got.cost - np.arange(n) * 10.0).tolist() == expected.tolist()
    other = share_matched_random(
        chosen, success, cost, 0, ("a", "b", "c"), n_seeds=1, seed=SEED + 1
    )
    expected_other = np.random.default_rng(SEED + 1).permutation(np.repeat(np.arange(3), [2, 1, 3]))
    assert (other.cost - np.arange(n) * 10.0).tolist() == expected_other.tolist()


def test_share_matched_random_averages_over_seeds():
    chosen = np.array([0, 1, 1, 1])
    success = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool)
    cost = np.ones((4, 2))
    got = share_matched_random(chosen, success, cost, 0, ("a", "b"), n_seeds=400)
    # One row goes to model a (success on rows 0 and 1), three to model b (rows 2 and 3).
    assert got.success.tolist() == pytest.approx([0.25, 0.25, 0.75, 0.75], abs=0.08)
    assert summarize(got)["successes"] == pytest.approx(2.0, abs=0.2)
    assert got.shares == {"a": 0.25, "b": 0.75}
    again = share_matched_random(chosen, success, cost, 0, ("a", "b"), n_seeds=400)
    assert np.array_equal(again.success, got.success)


def test_share_matched_random_regret_counts_rows_moved_off_the_strongest():
    chosen = np.array([0, 0, 1, 2, 2, 2])
    success = np.zeros((6, 3), dtype=bool)
    success[:, 0] = True  # only the strongest model succeeds
    got = share_matched_random(chosen, success, np.ones((6, 3)), 0, ("a", "b", "c"), n_seeds=11)
    assert summarize(got)["regret_count"] == pytest.approx(4.0)


def test_share_matched_random_rejects_empty_input():
    with pytest.raises(ValueError, match="no rows"):
        share_matched_random(
            np.array([], dtype=np.int64), np.zeros((0, 2), bool), np.zeros((0, 2)), 0, ("a", "b")
        )


def recompute_quality(paths):
    """Quality routing of the test prompts straight from policy.json, outside the evaluator."""
    policy = load_policy(paths.policy)
    pool = tuple(entry.name for entry in policy.models)
    manifest = read_json(paths.manifest)
    table = pivot(read_outcomes(splits.split_path(paths.splits_dir, "test")), manifest["pool"])
    scores = score_policy(policy, featurize(request_texts(table)))
    exp_cost = expected_cost_usd(
        table.prompt_bytes,
        pool,
        load_prices(paths.prices),
        {entry.name: entry.expected_output_tokens for entry in policy.models},
    )
    strongest = pool.index(policy.strongest_model)
    routed = {
        profile.name: route_quality(scores.lcb, exp_cost, strongest, profile.lambda_)
        for profile in policy.profiles
    }
    return policy, table, routed


def assert_quality_matches(report, paths):
    policy, table, routed = recompute_quality(paths)
    rows = np.arange(len(table))
    for profile in policy.profiles:
        chosen, escalated = routed[profile.name]
        got = report["strategies"][f"quality/{policy.predictor_type}/{profile.name}"]
        assert got["lambda"] == profile.lambda_
        assert got["accuracy"] == pytest.approx(table.success[rows, chosen].mean())
        assert got["total_cost_usd"] == pytest.approx(table.cost[rows, chosen].sum())
        assert got["escalation_rate"] == pytest.approx(escalated.mean())
        for j, model in enumerate(table.pool):
            assert got["shares"][model] == pytest.approx((chosen == j).mean())
    return policy, table, routed


def test_quality_matches_policy_lambda(evaluated, trained_paths):
    assert_quality_matches(evaluated, trained_paths)


def retarget_lambdas(paths, lambdas):
    """Rewrite the served policy, its candidate file and training.json with new lambdas."""
    policy = load_policy(paths.policy)
    profiles = tuple(
        dataclasses.replace(profile, lambda_=lambdas[profile.name]) for profile in policy.profiles
    )
    patched = dataclasses.replace(policy, profiles=profiles)
    version, _ = export_policy(patched, paths.policy)
    export_policy(patched, paths.candidates_dir / f"{policy.predictor_type}.json")
    training = read_json(paths.reports / "training.json")
    training["policy_version"] = version
    write_json(paths.reports / "training.json", training)


def test_quality_routing_follows_lambdas_that_split_the_pool(tmp_path):
    paths = build_synthetic_project(tmp_path)
    run_train(paths, git_sha="synthetic", now=FIXED_NOW)
    # The sentinel escalates everything; 0.3 and 0.5 route to different cheaper models.
    retarget_lambdas(paths, {"strict": 1.01, "balanced": 0.3, "economy": 0.5})
    report = run_evaluate(
        paths, git_sha="synthetic", now=FIXED_NOW, n_resamples=100, n_random_seeds=5
    )
    policy, table, routed = assert_quality_matches(report, paths)
    served = policy.predictor_type
    strongest = table.pool.index(policy.strongest_model)

    strict, escalated = routed["strict"]
    assert (strict == strongest).all() and escalated.all()
    balanced, _ = routed["balanced"]
    economy, _ = routed["economy"]
    assert len(np.unique(balanced)) >= 2 and not np.array_equal(balanced, economy)
    assert report["strategies"][f"quality/{served}/strict"]["escalation_rate"] == 1.0

    # McNemar counts and the realized regret are exact functions of the recorded outcomes.
    rows = np.arange(len(table))
    quality_ok, strongest_ok = table.success[rows, balanced], table.success[:, strongest]
    only_quality = int((quality_ok & ~strongest_ok).sum())
    only_strongest = int((~quality_ok & strongest_ok).sum())
    stats = report["statistics"]["balanced"]
    assert only_quality + only_strongest > 0
    assert (
        stats["mcnemar_vs_always_strongest"]["b"],
        stats["mcnemar_vs_always_strongest"]["c"],
    ) == (
        only_quality,
        only_strongest,
    )
    assert stats["regret"]["count"] == only_strongest

    # Share-matched random reproduces the served choice counts exactly, here with a real mix.
    for profile in PROFILE_NAMES:
        quality = report["strategies"][f"quality/{served}/{profile}"]
        random = report["strategies"][f"share_matched_random/{profile}"]
        assert random["shares"] == pytest.approx(quality["shares"])
    mixed = report["strategies"]["share_matched_random/balanced"]
    assert max(mixed["shares"].values()) < 1.0

    # The replay decisions are the same routing, prompt by prompt.
    rows = [
        json.loads(line)
        for line in (paths.replay_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for profile in PROFILE_NAMES:
        chosen, escalated = routed[profile]
        got = [r for r in rows if r["strategy"] == "quality" and r["profile"] == profile]
        assert [r["chosen"] for r in got] == [table.pool[j] for j in chosen]
        assert [r["reason"] for r in got] == [
            REASON_ESCALATED if e else REASON_PASSING for e in escalated
        ]


def test_quality_lambdas_come_from_the_candidate_certifications(evaluated, training):
    winner = training["selection"]["winner"]
    assert evaluated["served_predictor"] == winner
    certification = training["certification"]
    for kind in KINDS:
        profiles = certification["served"] if kind == winner else certification["offline"][kind]
        for profile in profiles:
            got = evaluated["strategies"][f"quality/{kind}/{profile['name']}"]
            assert got["lambda"] == profile["lambda"]


def test_lambda_sweep_covers_the_grid_for_each_kind(evaluated):
    assert set(evaluated["lambda_sweep"]) == set(KINDS)
    strongest = evaluated["strategies"]["always_strongest"]
    for kind in KINDS:
        sweep = evaluated["lambda_sweep"][kind]
        assert [point["lambda"] for point in sweep] == list(LAMBDA_GRID)
        assert set(sweep[0]) == {
            "lambda",
            "accuracy",
            "total_cost_usd",
            "cost_per_success_usd",
            "regret_rate",
            "escalation_rate",
        }
        # No Wilson bound reaches 1.0, so lambda = 1.00 always escalates to the strongest model.
        assert sweep[0]["escalation_rate"] == 1.0 and sweep[0]["regret_rate"] == 0.0
        assert sweep[0]["accuracy"] == strongest["accuracy"]
        assert sweep[0]["total_cost_usd"] == pytest.approx(strongest["total_cost_usd"])
        assert sweep[-1]["escalation_rate"] == 0.0


def test_calibration_section_per_kind_and_model(evaluated):
    n = evaluated["test"]["n"]
    assert set(evaluated["calibration"]) == set(KINDS)
    for kind in KINDS:
        assert set(evaluated["calibration"][kind]) == set(SYNTH_POOL)
        for entry in evaluated["calibration"][kind].values():
            assert set(entry) == {
                "auroc_raw",
                "auroc_by_family",
                "brier_isotonic",
                "brier_platt",
                "ece_isotonic",
                "ece_platt",
                "reliability",
            }
            assert set(entry["reliability"]) == {"isotonic", "platt"}
            for curve in entry["reliability"].values():
                assert sum(b["n"] for b in curve) == n
                assert all(set(b) == {"n", "mean_p", "frac_success"} for b in curve)
            for key in ("brier_isotonic", "brier_platt", "ece_isotonic", "ece_platt"):
                assert 0.0 <= entry[key] <= 1.0
            assert entry["auroc_raw"] is None or 0.0 <= entry["auroc_raw"] <= 1.0
            assert set(entry["auroc_by_family"]) == {"reasoning", "code", "qa", "instruction"}


def test_statistics_per_profile(evaluated):
    baselines = {
        "always_strongest",
        "always_cheapest",
        "static_keyword",
        "static_oracle_family",
        "latency",
    }
    served = evaluated["served_predictor"]
    assert set(evaluated["statistics"]) == set(PROFILE_NAMES)
    for profile, alpha in PROFILES:
        stats = evaluated["statistics"][profile]
        assert stats["quality"] == f"quality/{served}/{profile}"
        assert stats["n_resamples"] == 2000
        assert set(stats["baselines"]) == baselines | {f"share_matched_random/{profile}"}
        for comparison in stats["baselines"].values():
            assert set(comparison) == {
                "accuracy_diff",
                "cost_per_success_diff",
                "relative_cps_reduction",
                "degenerate_resamples",
            }
            assert comparison["accuracy_diff"]["ci_low"] <= comparison["accuracy_diff"]["ci_high"]
        mcnemar = stats["mcnemar_vs_always_strongest"]
        assert set(mcnemar) == {"b", "c", "p_value"}
        assert set(stats["tost_vs_always_strongest"]) >= {
            "estimate",
            "margin",
            "p_lower",
            "p_upper",
            "p_value",
            "equivalent",
            "non_inferior",
            "ci90",
        }
        regret = stats["regret"]
        quality = evaluated["strategies"][stats["quality"]]
        assert regret["alpha"] == alpha and regret["n"] == evaluated["test"]["n"]
        assert regret["count"] == round(quality["regret_count"])
        assert regret["rate"] == regret["count"] / regret["n"]
        assert 0.0 <= regret["ci_low"] <= regret["rate"] <= regret["ci_high"] <= 1.0


def test_holm_applied_per_profile(evaluated):
    for profile in PROFILE_NAMES:
        baselines = evaluated["statistics"][profile]["baselines"]
        assert len(baselines) == 6
        for key in ("cost_per_success_diff", "accuracy_diff"):
            raw = [c[key]["p_value"] for c in baselines.values()]
            adjusted = [c[key]["p_holm"] for c in baselines.values()]
            assert all(a >= r for a, r in zip(adjusted, raw, strict=True))
            assert adjusted == pytest.approx(holm(raw))


def test_evaluate_refuses_tampered_test_split(tmp_path, capsys):
    paths = build_synthetic_project(tmp_path, n_per_dataset=SMALL)
    run_train(paths, git_sha="synthetic", now=FIXED_NOW)
    test_file = splits.split_path(paths.splits_dir, "test")
    test_file.write_bytes(test_file.read_bytes().replace(b"alpha", b"alphb", 1))
    with pytest.raises(HeldOutSplitChangedError):
        run_evaluate(paths, git_sha="synthetic", now=FIXED_NOW, n_resamples=50, n_random_seeds=5)
    assert main(["--root", str(paths.root), "evaluate"]) == 1
    assert "test.jsonl" in capsys.readouterr().err
    assert not (paths.reports / "evaluation.json").exists()
    assert not (paths.replay_dir / "decisions.jsonl").exists()


def test_evaluate_requires_a_trained_policy(tmp_path, capsys):
    paths = build_synthetic_project(tmp_path, n_per_dataset=SMALL)
    assert main(["--root", str(paths.root), "evaluate"]) == 1
    assert "policy.json" in capsys.readouterr().err


def test_evaluate_rejects_a_candidate_from_another_training_run(tmp_path):
    paths = build_synthetic_project(tmp_path, n_per_dataset=SMALL)
    run_train(paths, git_sha="synthetic", now=FIXED_NOW)
    served = load_policy(paths.policy).predictor_type
    other = "gbm" if served == "lr" else "lr"
    candidate = paths.candidates_dir / f"{served}.json"
    candidate.write_bytes((paths.candidates_dir / f"{other}.json").read_bytes())
    with pytest.raises(evaluate_module.TrainerError, match="version"):
        run_evaluate(paths, git_sha="synthetic", now=FIXED_NOW, n_resamples=50, n_random_seeds=5)
    assert not (paths.reports / "evaluation.json").exists()


def test_evaluate_writes_pngs_markdown_and_decisions(evaluated, trained_paths):
    paths = trained_paths
    for kind in KINDS:
        png = paths.reports / f"reliability-{kind}.png"
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (paths.reports / "evaluation.md").is_file()
    assert read_json(paths.reports / "evaluation.json") == evaluated
    manifest = read_json(paths.manifest)
    n_test = manifest["files"]["test"]["rows"] // len(manifest["pool"])
    lines = (paths.replay_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(rows) == n_test * 7
    assert {r["strategy"] for r in rows} == {
        "quality",
        "strongest",
        "cheapest",
        "static",
        "latency",
    }


def test_decisions_are_per_prompt_in_order_and_match_the_strategies(evaluated, trained_paths):
    paths = trained_paths
    manifest = read_json(paths.manifest)
    policy = load_policy(paths.policy)
    table = pivot(read_outcomes(splits.split_path(paths.splits_dir, "test")), manifest["pool"])
    rows = [
        json.loads(line)
        for line in (paths.replay_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    order = ["quality"] * 3 + ["strongest", "cheapest", "static", "latency"]
    assert [r["strategy"] for r in rows[:7]] == order
    assert [r["profile"] for r in rows[:7]] == [*PROFILE_NAMES, None, None, None, None]
    assert [r["prompt_id"] for r in rows[::7]] == list(table.prompt_ids)
    assert len({r["prompt_id"] for r in rows}) == len(table)
    for i, prompt_id in enumerate(table.prompt_ids):
        block = rows[7 * i : 7 * i + 7]
        assert {r["prompt_id"] for r in block} == {prompt_id}
        assert {r["prompt_sha256"] for r in block} == {prompt_sha256(table.texts[i])}
        assert all(r["chosen"] in SYNTH_POOL for r in block)
        assert all(r["reason"] in (REASON_PASSING, REASON_ESCALATED) for r in block[:3])
        assert all(r["reason"] is None for r in block[3:])
        assert block[3]["chosen"] == policy.strongest_model
    pool_index = {name: j for j, name in enumerate(SYNTH_POOL)}
    for profile in PROFILE_NAMES:
        rows_p = [r for r in rows if r["strategy"] == "quality" and r["profile"] == profile]
        quality = evaluated["strategies"][f"quality/{evaluated['served_predictor']}/{profile}"]
        for model, share in quality["shares"].items():
            got = sum(1 for r in rows_p if r["chosen"] == model) / len(rows_p)
            assert got == pytest.approx(share)
        assert all(r["chosen"] in pool_index for r in rows_p)
    static = evaluated["strategies"]["static_keyword"]["shares"]
    static_rows = [r for r in rows if r["strategy"] == "static"]
    for model, share in static.items():
        assert sum(1 for r in static_rows if r["chosen"] == model) / len(static_rows) == (
            pytest.approx(share)
        )


def test_lodo_has_one_entry_per_dataset(evaluated):
    lodo = evaluated["lodo"]
    assert set(lodo) == {"datasets", "pooled"}
    assert sorted(lodo["datasets"]) == ["alpha", "beta", "delta", "gamma"]
    pooled = lodo["pooled"]
    assert pooled["ci_low"] <= pooled["ci_high"]
    assert pooled["profile"] == "balanced" and pooled["n"] == evaluated["test"]["n"]
    for dataset, entry in lodo["datasets"].items():
        assert entry["n"] == evaluated["test"]["by_dataset"][dataset]
        assert entry["strongest_model"] in SYNTH_POOL
        assert set(entry["lambdas"]) == set(PROFILE_NAMES)
        assert set(entry["quality"]) == set(PROFILE_NAMES)
        assert entry["quality"]["balanced"]["n"] == entry["n"]
        assert entry["always_strongest"]["regret_rate"] == 0.0
        assert entry["always_strongest"]["shares"][entry["strongest_model"]] == 1.0


def test_lodo_retrains_only_the_served_kind_with_the_main_lr_c(tmp_path, monkeypatch):
    paths = build_synthetic_project(tmp_path, n_per_dataset=SMALL)
    training = run_train(paths, git_sha="synthetic", now=FIXED_NOW)
    calls = []
    real = evaluate_module.fit_pipeline

    def spy(train, calibration, prices, **kwargs):
        calls.append((set(train.datasets.tolist()), kwargs))
        return real(train, calibration, prices, **kwargs)

    monkeypatch.setattr(evaluate_module, "fit_pipeline", spy)
    report = run_evaluate(
        paths, git_sha="synthetic", now=FIXED_NOW, n_resamples=100, n_random_seeds=5
    )
    served = report["served_predictor"]
    assert len(calls) == 4
    for held_out, (datasets, kwargs) in zip(
        ["alpha", "beta", "delta", "gamma"], calls, strict=True
    ):
        assert held_out not in datasets and len(datasets) == 3
        assert tuple(kwargs["kinds"]) == (served,)
        if served == "lr":
            assert kwargs["lr_c"] == training["lr"]["c_by_model"]
        else:
            assert kwargs.get("lr_c") is None


def _comparison(ci_high):
    return {
        "cost_per_success_diff": {
            "estimate": -0.1,
            "ci_low": -0.2,
            "ci_high": ci_high,
            "p_value": 0.01,
        }
    }


def h1_report(*, estimate=0.3, rand_high=-0.01, oracle_high=-0.02, regret_low=0.01, lodo_low=0.1):
    return {
        "served_predictor": "lr",
        "statistics": {
            "balanced": {
                "baselines": {
                    "always_strongest": {
                        "relative_cps_reduction": {
                            "estimate": estimate,
                            "ci_low": 0.2,
                            "ci_high": 0.4,
                        }
                    },
                    "share_matched_random/balanced": _comparison(rand_high),
                    "static_oracle_family": _comparison(oracle_high),
                },
                "regret": {
                    "alpha": 0.05,
                    "rate": 0.03,
                    "count": 5,
                    "n": 150,
                    "ci_low": regret_low,
                    "ci_high": 0.08,
                },
            }
        },
        "lodo": {"pooled": {"estimate": 0.2, "ci_low": lodo_low, "ci_high": 0.3}},
    }


def test_h1_verdict_rules():
    ok = h1_verdict(h1_report())
    assert ok["falsified"] is False and ok["falsified_by"] == []
    assert ok["profile"] == "balanced" and ok["predictor"] == "lr"
    assert ok["target"] == 0.25 and ok["target_met"] is True
    assert ok["cps_reduction_vs_strongest"] == {"estimate": 0.3, "ci_low": 0.2, "ci_high": 0.4}
    assert ok["vs_share_matched_random"]["beats"] is True
    assert ok["vs_static_oracle_family"]["beats"] is True
    assert ok["regret_exceeds_alpha_beyond_ci"] is False
    assert ok["lodo_gains_disappear"] is False
    assert ok["regret"]["rate"] == 0.03 and ok["regret"]["count"] == 5 and ok["regret"]["n"] == 150

    bad = h1_verdict(h1_report(rand_high=0.0))
    assert bad["falsified"] is True and bad["falsified_by"] == ["share_matched_random"]
    assert bad["vs_share_matched_random"]["beats"] is False
    assert bad["vs_share_matched_random"]["cost_per_success_diff"]["ci_high"] == 0.0


def test_h1_verdict_names_each_falsifying_condition():
    assert h1_verdict(h1_report(oracle_high=0.0))["falsified_by"] == ["static_oracle_family"]
    assert h1_verdict(h1_report(regret_low=0.0501))["falsified_by"] == ["regret_exceeds_alpha"]
    assert h1_verdict(h1_report(lodo_low=0.0))["falsified_by"] == ["lodo_gains_disappear"]
    several = h1_verdict(h1_report(rand_high=0.1, oracle_high=0.1, lodo_low=-0.1))
    assert several["falsified"] is True
    assert several["falsified_by"] == [
        "share_matched_random",
        "static_oracle_family",
        "lodo_gains_disappear",
    ]


def test_h1_verdict_edges():
    exactly_alpha = h1_verdict(h1_report(regret_low=0.05))
    assert (
        exactly_alpha["regret_exceeds_alpha_beyond_ci"] is False and not exactly_alpha["falsified"]
    )
    just_above = h1_verdict(h1_report(lodo_low=1e-9))
    assert just_above["lodo_gains_disappear"] is False and not just_above["falsified"]


def test_h1_target_is_reported_but_does_not_decide_falsification():
    missed = h1_verdict(h1_report(estimate=0.2))
    assert missed["target_met"] is False and missed["falsified"] is False
    assert h1_verdict(h1_report(estimate=0.25))["target_met"] is True


def test_h1_verdict_treats_undefined_intervals_as_not_beaten():
    report = h1_report()
    comparison = report["statistics"]["balanced"]["baselines"]["share_matched_random/balanced"]
    comparison["cost_per_success_diff"]["ci_high"] = None
    report["lodo"]["pooled"]["ci_low"] = None
    got = h1_verdict(report)
    assert got["falsified"] is True
    assert got["falsified_by"] == ["share_matched_random", "lodo_gains_disappear"]


def test_h1_in_the_report_follows_the_verdict_rule(evaluated):
    assert evaluated["h1"] == h1_verdict(evaluated)
    h1 = evaluated["h1"]
    assert h1["profile"] == "balanced" and h1["predictor"] == evaluated["served_predictor"]
    assert h1["falsified"] == bool(h1["falsified_by"])


def test_per_dataset_summaries_are_consistent_with_overall(evaluated):
    for name, overall in evaluated["strategies"].items():
        successes = sum(
            evaluated["per_dataset"][d][name]["successes"] for d in evaluated["per_dataset"]
        )
        assert successes == pytest.approx(overall["successes"])


def test_costs_section_matches_the_test_split(evaluated, trained_paths):
    manifest = read_json(trained_paths.manifest)
    rows = list(read_outcomes(splits.split_path(trained_paths.splits_dir, "test")))
    costs = evaluated["costs"]
    assert set(costs) == {"by_model", "by_dataset_model"}
    for model in SYNTH_POOL:
        recorded = sum(r.cost_usd for r in rows if r.model == model)
        assert costs["by_model"][model]["recorded_usd"] == pytest.approx(recorded)
        assert costs["by_model"][model]["list_usd"] > 0
    cells = {(c["dataset"], c["model"]) for c in costs["by_dataset_model"]}
    assert cells == {(d, m) for d in manifest["datasets"] for m in SYNTH_POOL}
    for cell in costs["by_dataset_model"]:
        assert cell["ratio"] == pytest.approx(cell["recorded_usd"] / cell["list_usd"])


def test_notes_and_extraction_status(evaluated):
    from routeforge_train.report_md import COST_NOTE, GUARANTEE_NOTE, LATENCY_NOTE, TEMPLATE_NOTE

    assert evaluated["notes"] == [GUARANTEE_NOTE, TEMPLATE_NOTE, LATENCY_NOTE, COST_NOTE]
    assert evaluated["extraction"]["status"] == "not_run"
    assert evaluated["extraction"]["note"] == EXTRACTION_NOTE


def test_extraction_status_reflects_the_label_file(tmp_path):
    from routeforge_train.paths import Paths

    paths = Paths(tmp_path)
    assert extraction_status(paths)["status"] == "not_run"
    paths.extraction_dir.mkdir(parents=True)
    (paths.extraction_dir / "outcomes.jsonl").write_text("{}\n", encoding="utf-8")
    got = extraction_status(paths)
    assert got["status"] == "labels_present_not_evaluated"
    assert got["note"] and got["note"] != EXTRACTION_NOTE


def test_evaluation_json_is_valid_strict_json(trained_paths, evaluated):
    text = (trained_paths.reports / "evaluation.json").read_text(encoding="utf-8")
    for token in ("NaN", "Infinity"):
        assert token not in text
    assert text.endswith("\n") and "\u2014" not in text


def test_evaluate_is_deterministic(tmp_path):
    paths = build_synthetic_project(tmp_path, n_per_dataset=SMALL)
    run_train(paths, git_sha="synthetic", now=FIXED_NOW)
    first = run_evaluate(
        paths, git_sha="synthetic", now=FIXED_NOW, n_resamples=100, n_random_seeds=5
    )
    names = [
        paths.reports / "evaluation.json",
        paths.reports / "evaluation.md",
        paths.reports / "reliability-lr.png",
        paths.reports / "reliability-gbm.png",
        paths.reports / "reliability-knn.png",
        paths.replay_dir / "decisions.jsonl",
    ]
    before = {p: sha256_file(p) for p in names}
    second = run_evaluate(
        paths, git_sha="synthetic", now=FIXED_NOW, n_resamples=100, n_random_seeds=5
    )
    assert second == first
    assert {p: sha256_file(p) for p in names} == before


def test_evaluate_never_changes_the_development_inputs(tmp_path):
    paths = build_synthetic_project(tmp_path, n_per_dataset=SMALL)
    run_train(paths, git_sha="synthetic", now=FIXED_NOW)
    watched = [paths.policy, paths.manifest, *sorted(paths.splits_dir.glob("*.jsonl"))]
    watched += sorted(paths.candidates_dir.glob("*.json"))
    before = {p: sha256_file(p) for p in watched}
    run_evaluate(paths, git_sha="synthetic", now=FIXED_NOW, n_resamples=50, n_random_seeds=5)
    assert {p: sha256_file(p) for p in watched} == before


def test_cli_evaluate_prints_the_verdict(tmp_path, capsys, monkeypatch):
    paths = build_synthetic_project(tmp_path, n_per_dataset=SMALL)
    for falsified, word in ((False, "supported"), (True, "falsified")):
        monkeypatch.setattr(
            evaluate_module,
            "run_evaluate",
            lambda _paths, f=falsified: {"h1": {"falsified": f}},
        )
        assert main(["--root", str(paths.root), "evaluate"]) == 0
        assert capsys.readouterr().out == (
            f"evaluate OK: trainer/artifacts/reports/evaluation.json; H1 {word}\n"
        )


def test_make_evaluate_target_runs_the_evaluate_command():
    from pathlib import Path

    makefile = (Path(__file__).resolve().parents[2] / "Makefile").read_text(encoding="utf-8")
    phony = next(line for line in makefile.splitlines() if line.startswith(".PHONY:"))
    assert "evaluate" in phony.split()
    assert "\nevaluate:\n\t$(TRAIN) evaluate\n" in makefile
