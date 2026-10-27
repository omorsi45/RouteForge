import numpy as np

from routeforge_train.rules import (
    KEYWORD_MAX,
    KEYWORD_MIN_DF,
    KEYWORD_MIN_PRECISION,
    KeywordRules,
    fit_keyword_rules,
)

POOL = ("A", "B", "C")


def _fit(texts, families, success=None, exp_cost=None, pool=POOL):
    n = len(texts)
    if success is None:
        success = np.ones((n, len(pool)), dtype=bool)
    if exp_cost is None:
        exp_cost = np.ones((n, len(pool)))
    return fit_keyword_rules(texts, families, success, exp_cost, pool)


def _three_families():
    texts = (
        [f"the solve integral q{i}" for i in range(40)]
        + [f"the python function q{i}" for i in range(40)]
        + [f"the shared words q{i}" for i in range(20)]
    )
    families = ["reasoning"] * 40 + ["code"] * 40 + ["qa"] * 20
    return texts, families


def test_constants():
    assert (KEYWORD_MIN_DF, KEYWORD_MIN_PRECISION, KEYWORD_MAX) == (20, 0.9, 25)


def test_keyword_rules_learn_family_tokens_and_classify():
    texts, families = _three_families()
    rules = _fit(texts, families)
    assert "python" in rules.family_keywords["code"]
    assert "solve" in rules.family_keywords["reasoning"]
    assert "the" not in {k for kws in rules.family_keywords.values() for k in kws}
    assert rules.classify("write a python function") == "code"
    assert rules.classify("solve an integral") == "reasoning"


def test_keyword_rules_respect_min_df_and_precision():
    texts = (
        [f"common q{i}" for i in range(19)]
        + [f"mixed q{i}" for i in range(30)]
        + [f"mixed q{i}" for i in range(10)]
        + [f"edge q{i}" for i in range(36)]
        + [f"edge q{i}" for i in range(4)]
    )
    families = ["a"] * 49 + ["b"] * 10 + ["a"] * 36 + ["b"] * 4
    rules = _fit(texts, families)
    # common: df 19 < 20. mixed: family a has 30 of 40 = 0.75 < 0.9.
    # edge: family a has 36 of 40 = 0.9 exactly, which qualifies.
    assert rules.family_keywords["a"] == ("edge",)
    assert rules.family_keywords["b"] == ()


def test_keyword_rules_cap_and_sort_by_family_df_then_token():
    tokens = [f"t{i:02d}" for i in range(30)]
    texts, families = [], []
    for i in range(60):
        texts.append(" ".join(tokens) + f" u{i}")
        families.append("a")
    # make t29 more frequent than the rest by adding it to extra prompts
    for i in range(10):
        texts.append(f"t29 u{100 + i} filler{i}")
        families.append("a")
    rules = _fit(texts, families)
    kws = rules.family_keywords["a"]
    assert len(kws) == KEYWORD_MAX
    assert kws[0] == "t29"
    assert kws[1:] == tuple(f"t{i:02d}" for i in range(24))


def test_keyword_rules_no_match_uses_default_family():
    texts, families = _three_families()
    rules = _fit(texts, families)
    assert rules.family_order == ("code", "reasoning", "qa")
    assert rules.classify("zzz") == rules.default_family == rules.family_order[0]


def test_keyword_rules_ties_follow_family_order():
    texts, families = _three_families()
    rules = _fit(texts, families)
    expected = next(f for f in rules.family_order if f in {"code", "reasoning"})
    assert rules.classify("solve python") == expected
    assert rules.classify("python solve") == expected


def test_keyword_rules_most_keywords_wins():
    texts, families = _three_families()
    rules = _fit(texts, families)
    assert rules.classify("solve integral python") == "reasoning"


def test_family_model_cheapest_within_two_points():
    n = 100
    texts = [f"hello world q{i}" for i in range(n)]
    success = np.zeros((n, 3), dtype=bool)
    for col, k in enumerate((80, 78, 77)):
        success[:k, col] = True
    exp_cost = np.tile(np.array([3.0, 1.0, 0.1]), (n, 1))
    rules = _fit(texts, ["code"] * n, success, exp_cost)
    assert rules.family_model == {"code": "B"}


def test_family_model_cost_ties_follow_pool_order():
    n = 50
    texts = [f"hello world q{i}" for i in range(n)]
    rules = _fit(texts, ["code"] * n)
    assert rules.family_model == {"code": "A"}


def test_keyword_rules_json_round_trip():
    texts, families = _three_families()
    rules = _fit(texts, families)
    doc = rules.to_json()
    assert isinstance(doc["family_keywords"]["code"], list)
    assert KeywordRules.from_json(doc) == rules
