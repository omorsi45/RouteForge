from itertools import pairwise

import numpy as np
import pytest

from routeforge_train.calibration import (
    MIN_BLOCK,
    WILSON_Z,
    Block,
    Calibration,
    Platt,
    fit_calibration,
    fit_platt,
    wilson_lcb,
)


def _groups(*groups):
    """Build (raw, success) from (raw_value, n, k) groups."""
    raw = np.concatenate([np.full(n, r) for r, n, _ in groups])
    success = np.concatenate(
        [np.r_[np.ones(k, dtype=bool), np.zeros(n - k, dtype=bool)] for _, n, k in groups]
    )
    return raw, success


def _nk(cal):
    return [(b.n, b.k) for b in cal.blocks]


def test_constants():
    assert (MIN_BLOCK, WILSON_Z) == (40, 1.645)


def test_wilson_lcb_known_values():
    assert wilson_lcb(412, 450) == pytest.approx(0.8914311779213744, abs=1e-15)
    assert wilson_lcb(0, 10) == 0.0
    assert wilson_lcb(10, 10) == pytest.approx(0.7870282011880192, abs=1e-15)
    assert wilson_lcb(10, 10) < 1.0
    assert wilson_lcb(40, 40) == pytest.approx(0.936635989886673, abs=1e-15)


def test_wilson_lcb_rejects_empty_sample():
    with pytest.raises(ValueError, match="n"):
        wilson_lcb(0, 0)


def test_wilson_lcb_zero_successes_is_exactly_zero():
    assert all(wilson_lcb(0, n) == 0.0 for n in range(1, 5001))


def test_wilson_lcb_never_exceeds_p_and_stays_in_unit_interval():
    for n in (1, 2, 3, 7, 40, 41, 100, 450, 1000, 5000):
        ks = sorted({0, 1, 2, n // 3, n // 2, n - 2, n - 1, n} & set(range(n + 1)))
        for k in ks:
            lcb = wilson_lcb(k, n)
            assert 0.0 <= lcb <= k / n
            assert lcb < 1.0


def test_fit_calibration_blocks_have_min_40_and_monotone_p():
    rng = np.random.default_rng(5)
    raw = rng.random(500)
    success = rng.random(500) < raw
    cal = fit_calibration(raw, success)
    blocks = cal.blocks
    assert all(b.n >= 40 for b in blocks)
    assert all(a.p <= b.p for a, b in pairwise(blocks))
    assert all(a.lo < b.lo for a, b in pairwise(blocks))
    assert blocks[0].lo == 0.0 and blocks[-1].hi == 1.0
    assert all(a.hi == b.lo for a, b in pairwise(blocks))
    assert all(b.p == b.k / b.n for b in blocks)
    assert sum(b.n for b in blocks) == 500
    assert sum(b.k for b in blocks) == int(success.sum())
    assert all(b.lcb == wilson_lcb(b.k, b.n) for b in blocks)


def test_fit_calibration_never_splits_identical_scores():
    raw = np.full(100, 0.5)
    success = np.arange(100) % 2 == 0
    cal = fit_calibration(raw, success)
    assert _nk(cal) == [(100, 50)]


def test_fit_calibration_pools_violating_groups_before_merging():
    raw, success = _groups((0.2, 50, 30), (0.6, 50, 20))
    cal = fit_calibration(raw, success)
    assert _nk(cal) == [(100, 50)]


def test_fit_calibration_merge_prefers_closer_p():
    raw, success = _groups((0.1, 50, 10), (0.5, 10, 5), (0.9, 50, 30))
    cal = fit_calibration(raw, success)
    assert _nk(cal) == [(50, 10), (60, 35)]
    assert [b.lo for b in cal.blocks] == [0.0, 0.5]


def test_fit_calibration_merge_ties_go_left():
    raw, success = _groups((0.1, 50, 10), (0.5, 10, 4), (0.9, 50, 30))
    cal = fit_calibration(raw, success)
    assert _nk(cal) == [(60, 14), (50, 30)]
    assert [b.lo for b in cal.blocks] == [0.0, 0.9]


def test_fit_calibration_first_and_last_undersized_blocks_merge_inward():
    raw, success = _groups((0.1, 10, 1), (0.5, 50, 25), (0.9, 10, 9))
    cal = fit_calibration(raw, success)
    assert all(b.n >= 40 for b in cal.blocks)
    assert sum(b.n for b in cal.blocks) == 70


def test_fit_calibration_small_sample_single_block():
    rng = np.random.default_rng(1)
    raw = rng.random(30)
    success = rng.random(30) < 0.5
    cal = fit_calibration(raw, success)
    assert len(cal.blocks) == 1
    assert cal.blocks[0].n == 30
    assert (cal.blocks[0].lo, cal.blocks[0].hi) == (0.0, 1.0)


def test_calibration_lookup_boundaries():
    cal = Calibration(
        (
            Block(0.0, 0.3, 40, 10, 0.25, 0.1),
            Block(0.3, 0.7, 40, 20, 0.5, 0.4),
            Block(0.7, 1.0, 40, 30, 0.75, 0.6),
        )
    )
    raw = np.array([0.0, 0.2999999, 0.3, 0.7, 1.0])
    assert cal.lookup(raw).tolist() == [0, 0, 1, 2, 2]
    assert cal.p_of(raw).tolist() == [0.25, 0.25, 0.5, 0.75, 0.75]
    assert cal.lcb_of(raw).tolist() == [0.1, 0.1, 0.4, 0.6, 0.6]


def test_calibration_lookup_clips_scores_below_first_lo():
    cal = Calibration((Block(0.0, 1.0, 40, 20, 0.5, 0.4),))
    assert cal.lookup(np.array([-0.5])).tolist() == [0]


def test_calibration_json_round_trip():
    raw, success = _groups((0.1, 50, 10), (0.5, 50, 25), (0.9, 50, 40))
    cal = fit_calibration(raw, success)
    doc = cal.to_json()
    assert set(doc) == {"blocks"}
    assert set(doc["blocks"][0]) == {"lo", "hi", "n", "k", "p", "lcb"}
    assert Calibration.from_json(doc) == cal


def test_platt_fits_monotone():
    rng = np.random.default_rng(3)
    raw = rng.random(400)
    success = rng.random(400) < raw
    platt = fit_platt(raw, success)
    assert isinstance(platt, Platt)
    grid = np.linspace(0.01, 0.99, 50)
    out = platt.predict(grid)
    assert np.all(np.diff(out) > 0)
    assert np.all((out > 0) & (out < 1))
