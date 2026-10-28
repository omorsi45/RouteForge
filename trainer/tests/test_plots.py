import pytest

from routeforge_train.plots import reliability_png

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def bins(*points):
    return [{"n": 10, "mean_p": p, "frac_success": f} for p, f in points]


def two_models():
    curve = bins((0.1, 0.05), (0.5, 0.55), (0.9, 0.85))
    return {
        "model-a": {"isotonic": curve, "platt": curve},
        "model-b": {"isotonic": curve, "platt": bins((0.2, 0.3), (0.6, 0.5), (0.8, 0.9))},
    }


def test_reliability_png_writes_png(tmp_path):
    path = tmp_path / "out" / "reliability-lr.png"
    reliability_png(path, "Reliability (lr)", two_models())
    assert path.read_bytes().startswith(PNG_SIGNATURE)
    assert [p.name for p in path.parent.iterdir()] == ["reliability-lr.png"]


def test_reliability_png_is_byte_stable_and_has_no_software_tag(tmp_path):
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    reliability_png(first, "Reliability (lr)", two_models())
    reliability_png(second, "Reliability (lr)", two_models())
    assert first.read_bytes() == second.read_bytes()
    assert b"Software" not in first.read_bytes() and b"matplotlib" not in first.read_bytes()


def test_reliability_png_fills_a_two_by_three_grid(tmp_path):
    curve = bins((0.1, 0.1), (0.9, 0.9))
    six = {f"m{i}": {"isotonic": curve, "platt": curve} for i in range(6)}
    path = tmp_path / "six.png"
    reliability_png(path, "six models", six)
    assert path.read_bytes().startswith(PNG_SIGNATURE)


def test_reliability_png_rejects_more_than_six_models(tmp_path):
    curve = bins((0.1, 0.1))
    seven = {f"m{i}": {"isotonic": curve, "platt": curve} for i in range(7)}
    with pytest.raises(ValueError, match="at most 6"):
        reliability_png(tmp_path / "seven.png", "seven models", seven)
    assert not (tmp_path / "seven.png").exists()


def test_reliability_png_accepts_models_without_bins(tmp_path):
    path = tmp_path / "empty.png"
    reliability_png(path, "empty", {"m": {"isotonic": [], "platt": []}})
    assert path.read_bytes().startswith(PNG_SIGNATURE)
