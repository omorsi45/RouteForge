from pathlib import Path

import pytest

from routeforge_train.errors import TrainerError
from routeforge_train.llmrouterbench import POOL
from routeforge_train.prices import Price, PriceTable, list_cost_usd, load_prices

REPO_ROOT = Path(__file__).resolve().parents[2]

VALID = """\
effective_date: "2026-10-26"
source: "test table"
currency: USD
models:
  - name: a
    input_per_mtok: 1.0
    output_per_mtok: 2
  - name: b
    input_per_mtok: 0
    output_per_mtok: 0.5
"""


def write(tmp_path, text):
    path = tmp_path / "prices.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_committed_prices():
    t = load_prices(REPO_ROOT / "deploy/prices.yaml")
    assert list(t.models) == list(POOL) and t.effective_date == "2026-10-26"
    assert t.price("gpt-5") == Price(1.25, 10.0) == t.price("gemini-2.5-pro")
    assert t.price("claude-sonnet-4") == Price(3.0, 15.0)


def test_load_prices_valid_file(tmp_path):
    t = load_prices(write(tmp_path, VALID))
    assert t == PriceTable("2026-10-26", "test table", {"a": Price(1.0, 2.0), "b": Price(0.0, 0.5)})
    assert isinstance(t.models["a"].output_per_mtok, float)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.replace("name: b", "name: a"),
        lambda s: s.replace("output_per_mtok: 0.5", "output_per_mtok: -0.5"),
        lambda s: s.replace("input_per_mtok: 0", "input_per_mtok: .nan"),
        lambda s: s.replace("input_per_mtok: 0", "input_per_mtok: .inf"),
        lambda s: s.replace("input_per_mtok: 1.0", 'input_per_mtok: "1.0"'),
        lambda s: s.replace("input_per_mtok: 1.0", "input_per_mtok: true"),
        lambda s: s.replace('"2026-10-26"', '"2026-13-45"'),
        lambda s: s.replace('"2026-10-26"', "2026-10-26"),
        lambda s: s.replace('"2026-10-26"', '"26-10-2026"'),
        lambda s: s.replace('effective_date: "2026-10-26"\n', ""),
        lambda s: s.replace('source: "test table"\n', ""),
        lambda s: s[: s.index("models:")] + "models: []\n",
        lambda s: s.replace("    output_per_mtok: 2\n", ""),
        lambda s: "- just a list\n",
        lambda s: "models: [unclosed",
    ],
)
def test_load_prices_rejects_bad_files(tmp_path, mutation):
    with pytest.raises(TrainerError):
        load_prices(write(tmp_path, mutation(VALID)))


def test_load_prices_missing_file_is_trainer_error(tmp_path):
    with pytest.raises(TrainerError, match="nope.yaml"):
        load_prices(tmp_path / "nope.yaml")


def test_price_missing_model_names_it():
    t = PriceTable("2026-10-26", "s", {"a": Price(1.0, 2.0)})
    with pytest.raises(TrainerError, match="unknown-model"):
        t.price("unknown-model")


def test_list_cost_usd():
    assert list_cost_usd(Price(1.25, 10.0), 1000, 2000) == pytest.approx(0.02125)
    assert list_cost_usd(Price(1.25, 10.0), 0, 0) == 0.0
