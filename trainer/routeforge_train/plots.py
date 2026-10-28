"""Reliability diagrams of the offline evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from routeforge_train.jsonio import atomic_writer

_ROWS = 2
_COLS = 3
_DPI = 120
_CURVES = (("isotonic", "o", "isotonic"), ("platt", "s", "Platt"))


def reliability_png(
    path: Path, title: str, per_model: Mapping[str, Mapping[str, list[dict]]]
) -> None:
    """Write a 2 x 3 grid of reliability curves, one panel per model in the given order.

    Each model maps `isotonic` and `platt` to reliability bins; the dashed diagonal is perfect
    calibration. The PNG carries no Software tag, so a rerun on one machine is byte-stable.
    """
    models = list(per_model)
    if len(models) > _ROWS * _COLS:
        raise ValueError(
            f"a reliability plot shows at most {_ROWS * _COLS} models, got {len(models)}"
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        _ROWS, _COLS, figsize=(11, 7), sharex=True, sharey=True, constrained_layout=True
    )
    try:
        panels = axes.ravel()
        for i, model in enumerate(models):
            ax = panels[i]
            ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1, label="perfect")
            for key, marker, label in _CURVES:
                bins = per_model[model][key]
                ax.plot(
                    [b["mean_p"] for b in bins],
                    [b["frac_success"] for b in bins],
                    marker=marker,
                    markersize=4,
                    linewidth=1.2,
                    label=label,
                )
            ax.set_title(model, fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            if i % _COLS == 0:
                ax.set_ylabel("fraction successful")
            if i >= _COLS:
                ax.set_xlabel("mean predicted success")
        for ax in panels[len(models) :]:
            ax.set_axis_off()
        panels[0].legend(loc="upper left", fontsize=8)
        fig.suptitle(title)
        with atomic_writer(path, "wb") as f:
            fig.savefig(f, format="png", dpi=_DPI, metadata={"Software": None})
    finally:
        plt.close(fig)
