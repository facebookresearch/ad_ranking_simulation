# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The paper's clip-sensitivity figure.

`plot_clip_sensitivity` shows regret, accuracy, and value error against
overlap, one curve per clip value plus the unweighted baseline. It is how the
bias-variance dial reads: tight clips flatten the weights toward the unweighted
curve, loose clips let variance in at low overlap.

Aggregation is separated from drawing so the numbers behind a figure can be
tested without rendering anything.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Mapping, Sequence, TypedDict

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

CLIP_METRICS: tuple[tuple[str, str], ...] = (
    ("regret", "Top-1 regret (lower is better)"),
    ("acc1", "Top-1 accuracy (higher is better)"),
    ("mae_v", "Value MAE (lower is better)"),
)

DEFAULT_ESS_BIN_SIZE = 0.10


class Curve(TypedDict):
    """One line on a plot: a value and an error bar per bin."""

    mean: list[float]
    se: list[float]


class ClipCurves(TypedDict):
    """Shared bin centers, one weighted curve per clip, one baseline."""

    ess: list[float]
    clips: dict[float, Curve]
    unweighted: Curve


def _mean_and_se(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), 0.0
    return float(np.mean(values)), float(np.std(values) / np.sqrt(len(values)))


def clip_sensitivity_curves(
    records: Sequence[Mapping[str, Any]],
    metric: str,
    ess_bin_size: float = DEFAULT_ESS_BIN_SIZE,
) -> ClipCurves:
    """Mean and standard error of one metric per overlap bin.

    Returns the shared bin centers, one curve per clip value for the weighted
    model, and one baseline curve for the unweighted model. The unweighted
    model is clip-independent, so its records are read from a single clip to
    avoid counting it once per clip value.
    """
    clips = sorted({float(r["clip"]) for r in records})
    weighted: dict[tuple[float, float], list[Mapping[str, Any]]] = defaultdict(list)
    unweighted: dict[float, list[Mapping[str, Any]]] = defaultdict(list)

    for record in records:
        center = round(float(record["kernel_ess"]) / ess_bin_size) * ess_bin_size
        weighted[(float(record["clip"]), center)].append(record)
        if clips and float(record["clip"]) == clips[0]:
            unweighted[center].append(record)

    centers = sorted({center for _, center in weighted})
    curves: dict[float, Curve] = {}
    for clip in clips:
        means, errors = zip(
            *[
                _mean_and_se([r[f"{metric}_w"] for r in weighted.get((clip, c), [])])
                for c in centers
            ]
        )
        curves[clip] = Curve(mean=list(means), se=list(errors))

    base_means, base_errors = zip(
        *[
            _mean_and_se([r[f"{metric}_u"] for r in unweighted.get(c, [])])
            for c in centers
        ]
    )
    return ClipCurves(
        ess=centers,
        clips=curves,
        unweighted=Curve(mean=list(base_means), se=list(base_errors)),
    )


def plot_clip_sensitivity(
    records: Sequence[Mapping[str, Any]],
    path: str,
    ess_bin_size: float = DEFAULT_ESS_BIN_SIZE,
) -> None:
    """Three panels: each metric against overlap, one curve per clip."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(CLIP_METRICS), figsize=(18, 5))
    for axis, (metric, label) in zip(axes, CLIP_METRICS):
        curves = clip_sensitivity_curves(records, metric, ess_bin_size)
        centers = curves["ess"]
        baseline = curves["unweighted"]
        axis.errorbar(
            centers,
            baseline["mean"],
            yerr=baseline["se"],
            fmt="o--",
            color="#444444",
            label="unweighted",
            lw=2,
            ms=7,
            capsize=3,
            zorder=5,
        )
        clips = sorted(curves["clips"])
        colors = plt.get_cmap("plasma")(np.linspace(0.1, 0.85, len(clips)))
        for clip, color in zip(clips, colors):
            curve = curves["clips"][clip]
            axis.errorbar(
                centers,
                curve["mean"],
                yerr=curve["se"],
                fmt="o-",
                color=color,
                label=f"clip={clip:.0f}",
                lw=2,
                ms=6,
                capsize=3,
            )
        axis.set_xlabel("Kernel ESS (overlap)")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
