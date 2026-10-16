# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for figure aggregation and rendering.

Aggregation is tested on its numbers; rendering is tested only for producing a
non-empty image, since asserting on pixels would be brittle without catching
anything a reader would care about.
"""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np
from adsim.figures import (
    clip_sensitivity_curves,
    plot_clip_sensitivity,
)


def _clip_record(
    clip: float, kernel_ess: float, **overrides: float
) -> dict[str, float]:
    record = {
        "clip": clip,
        "kernel_ess": kernel_ess,
        "regret_u": 0.004,
        "regret_w": 0.002,
        "acc1_u": 0.6,
        "acc1_w": 0.8,
        "mae_v_u": 0.03,
        "mae_v_w": 0.02,
    }
    record.update(overrides)
    return record


class ClipSensitivityCurvesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            _clip_record(1.0, 0.31),
            _clip_record(1.0, 0.72),
            _clip_record(20.0, 0.29),
            _clip_record(20.0, 0.68),
        ]

    def test_one_curve_per_clip(self) -> None:
        curves = clip_sensitivity_curves(self.records, "regret")
        self.assertEqual({1.0, 20.0}, set(curves["clips"]))

    def test_bins_by_kernel_overlap(self) -> None:
        curves = clip_sensitivity_curves(self.records, "regret")
        self.assertEqual([0.3, 0.7], [round(c, 10) for c in curves["ess"]])

    def test_curves_and_bins_are_the_same_length(self) -> None:
        curves = clip_sensitivity_curves(self.records, "acc1")
        for clip, curve in curves["clips"].items():
            with self.subTest(clip=clip):
                self.assertEqual(len(curves["ess"]), len(curve["mean"]))
        self.assertEqual(len(curves["ess"]), len(curves["unweighted"]["mean"]))

    def test_unweighted_baseline_is_counted_once_not_once_per_clip(self) -> None:
        """The unweighted model is clip-independent; averaging it over every
        clip's copy of the same record would understate its error bars."""
        curves = clip_sensitivity_curves(self.records, "regret")
        self.assertAlmostEqual(0.004, curves["unweighted"]["mean"][0])
        self.assertEqual(0.0, curves["unweighted"]["se"][0])

    def test_empty_bin_is_nan_not_an_error(self) -> None:
        records = [_clip_record(1.0, 0.3), _clip_record(20.0, 0.7)]
        curves = clip_sensitivity_curves(records, "regret")
        values = curves["clips"][1.0]["mean"]
        self.assertTrue(any(np.isnan(v) for v in values))


class RenderingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clip_records = [
            _clip_record(clip, ess) for clip in (1.0, 20.0) for ess in (0.3, 0.5, 0.7)
        ]

    def _render(
        self,
        plot: Callable[[list[dict[str, float]], str], None],
        records: list[dict[str, float]],
        name: str,
    ) -> int:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            plot(records, str(path))
            self.assertTrue(path.exists())
            return path.stat().st_size

    def test_clip_sensitivity_writes_an_image(self) -> None:
        self.assertGreater(
            self._render(plot_clip_sensitivity, self.clip_records, "clip.png"), 1000
        )
