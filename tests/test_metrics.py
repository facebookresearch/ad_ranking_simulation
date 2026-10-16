# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Behavioral tests for decision metrics and the cold-start diagnostic."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from adsim.metrics import (
    cold_start_mass,
    cold_start_masses,
    COLD_START_TAUS,
    compute_decision_metrics,
    DecisionMetrics,
)

NAMES = ["p1", "p2", "p3"]
V_LOG = 0.10
N_CLASSES = 7


def _metrics(
    v_true: dict[str, float],
    v_hat: dict[str, float],
    v_log_hat: float = V_LOG,
    **kwargs: float,
) -> DecisionMetrics:
    return compute_decision_metrics(v_true, v_hat, V_LOG, v_log_hat, NAMES, **kwargs)


class DecisionMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        # p3 is truly best, p1 truly worst.
        self.v_true = {"p1": 0.09, "p2": 0.11, "p3": 0.13}

    def test_perfect_estimator_scores_perfectly(self) -> None:
        result = _metrics(self.v_true, self.v_true)
        self.assertEqual(1, result["acc1"])
        self.assertAlmostEqual(0.0, result["regret"])
        self.assertAlmostEqual(1.0, result["dir_acc"])
        self.assertAlmostEqual(0.0, result["mae_v"])
        self.assertAlmostEqual(0.0, result["mae_delta"])

    def test_picking_the_wrong_candidate_costs_regret(self) -> None:
        """Regret is the true lift forgone, not the estimation error."""
        v_hat = {"p1": 0.20, "p2": 0.11, "p3": 0.13}
        result = _metrics(self.v_true, v_hat)
        self.assertEqual(0, result["acc1"])
        self.assertAlmostEqual(0.13 - 0.09, result["regret"], places=9)

    def test_a_shared_additive_bias_does_not_change_the_decision(self) -> None:
        """Every metric except mae_v is a lift, so a common offset cancels."""
        shifted = {p: v + 0.05 for p, v in self.v_true.items()}
        result = _metrics(self.v_true, shifted, v_log_hat=V_LOG + 0.05)
        self.assertEqual(1, result["acc1"])
        self.assertAlmostEqual(0.0, result["regret"])
        self.assertAlmostEqual(0.0, result["mae_delta"])
        self.assertAlmostEqual(0.05, result["mae_v"], places=9)

    def test_direction_accuracy_counts_side_of_the_logging_policy(self) -> None:
        """p1 is truly below logging; calling it above is one directional miss."""
        v_hat = {"p1": 0.12, "p2": 0.11, "p3": 0.13}
        self.assertAlmostEqual(2.0 / 3.0, _metrics(self.v_true, v_hat)["dir_acc"])

    def test_candidates_at_the_logging_value_are_excluded_from_direction(self) -> None:
        v_true = {"p1": V_LOG, "p2": 0.11, "p3": 0.13}
        v_hat = {"p1": 0.30, "p2": 0.11, "p3": 0.13}
        self.assertAlmostEqual(1.0, _metrics(v_true, v_hat)["dir_acc"])

    def test_direction_accuracy_is_nan_when_nothing_clears_tau(self) -> None:
        flat = {p: V_LOG for p in NAMES}
        self.assertTrue(np.isnan(_metrics(flat, flat)["dir_acc"]))

    def test_tau_controls_which_candidates_count(self) -> None:
        v_true = {"p1": V_LOG + 1e-5, "p2": 0.11, "p3": 0.13}
        v_hat = {"p1": V_LOG - 1e-5, "p2": 0.11, "p3": 0.13}
        loose = _metrics(v_true, v_hat, tau=1e-6)["dir_acc"]
        strict = _metrics(v_true, v_hat, tau=1e-4)["dir_acc"]
        self.assertAlmostEqual(2.0 / 3.0, loose)
        self.assertAlmostEqual(1.0, strict)

    def test_mae_v_is_absolute_level_error(self) -> None:
        v_hat = {p: v + 0.02 for p, v in self.v_true.items()}
        self.assertAlmostEqual(0.02, _metrics(self.v_true, v_hat)["mae_v"], places=9)

    def test_per_policy_baseline_is_applied_per_candidate(self) -> None:
        """Each candidate's lift is measured against its own model's baseline."""
        v_hat = {"p1": 0.20, "p2": 0.21, "p3": 0.22}
        per_policy = {"p1": 0.19, "p2": 0.20, "p3": 0.21}
        result = compute_decision_metrics(self.v_true, v_hat, V_LOG, per_policy, NAMES)
        # Every candidate has the same lift of 0.01, so argmax falls to the first.
        self.assertEqual(0, result["acc1"])

    def test_per_policy_and_shared_baselines_agree_when_constant(self) -> None:
        v_hat = {"p1": 0.11, "p2": 0.12, "p3": 0.14}
        shared = _metrics(self.v_true, v_hat, v_log_hat=0.10)
        per_policy = compute_decision_metrics(
            self.v_true, v_hat, V_LOG, {p: 0.10 for p in NAMES}, NAMES
        )
        self.assertEqual(dict(shared), dict(per_policy))

    def test_policy_order_does_not_change_the_score(self) -> None:
        v_hat = {"p1": 0.11, "p2": 0.12, "p3": 0.14}
        forward = compute_decision_metrics(
            self.v_true, v_hat, V_LOG, V_LOG, ["p1", "p2", "p3"]
        )
        reverse = compute_decision_metrics(
            self.v_true, v_hat, V_LOG, V_LOG, ["p3", "p2", "p1"]
        )
        self.assertEqual(forward["acc1"], reverse["acc1"])
        self.assertAlmostEqual(forward["regret"], reverse["regret"])
        self.assertAlmostEqual(forward["mae_v"], reverse["mae_v"])


class ColdStartMassTest(unittest.TestCase):
    def _probs(self, p_logging: float) -> torch.Tensor:
        rest = (1.0 - p_logging) / (N_CLASSES - 1)
        row = [p_logging] + [rest] * (N_CLASSES - 1)
        return torch.tensor([row] * 100)

    def test_all_impressions_well_covered_gives_zero(self) -> None:
        self.assertEqual(0.0, cold_start_mass(self._probs(0.5), tau=0.05))

    def test_all_impressions_uncovered_gives_one(self) -> None:
        self.assertEqual(1.0, cold_start_mass(self._probs(0.001), tau=0.05))

    def test_reports_the_fraction_below_tau(self) -> None:
        probs = torch.cat([self._probs(0.001)[:25], self._probs(0.5)[:75]])
        self.assertAlmostEqual(0.25, cold_start_mass(probs, tau=0.05))

    def test_is_monotone_in_tau(self) -> None:
        gen = torch.Generator().manual_seed(9)
        probs = torch.softmax(torch.randn(500, N_CLASSES, generator=gen) * 2.0, dim=-1)
        masses = [cold_start_mass(probs, tau) for tau in (0.01, 0.05, 0.1)]
        self.assertLessEqual(masses[0], masses[1])
        self.assertLessEqual(masses[1], masses[2])

    def test_respects_a_non_default_logging_class(self) -> None:
        # Class 0 holds 0.5; the remaining six split the rest at ~0.083 each.
        probs = self._probs(0.5)
        self.assertEqual(0.0, cold_start_mass(probs, tau=0.1, logging_class=0))
        self.assertEqual(1.0, cold_start_mass(probs, tau=0.1, logging_class=1))

    def test_masses_covers_every_default_threshold(self) -> None:
        gen = torch.Generator().manual_seed(9)
        probs = torch.softmax(torch.randn(200, N_CLASSES, generator=gen) * 2.0, dim=-1)
        masses = cold_start_masses(probs)
        self.assertEqual(set(COLD_START_TAUS), set(masses.keys()))
        for tau, value in masses.items():
            self.assertEqual(cold_start_mass(probs, tau), value)
