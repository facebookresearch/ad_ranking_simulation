# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Behavioral tests for candidate positioning by value and overlap."""

from __future__ import annotations

import unittest

import torch
from adsim.calibration import (
    calibration_sweep,
    compute_pairwise_density_weights,
    lookup_alpha_beta,
    SweepPoint,
)
from adsim.kernels import (
    score_pairs,
    top1_from_slate,
)
from adsim.oracle import make_oracle
from adsim.policy import make_policy
from adsim.world import make_world

NUM_USERS = 1_500
NUM_ADS = 400
USER_DIM = 3
AD_DIM = 3
N_IMPRESSIONS = 600
TOP_K = 10

N_KERNELS = 64
DELTA = 0.3
SIGMA_RANGE = (0.3, 0.5)
ALPHA_RANGE = (0.1, 1.0)
CLIP = 100.0

# The sweep is the slowest thing in this file and every assertion below reads
# the same one, so it is computed once per test process.
_SWEEP_CACHE: list[SweepPoint] | None = None


class CalibrationTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.users, self.ads = make_world(NUM_USERS, NUM_ADS, USER_DIM, AD_DIM)
        self.oracle = make_oracle(0.15, self.users, self.ads, n_kernels=128)

        gen = torch.Generator().manual_seed(5000)
        self.sampled_users = self.users[
            torch.randint(0, NUM_USERS, (N_IMPRESSIONS,), generator=gen)
        ]
        self.candidate_ads = self.ads[
            torch.randint(0, NUM_ADS, (N_IMPRESSIONS, TOP_K), generator=gen)
        ]

        log_center = torch.full((USER_DIM + AD_DIM,), 0.2)
        self.log_policy = make_policy(
            N_KERNELS,
            log_center[:USER_DIM],
            log_center[USER_DIM:],
            DELTA,
            SIGMA_RANGE,
            ALPHA_RANGE,
            seed=242,
        )
        self.log_top1_ads = top1_from_slate(
            self.log_policy, self.sampled_users, self.candidate_ads, TOP_K
        )
        with torch.no_grad():
            self.v_log = (
                score_pairs(self.oracle, self.sampled_users, self.log_top1_ads)
                .mean()
                .item()
            )

        self.oracle_center = torch.cat(
            [self.oracle["center_u"], self.oracle["center_a"]]
        )
        diff = log_center - self.oracle_center
        self.distance = diff.norm().item()
        self.v_along = diff / self.distance
        rng = torch.Generator().manual_seed(999)
        w = torch.randn(USER_DIM + AD_DIM, generator=rng)
        w = w - (w @ self.v_along) * self.v_along
        self.w_orth = w / w.norm()

    def _run_sweep(self, calib_n_samples: int = 300) -> list[SweepPoint]:
        """Run the sweep for real. Slow, so most tests use the cached copy."""
        return calibration_sweep(
            self.oracle,
            self.log_policy,
            self.sampled_users,
            self.candidate_ads,
            self.log_top1_ads,
            self.oracle_center,
            self.v_along,
            self.w_orth,
            self.distance,
            self.v_log,
            N_KERNELS,
            DELTA,
            SIGMA_RANGE,
            ALPHA_RANGE,
            TOP_K,
            seed=42,
            clip=CLIP,
            calib_n_samples=calib_n_samples,
        )

    def _sweep(self) -> list[SweepPoint]:
        """Cached sweep shared by every read-only assertion."""
        global _SWEEP_CACHE
        if _SWEEP_CACHE is None:
            _SWEEP_CACHE = self._run_sweep()
        return _SWEEP_CACHE


class PairwiseDensityWeightsTest(CalibrationTestBase):
    def test_identical_policies_give_unit_weights_and_full_ess(self) -> None:
        """No shift means every weight is 1 and ESS saturates."""
        weights, _, info = compute_pairwise_density_weights(
            self.log_policy,
            self.log_policy,
            self.sampled_users,
            self.log_top1_ads,
            self.sampled_users,
            self.log_top1_ads,
            clip=CLIP,
        )
        self.assertAlmostEqual(1.0, weights.mean().item(), places=4)
        self.assertAlmostEqual(1.0, info["ess"], places=4)

    def test_ess_is_a_normalized_fraction(self) -> None:
        _, _, info = compute_pairwise_density_weights(
            self.log_policy,
            self.oracle,
            self.sampled_users,
            self.log_top1_ads,
            self.sampled_users,
            self.log_top1_ads,
            clip=CLIP,
        )
        self.assertGreater(info["ess"], 0.0)
        self.assertLessEqual(info["ess"], 1.0)
        self.assertEqual(N_IMPRESSIONS, info["N"])

    def test_clipping_bounds_the_weights(self) -> None:
        _, raw, info = compute_pairwise_density_weights(
            self.log_policy,
            self.oracle,
            self.sampled_users,
            self.log_top1_ads,
            self.sampled_users,
            self.log_top1_ads,
            clip=2.0,
        )
        self.assertLessEqual(info["w_max"], 2.0)
        self.assertGreaterEqual(raw.max().item(), info["w_max"])

    def test_tighter_clip_raises_ess(self) -> None:
        """Clipping trades bias for variance, and ESS is the variance readout."""

        def ess(clip: float) -> float:
            _, _, info = compute_pairwise_density_weights(
                self.log_policy,
                self.oracle,
                self.sampled_users,
                self.log_top1_ads,
                self.sampled_users,
                self.log_top1_ads,
                clip=clip,
            )
            return info["ess"]

        self.assertGreaterEqual(ess(1.0), ess(100.0))


class CalibrationSweepTest(CalibrationTestBase):
    def test_covers_the_whole_grid(self) -> None:
        sweep = self._sweep()
        self.assertEqual(16 * 21, len(sweep))

    def test_every_point_reports_value_and_overlap(self) -> None:
        for point in self._sweep():
            self.assertGreater(point["V_top1"], 0.0)
            self.assertGreater(point["ess"], 0.0)
            self.assertLessEqual(point["ess"], 1.0)
            self.assertAlmostEqual(
                point["vmult"], point["V_top1"] / self.v_log, places=12
            )

    def test_spans_a_usable_range_of_value_and_overlap(self) -> None:
        """The grid has to bracket the experiment's targets to be useful."""
        sweep = self._sweep()
        vmults = [p["vmult"] for p in sweep]
        esss = [p["ess"] for p in sweep]
        # The grid must bracket the logging policy on value and sweep overlap
        # from near-disjoint to near-identical, or targets are unreachable.
        self.assertLess(min(vmults), 1.0)
        self.assertGreater(max(vmults), 1.0)
        self.assertGreater(max(vmults) - min(vmults), 0.2)
        self.assertLess(min(esss), 0.3)
        self.assertGreater(max(esss), 0.85)

    def test_is_deterministic(self) -> None:
        """Recomputed rather than cached, or this asserts nothing."""
        self.assertEqual(self._run_sweep(), self._run_sweep())


class LookupAlphaBetaTest(CalibrationTestBase):
    def test_returns_a_point_at_the_requested_value(self) -> None:
        sweep = self._sweep()
        for vmult in (0.95, 1.0, 1.05, 1.1):
            with self.subTest(vmult=vmult):
                match = lookup_alpha_beta(sweep, vmult, 0.65)
                self.assertIsNotNone(match)
                assert match is not None
                self.assertLessEqual(abs(match["vmult"] - vmult), 0.06)

    def test_picks_the_closest_overlap_at_that_value(self) -> None:
        sweep = self._sweep()
        target_ess = 0.45
        match = lookup_alpha_beta(sweep, 1.05, target_ess)
        assert match is not None
        eligible = [p for p in sweep if abs(p["vmult"] - 1.05) <= 0.03]
        best = min(abs(p["ess"] - target_ess) for p in eligible)
        self.assertAlmostEqual(best, abs(match["ess"] - target_ess), places=12)

    def test_higher_ess_target_selects_higher_ess_point(self) -> None:
        sweep = self._sweep()
        low = lookup_alpha_beta(sweep, 1.05, 0.25)
        high = lookup_alpha_beta(sweep, 1.05, 0.85)
        assert low is not None and high is not None
        self.assertLess(low["ess"], high["ess"])

    def test_returns_none_when_value_is_off_the_grid(self) -> None:
        self.assertIsNone(lookup_alpha_beta(self._sweep(), 99.0, 0.5))

    def test_widens_tolerance_before_giving_up(self) -> None:
        """A value reachable only within 2x tol still resolves."""
        sweep = self._sweep()
        target = max(p["vmult"] for p in sweep) - 0.04
        self.assertIsNotNone(lookup_alpha_beta(sweep, target, 0.5, tol=0.005))
