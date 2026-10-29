# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Behavioral tests for logged-data generation and candidate geometry."""

from __future__ import annotations

import unittest

import torch
from adsim.calibration import (
    orthogonal_directions,
    shift_axis,
    sweep_orthogonal_direction,
)
from adsim.kernels import (
    score_pairs,
    top1_from_slate,
)
from adsim.logging_sim import (
    clamp_reward,
    collect_logged_data,
    crossfit_split_indices,
    make_logging_policy,
    sample_impressions,
    split_indices,
)
from adsim.oracle import make_oracle
from adsim.world import make_world

NUM_USERS = 2_000
NUM_ADS = 500
USER_DIM = 3
AD_DIM = 3
N_IMPRESSIONS = 1_000
TOP_K = 10

N_KERNELS = 64
DELTA = 0.3
SIGMA_RANGE = (0.3, 0.5)
ALPHA_RANGE = (0.1, 1.0)
N_POLICIES = 6


class LoggingSimTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.users, self.ads = make_world(NUM_USERS, NUM_ADS, USER_DIM, AD_DIM)
        self.oracle = make_oracle(0.15, self.users, self.ads, n_kernels=128)
        self.sampled_users, self.candidate_ads = sample_impressions(
            self.users, self.ads, N_IMPRESSIONS, TOP_K, seed=5042
        )
        self.log_policy, self.log_center = make_logging_policy(
            N_KERNELS,
            USER_DIM,
            AD_DIM,
            DELTA,
            SIGMA_RANGE,
            ALPHA_RANGE,
            center_seed=142,
            policy_seed=242,
        )


class SampleImpressionsTest(LoggingSimTestBase):
    def test_shapes(self) -> None:
        self.assertEqual((N_IMPRESSIONS, USER_DIM), tuple(self.sampled_users.shape))
        self.assertEqual(
            (N_IMPRESSIONS, TOP_K, AD_DIM), tuple(self.candidate_ads.shape)
        )

    def test_draws_from_the_populations(self) -> None:
        """Every sampled row must be an actual member of the population."""
        self.assertTrue(
            (self.sampled_users.unsqueeze(1) == self.users).all(-1).any(-1).all()
        )

    def test_same_seed_reproduces_impressions(self) -> None:
        again, _ = sample_impressions(
            self.users, self.ads, N_IMPRESSIONS, TOP_K, seed=5042
        )
        self.assertTrue(torch.equal(self.sampled_users, again))

    def test_different_seed_changes_impressions(self) -> None:
        other, _ = sample_impressions(
            self.users, self.ads, N_IMPRESSIONS, TOP_K, seed=6042
        )
        self.assertFalse(torch.equal(self.sampled_users, other))


class MakeLoggingPolicyTest(LoggingSimTestBase):
    def test_center_is_within_the_draw_box(self) -> None:
        self.assertLessEqual(self.log_center.abs().max().item(), 0.5)
        self.assertEqual((USER_DIM + AD_DIM,), tuple(self.log_center.shape))

    def test_center_is_offset_from_the_oracle(self) -> None:
        """A logging policy identical to the oracle would make the task trivial."""
        oracle_center = torch.cat([self.oracle["center_u"], self.oracle["center_a"]])
        self.assertGreater((self.log_center - oracle_center).norm().item(), 0.0)

    def test_center_and_kernels_use_independent_seeds(self) -> None:
        _, center_a = make_logging_policy(
            N_KERNELS,
            USER_DIM,
            AD_DIM,
            DELTA,
            SIGMA_RANGE,
            ALPHA_RANGE,
            center_seed=142,
            policy_seed=999,
        )
        self.assertTrue(torch.equal(self.log_center, center_a))


class CollectLoggedDataTest(LoggingSimTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.logged = collect_logged_data(
            self.log_policy, self.oracle, self.sampled_users, self.candidate_ads, TOP_K
        )

    def test_serves_the_policy_top1(self) -> None:
        expected = top1_from_slate(
            self.log_policy, self.sampled_users, self.candidate_ads, TOP_K
        )
        self.assertTrue(torch.equal(expected, self.logged["served_ads"]))

    def test_features_are_user_concat_served_ad(self) -> None:
        features = self.logged["features"]
        self.assertEqual((N_IMPRESSIONS, USER_DIM + AD_DIM), tuple(features.shape))
        self.assertTrue(torch.equal(self.sampled_users, features[:, :USER_DIM]))
        self.assertTrue(torch.equal(self.logged["served_ads"], features[:, USER_DIM:]))

    def test_rewards_are_valid_bernoulli_targets(self) -> None:
        rewards = self.logged["rewards"]
        self.assertGreaterEqual(rewards.min().item(), 0.0)
        self.assertLessEqual(rewards.max().item(), 1.0)

    def test_v_log_is_the_unclamped_mean(self) -> None:
        """Value is the true oracle mean; clamping applies only to the targets."""
        self.assertGreater(self.logged["v_log"], 0.0)
        self.assertGreaterEqual(self.logged["rewards"].mean().item(), 0.0)

    def test_logging_beats_serving_a_random_slate_entry(self) -> None:
        random_pick = self.candidate_ads[:, 0, :]
        with torch.no_grad():
            random_value = (
                score_pairs(self.oracle, self.sampled_users, random_pick).mean().item()
            )
        self.assertGreater(self.logged["v_log"], random_value)


class ClampRewardTest(unittest.TestCase):
    def test_bounds_to_unit_interval(self) -> None:
        values = torch.tensor([-0.5, 0.0, 0.25, 1.0, 3.0])
        clamped = clamp_reward(values)
        self.assertTrue(torch.equal(torch.tensor([0.0, 0.0, 0.25, 1.0, 1.0]), clamped))


class SplitIndicesTest(unittest.TestCase):
    def test_default_split_is_40_20_40(self) -> None:
        split = split_indices(1_000, seed=6042)
        self.assertEqual(400, len(split["train"]))
        self.assertEqual(200, len(split["calibration"]))
        self.assertEqual(400, len(split["evaluation"]))

    def test_folds_are_disjoint_and_cover_everything(self) -> None:
        split = split_indices(1_000, seed=6042)
        combined = torch.cat(
            [split["train"], split["calibration"], split["evaluation"]]
        )
        self.assertEqual(1_000, len(combined))
        self.assertEqual(1_000, len(set(combined.tolist())))

    def test_custom_fractions(self) -> None:
        split = split_indices(
            1_000, seed=6042, train_fraction=0.5, calibration_fraction=0.1
        )
        self.assertEqual(500, len(split["train"]))
        self.assertEqual(100, len(split["calibration"]))
        self.assertEqual(400, len(split["evaluation"]))

    def test_dc_fold_matches_train_when_not_crossfitting(self) -> None:
        """Without cross-fitting, the classifier is fitted on the reward rows."""
        split = split_indices(1_000, seed=6042)
        self.assertTrue(torch.equal(split["train"], split["dc"]))

    def test_same_seed_reproduces_split(self) -> None:
        self.assertTrue(
            torch.equal(
                split_indices(1_000, seed=6042)["train"],
                split_indices(1_000, seed=6042)["train"],
            )
        )


class GeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.oracle_center = torch.zeros(USER_DIM + AD_DIM)
        self.log_center = torch.full((USER_DIM + AD_DIM,), 0.2)

    def test_shift_axis_is_a_unit_vector_and_the_gap(self) -> None:
        v_along, distance = shift_axis(self.oracle_center, self.log_center)
        self.assertAlmostEqual(1.0, v_along.norm().item(), places=5)
        self.assertAlmostEqual(
            (self.log_center - self.oracle_center).norm().item(), distance, places=6
        )

    def test_shift_axis_is_degenerate_when_centers_coincide(self) -> None:
        v_along, distance = shift_axis(self.oracle_center, self.oracle_center)
        self.assertEqual(0.0, distance)
        self.assertEqual(0.0, v_along.norm().item())

    def test_orthogonal_directions_are_unit_and_orthogonal(self) -> None:
        v_along, _ = shift_axis(self.oracle_center, self.log_center)
        for direction in orthogonal_directions(v_along, N_POLICIES, seed=42):
            self.assertAlmostEqual(1.0, direction.norm().item(), places=5)
            self.assertAlmostEqual(0.0, (direction @ v_along).item(), places=5)

    def test_orthogonal_directions_differ_per_candidate(self) -> None:
        """Collinear candidates would only be more and less extreme, not distinct."""
        v_along, _ = shift_axis(self.oracle_center, self.log_center)
        directions = orthogonal_directions(v_along, N_POLICIES, seed=42)
        self.assertEqual(N_POLICIES, len(directions))
        self.assertFalse(torch.equal(directions[0], directions[1]))

    def test_sweep_direction_is_unit_and_orthogonal(self) -> None:
        v_along, _ = shift_axis(self.oracle_center, self.log_center)
        w = sweep_orthogonal_direction(v_along, seed=999)
        self.assertAlmostEqual(1.0, w.norm().item(), places=5)
        self.assertAlmostEqual(0.0, (w @ v_along).item(), places=5)


class CrossfitSplitIndicesTest(unittest.TestCase):
    def test_fold_sizes_are_absolute(self) -> None:
        split = crossfit_split_indices(
            7_000, seed=6042, n_dc=2_000, n_train=2_000, n_calibration=1_000
        )
        self.assertEqual(2_000, len(split["dc"]))
        self.assertEqual(2_000, len(split["train"]))
        self.assertEqual(1_000, len(split["calibration"]))
        self.assertEqual(2_000, len(split["evaluation"]))

    def test_classifier_and_reward_folds_are_disjoint(self) -> None:
        """This is the entire point of cross-fitting."""
        split = crossfit_split_indices(
            7_000, seed=6042, n_dc=2_000, n_train=2_000, n_calibration=1_000
        )
        self.assertEqual(
            set(), set(split["dc"].tolist()) & set(split["train"].tolist())
        )

    def test_every_fold_is_disjoint_and_covers_everything(self) -> None:
        split = crossfit_split_indices(
            7_000, seed=6042, n_dc=2_000, n_train=2_000, n_calibration=1_000
        )
        combined = torch.cat(
            [split["dc"], split["train"], split["calibration"], split["evaluation"]]
        )
        self.assertEqual(7_000, len(combined))
        self.assertEqual(7_000, len(set(combined.tolist())))

    def test_reward_fold_is_the_same_size_as_the_uncrossfitted_one(self) -> None:
        """The reason for absolute counts: neither model is starved."""
        standard = split_indices(50_000, seed=6042)
        crossfit = crossfit_split_indices(
            70_000, seed=6042, n_dc=20_000, n_train=20_000, n_calibration=10_000
        )
        self.assertEqual(len(standard["train"]), len(crossfit["train"]))
        self.assertEqual(len(standard["evaluation"]), len(crossfit["evaluation"]))

    def test_rejects_a_budget_with_no_room_to_evaluate(self) -> None:
        with self.assertRaises(ValueError):
            crossfit_split_indices(
                5_000, seed=6042, n_dc=2_000, n_train=2_000, n_calibration=1_000
            )

    def test_same_seed_reproduces_split(self) -> None:
        args = {"seed": 6042, "n_dc": 2_000, "n_train": 2_000, "n_calibration": 1_000}
        self.assertTrue(
            torch.equal(
                crossfit_split_indices(7_000, **args)["dc"],
                crossfit_split_indices(7_000, **args)["dc"],
            )
        )
