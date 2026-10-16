# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Behavioral tests for candidate policies and their calibration."""

from __future__ import annotations

import unittest

import torch
from adsim.kernels import (
    score_pairs,
    top1_from_slate,
)
from adsim.oracle import make_oracle
from adsim.policy import (
    calibrate_policy_top1,
    evaluate_policy_value,
    make_policy,
    Policy,
)
from adsim.world import make_world

NUM_USERS = 2_000
NUM_ADS = 500
USER_DIM = 3
AD_DIM = 3
N_IMPRESSIONS = 1_000
TOP_K = 10

N_KERNELS = 200
DELTA = 0.3
SIGMA_RANGE = (0.3, 0.5)
ALPHA_RANGE = (0.1, 1.0)


class PolicyTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.users, self.ads = make_world(NUM_USERS, NUM_ADS, USER_DIM, AD_DIM)
        self.oracle = make_oracle(0.15, self.users, self.ads, n_kernels=400)
        gen = torch.Generator().manual_seed(5000)
        self.sampled_users = self.users[
            torch.randint(0, NUM_USERS, (N_IMPRESSIONS,), generator=gen)
        ]
        self.candidate_ads = self.ads[
            torch.randint(0, NUM_ADS, (N_IMPRESSIONS, TOP_K), generator=gen)
        ]

    def _policy(self, offset: float = 0.0, seed: int = 342) -> Policy:
        center = torch.full((USER_DIM + AD_DIM,), offset)
        return make_policy(
            N_KERNELS,
            center[:USER_DIM],
            center[USER_DIM:],
            DELTA,
            SIGMA_RANGE,
            ALPHA_RANGE,
            seed=seed,
        )


class MakePolicyTest(PolicyTestBase):
    def test_kernel_shapes(self) -> None:
        policy = self._policy()
        self.assertEqual((N_KERNELS, USER_DIM), tuple(policy["mu_u"].shape))
        self.assertEqual((N_KERNELS, AD_DIM), tuple(policy["mu_a"].shape))
        self.assertEqual((N_KERNELS,), tuple(policy["alphas"].shape))
        self.assertEqual(N_KERNELS, policy["K"])

    def test_kernels_are_centered_on_request(self) -> None:
        policy = self._policy(offset=0.5)
        self.assertLessEqual((policy["mu_u"] - 0.5).abs().max().item(), DELTA)
        self.assertLessEqual((policy["mu_a"] - 0.5).abs().max().item(), DELTA)

    def test_is_unscaled_at_construction(self) -> None:
        """Amplitudes come straight from `alpha_range`; scaling is a later step."""
        policy = self._policy()
        self.assertGreaterEqual(policy["alphas"].min().item(), ALPHA_RANGE[0])
        self.assertLessEqual(policy["alphas"].max().item(), ALPHA_RANGE[1])
        self.assertNotIn("V_top1", policy)

    def test_same_seed_reproduces_policy(self) -> None:
        self.assertTrue(
            torch.equal(self._policy(seed=7)["alphas"], self._policy(seed=7)["alphas"])
        )


class CalibratePolicyTop1Test(PolicyTestBase):
    def test_matches_oracle_score_mass_on_selected_ads(self) -> None:
        policy = calibrate_policy_top1(
            self._policy(), self.oracle, self.sampled_users, self.candidate_ads, TOP_K
        )
        top1 = top1_from_slate(policy, self.sampled_users, self.candidate_ads, TOP_K)
        policy_mass = score_pairs(policy, self.sampled_users, top1).sum().item()
        oracle_mass = score_pairs(self.oracle, self.sampled_users, top1).sum().item()
        self.assertAlmostEqual(oracle_mass, policy_mass, delta=oracle_mass * 1e-4)

    def test_records_true_top1_value(self) -> None:
        policy = calibrate_policy_top1(
            self._policy(), self.oracle, self.sampled_users, self.candidate_ads, TOP_K
        )
        self.assertIn("V_top1", policy)
        expected = evaluate_policy_value(
            policy, self.oracle, self.sampled_users, self.candidate_ads, TOP_K
        )
        self.assertAlmostEqual(expected, policy["V_top1"], places=12)

    def test_rescaling_preserves_the_ranking(self) -> None:
        """Scaling every amplitude by one constant cannot reorder a slate."""
        policy = self._policy()
        before = top1_from_slate(policy, self.sampled_users, self.candidate_ads, TOP_K)
        calibrate_policy_top1(
            policy, self.oracle, self.sampled_users, self.candidate_ads, TOP_K
        )
        after = top1_from_slate(policy, self.sampled_users, self.candidate_ads, TOP_K)
        self.assertTrue(torch.equal(before, after))

    def test_mutates_in_place(self) -> None:
        """Call sites rely on the argument being updated, not just returned."""
        policy = self._policy()
        original = policy["alphas"].clone()
        returned = calibrate_policy_top1(
            policy, self.oracle, self.sampled_users, self.candidate_ads, TOP_K
        )
        self.assertIs(policy, returned)
        self.assertFalse(torch.equal(original, policy["alphas"]))


class EvaluatePolicyValueTest(PolicyTestBase):
    def test_value_is_bounded_by_the_best_available_ad(self) -> None:
        """No policy can beat picking the oracle-best ad in each slate."""
        policy = self._policy()
        value = evaluate_policy_value(
            policy, self.oracle, self.sampled_users, self.candidate_ads, TOP_K
        )
        best_ads = top1_from_slate(
            self.oracle, self.sampled_users, self.candidate_ads, TOP_K
        )
        best_value = (
            score_pairs(self.oracle, self.sampled_users, best_ads).mean().item()
        )
        self.assertLessEqual(value, best_value)
        self.assertGreater(value, 0.0)

    def test_policy_aligned_with_oracle_beats_a_displaced_one(self) -> None:
        """Moving a policy away from the oracle center costs it value."""
        aligned = evaluate_policy_value(
            self._policy(offset=0.0),
            self.oracle,
            self.sampled_users,
            self.candidate_ads,
            TOP_K,
        )
        displaced = evaluate_policy_value(
            self._policy(offset=0.9),
            self.oracle,
            self.sampled_users,
            self.candidate_ads,
            TOP_K,
        )
        self.assertGreater(aligned, displaced)

    def test_is_deterministic(self) -> None:
        policy = self._policy()
        args = (self.oracle, self.sampled_users, self.candidate_ads, TOP_K)
        self.assertEqual(
            evaluate_policy_value(policy, *args),
            evaluate_policy_value(policy, *args),
        )
