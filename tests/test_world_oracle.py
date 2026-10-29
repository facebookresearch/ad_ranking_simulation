# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Behavioral tests for the world and the factored-kernel reward oracle."""

from __future__ import annotations

import unittest

import torch
from adsim.kernels import factored_kernel_eval
from adsim.oracle import make_oracle
from adsim.world import make_world

NUM_USERS = 2_000
NUM_ADS = 500
USER_DIM = 3
AD_DIM = 3
TARGET_CTR = 0.15


class WorldTest(unittest.TestCase):
    def test_shapes(self) -> None:
        users, ads = make_world(NUM_USERS, NUM_ADS, USER_DIM, AD_DIM)
        self.assertEqual((NUM_USERS, USER_DIM), tuple(users.shape))
        self.assertEqual((NUM_ADS, AD_DIM), tuple(ads.shape))

    def test_drawn_from_unit_hypercube(self) -> None:
        users, ads = make_world(NUM_USERS, NUM_ADS, USER_DIM, AD_DIM)
        for name, tensor in (("users", users), ("ads", ads)):
            with self.subTest(tensor=name):
                self.assertGreaterEqual(tensor.min().item(), -1.0)
                self.assertLess(tensor.max().item(), 1.0)

    def test_same_seed_reproduces_population(self) -> None:
        first = make_world(NUM_USERS, NUM_ADS, USER_DIM, AD_DIM, seed=42)
        second = make_world(NUM_USERS, NUM_ADS, USER_DIM, AD_DIM, seed=42)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_different_seed_changes_population(self) -> None:
        users_a, _ = make_world(NUM_USERS, NUM_ADS, USER_DIM, AD_DIM, seed=42)
        users_b, _ = make_world(NUM_USERS, NUM_ADS, USER_DIM, AD_DIM, seed=43)
        self.assertFalse(torch.equal(users_a, users_b))


class FactoredKernelEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.users, self.ads = make_world(NUM_USERS, NUM_ADS, USER_DIM, AD_DIM)
        self.oracle = make_oracle(TARGET_CTR, self.users, self.ads, n_kernels=64)

    def _score(self, users: torch.Tensor, ads: torch.Tensor) -> torch.Tensor:
        return factored_kernel_eval(
            users,
            ads,
            self.oracle["mu_u"],
            self.oracle["mu_a"],
            self.oracle["sigma_u"],
            self.oracle["sigma_a"],
            self.oracle["alphas"],
        )

    def test_scores_are_row_aligned_and_positive(self) -> None:
        scores = self._score(self.users[:32], self.ads[:32])
        self.assertEqual((32,), tuple(scores.shape))
        self.assertGreater(scores.min().item(), 0.0)

    def test_score_peaks_at_kernel_center(self) -> None:
        """A pair sitting on a kernel center scores above a distant pair."""
        center_u = self.oracle["mu_u"][0:1]
        center_a = self.oracle["mu_a"][0:1]
        at_center = self._score(center_u, center_a)
        far = self._score(-center_u - 3.0, -center_a - 3.0)
        self.assertGreater(at_center.item(), far.item())


class MakeOracleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.users, self.ads = make_world(NUM_USERS, NUM_ADS, USER_DIM, AD_DIM)

    def test_mean_reward_is_rescaled_to_target(self) -> None:
        oracle = make_oracle(TARGET_CTR, self.users, self.ads, n_kernels=400)
        self.assertAlmostEqual(TARGET_CTR, oracle["V_cf"], delta=0.01)

    def test_target_ctr_is_honored_across_levels(self) -> None:
        for target in (0.05, 0.15, 0.30):
            with self.subTest(target=target):
                oracle = make_oracle(target, self.users, self.ads, n_kernels=400)
                self.assertAlmostEqual(target, oracle["V_cf"], delta=target * 0.1)

    def test_same_seed_reproduces_oracle(self) -> None:
        first = make_oracle(TARGET_CTR, self.users, self.ads, seed=42)
        second = make_oracle(TARGET_CTR, self.users, self.ads, seed=42)
        self.assertTrue(torch.equal(first["alphas"], second["alphas"]))
        self.assertEqual(first["V_cf"], second["V_cf"])

    def test_reports_kernel_geometry_for_policy_positioning(self) -> None:
        """`adsim.policy` reads the center and delta back off the oracle."""
        center = (torch.full((USER_DIM,), 0.25), torch.full((AD_DIM,), -0.1))
        oracle = make_oracle(TARGET_CTR, self.users, self.ads, center=center, delta=0.4)
        self.assertTrue(torch.equal(center[0], oracle["center_u"]))
        self.assertTrue(torch.equal(center[1], oracle["center_a"]))
        self.assertEqual(0.4, oracle["delta"])

    def test_kernels_stay_within_delta_of_center(self) -> None:
        oracle = make_oracle(TARGET_CTR, self.users, self.ads, delta=0.5, n_kernels=400)
        self.assertLessEqual(oracle["mu_u"].abs().max().item(), 0.5)
        self.assertLessEqual(oracle["mu_a"].abs().max().item(), 0.5)
