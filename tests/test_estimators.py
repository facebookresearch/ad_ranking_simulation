# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Behavioral tests for the off-policy value estimators."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from adsim.estimators import (
    direct_method_value,
    dr_values,
    ips_values,
)
from adsim.reward_model import (
    MLPDMRewardModel,
    predict_calibrated,
)

N_EVAL = 600
FEATURE_DIM = 6


def _rng(seed: int = 17) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class DirectMethodTest(unittest.TestCase):
    def test_is_the_mean_calibrated_prediction(self) -> None:
        torch.manual_seed(3)
        model = MLPDMRewardModel(FEATURE_DIM, hidden_dims=[16])
        features = torch.randn(N_EVAL, FEATURE_DIM, generator=_rng())
        expected = predict_calibrated(model, features, 1.3).mean().item()
        self.assertEqual(expected, direct_method_value(model, features, 1.3))


class IpsValuesTest(unittest.TestCase):
    def setUp(self) -> None:
        gen = _rng()
        self.rewards = torch.rand(N_EVAL, generator=gen)
        self.weights = torch.rand(N_EVAL, generator=gen) * 3.0

    def test_unit_weights_reduce_to_the_logged_mean(self) -> None:
        plain, self_normalized = ips_values(self.rewards, torch.ones(N_EVAL))
        expected = self.rewards.mean().item()
        self.assertAlmostEqual(expected, plain, places=6)
        self.assertAlmostEqual(expected, self_normalized, places=6)

    def test_self_normalization_divides_by_the_weight_sum(self) -> None:
        plain, self_normalized = ips_values(self.rewards, self.weights)
        ratio = self.weights.sum().item() / N_EVAL
        self.assertAlmostEqual(plain / ratio, self_normalized, places=6)

    def test_negative_weights_are_floored(self) -> None:
        """A negative density ratio is meaningless and must not subtract reward."""
        negative = torch.full((N_EVAL,), -1.0)
        plain, self_normalized = ips_values(self.rewards, negative)
        self.assertEqual(0.0, plain)
        self.assertEqual(0.0, self_normalized)

    def test_zero_weights_do_not_divide_by_zero(self) -> None:
        plain, self_normalized = ips_values(self.rewards, torch.zeros(N_EVAL))
        self.assertEqual(0.0, plain)
        self.assertTrue(np.isfinite(self_normalized))

    def test_concentrating_weight_selects_those_rewards(self) -> None:
        weights = torch.zeros(N_EVAL)
        weights[:10] = 1.0
        _, self_normalized = ips_values(self.rewards, weights)
        self.assertAlmostEqual(
            self.rewards[:10].mean().item(), self_normalized, places=5
        )


class DrValuesTest(unittest.TestCase):
    def setUp(self) -> None:
        gen = _rng()
        self.rewards = torch.rand(N_EVAL, generator=gen)
        self.weights = torch.rand(N_EVAL, generator=gen) * 2.0
        self.r_hat_logged = torch.rand(N_EVAL, generator=gen)
        self.r_hat_target = torch.rand(N_EVAL, generator=gen)

    def test_zero_weights_reduce_to_the_direct_method(self) -> None:
        plain, self_normalized = dr_values(
            self.rewards, torch.zeros(N_EVAL), self.r_hat_logged, self.r_hat_target
        )
        expected = self.r_hat_target.mean().item()
        self.assertAlmostEqual(expected, plain, places=9)
        self.assertAlmostEqual(expected, self_normalized, places=9)

    def test_a_perfect_reward_model_leaves_no_correction(self) -> None:
        """If the model predicts the logged rewards exactly, DR is the DM value."""
        plain, _ = dr_values(
            self.rewards, self.weights, self.rewards, self.r_hat_target
        )
        self.assertAlmostEqual(self.r_hat_target.mean().item(), plain, places=9)

    def test_correction_moves_the_estimate_toward_observed_reward(self) -> None:
        """A model that under-predicts logged reward must be corrected upward."""
        under = self.rewards - 0.2
        plain, _ = dr_values(self.rewards, self.weights, under, self.r_hat_target)
        self.assertGreater(plain, self.r_hat_target.mean().item())

    def test_unit_weights_make_plain_and_self_normalized_agree(self) -> None:
        plain, self_normalized = dr_values(
            self.rewards, torch.ones(N_EVAL), self.r_hat_logged, self.r_hat_target
        )
        self.assertAlmostEqual(plain, self_normalized, places=9)
