# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Behavioral tests for the Direct Method reward model."""

from __future__ import annotations

import unittest

import torch
from adsim.reward_model import (
    calibrate_temperature,
    EPS_REWARD,
    LOGIT_BOUND,
    MLPDMRewardModel,
    predict_calibrated,
    train_reward_model,
)

D_IN = 6
N_SAMPLES = 1_000
HIDDEN = [64]
CPU = torch.device("cpu")


def _dataset(seed: int = 5, n: int = N_SAMPLES) -> tuple[torch.Tensor, torch.Tensor]:
    """Features whose first coordinate drives the reward, so a fit is learnable."""
    gen = torch.Generator().manual_seed(seed)
    features = torch.randn(n, D_IN, generator=gen)
    labels = torch.sigmoid(2.0 * features[:, 0])
    return features, labels


class MLPDMRewardModelTest(unittest.TestCase):
    def test_forward_returns_one_logit_per_row(self) -> None:
        model = MLPDMRewardModel(D_IN, hidden_dims=HIDDEN)
        self.assertEqual((17,), tuple(model(torch.randn(17, D_IN)).shape))

    def test_predict_proba_is_in_unit_interval(self) -> None:
        model = MLPDMRewardModel(D_IN, hidden_dims=HIDDEN)
        probs = model.predict_proba(torch.randn(17, D_IN))
        self.assertGreaterEqual(probs.min().item(), 0.0)
        self.assertLessEqual(probs.max().item(), 1.0)

    def test_penultimate_width_follows_hidden_dims(self) -> None:
        model = MLPDMRewardModel(D_IN, hidden_dims=[32, 8])
        self.assertEqual(8, model.embedding_dim)
        self.assertEqual(
            (5, 8), tuple(model.get_penultimate_features(torch.randn(5, D_IN)).shape)
        )

    def test_linear_model_exposes_its_input_as_embedding(self) -> None:
        model = MLPDMRewardModel(D_IN, hidden_dims=[])
        features = torch.randn(5, D_IN)
        self.assertEqual(D_IN, model.embedding_dim)
        self.assertTrue(torch.equal(features, model.get_penultimate_features(features)))

    def test_defaults_to_two_hidden_layers(self) -> None:
        self.assertEqual([256, 128], MLPDMRewardModel(D_IN).hidden_dims)


class TrainRewardModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features, self.labels = _dataset()

    def _fit(
        self, weights: torch.Tensor | None = None, epochs: int = 12, seed: int = 3
    ) -> MLPDMRewardModel:
        torch.manual_seed(seed)
        model = MLPDMRewardModel(D_IN, hidden_dims=HIDDEN)
        torch.manual_seed(99)
        return train_reward_model(
            model,
            self.features,
            self.labels,
            weights=weights,
            epochs=epochs,
            device=CPU,
        )

    def test_fit_reduces_prediction_error(self) -> None:
        torch.manual_seed(3)
        untrained = MLPDMRewardModel(D_IN, hidden_dims=HIDDEN)
        with torch.no_grad():
            before = (untrained.predict_proba(self.features) - self.labels).abs().mean()
        trained = self._fit()
        with torch.no_grad():
            after = (trained.predict_proba(self.features) - self.labels).abs().mean()
        self.assertLess(after.item(), before.item())

    def test_returns_a_frozen_cpu_model(self) -> None:
        trained = self._fit(epochs=1)
        self.assertFalse(trained.training)
        for param in trained.parameters():
            self.assertFalse(param.requires_grad)
            self.assertEqual("cpu", param.device.type)

    def test_trains_in_place(self) -> None:
        torch.manual_seed(3)
        model = MLPDMRewardModel(D_IN, hidden_dims=HIDDEN)
        returned = train_reward_model(
            model, self.features, self.labels, epochs=1, device=CPU
        )
        self.assertIs(model, returned)

    def test_weights_are_scale_invariant(self) -> None:
        """Weights are renormalized to sum to N, so doubling them changes nothing."""
        weights = (
            torch.rand(N_SAMPLES, generator=torch.Generator().manual_seed(1)) + 0.2
        )
        single = self._fit(weights=weights, epochs=2)
        double = self._fit(weights=weights * 2.0, epochs=2)
        for key, value in single.state_dict().items():
            self.assertTrue(torch.equal(value, double.state_dict()[key]), key)

    def test_uniform_weights_match_the_unweighted_fit(self) -> None:
        uniform = self._fit(weights=torch.ones(N_SAMPLES), epochs=2)
        unweighted = self._fit(weights=None, epochs=2)
        for key, value in uniform.state_dict().items():
            self.assertTrue(
                torch.allclose(value, unweighted.state_dict()[key], atol=1e-6), key
            )

    def test_weighting_moves_the_fit(self) -> None:
        """Concentrating weight on half the data must change what is learned."""
        skewed = torch.ones(N_SAMPLES)
        skewed[: N_SAMPLES // 2] = 10.0
        weighted = self._fit(weights=skewed, epochs=2)
        unweighted = self._fit(weights=None, epochs=2)
        self.assertFalse(
            torch.equal(
                weighted.state_dict()["fc_out.bias"],
                unweighted.state_dict()["fc_out.bias"],
            )
        )


class ClampTest(unittest.TestCase):
    def test_logit_bound_corresponds_to_eps(self) -> None:
        """Exact in float64; evaluated in float32, so allow float32 resolution."""
        self.assertAlmostEqual(
            EPS_REWARD, torch.sigmoid(torch.tensor(-LOGIT_BOUND)).item(), delta=1e-7
        )

    def test_calibrated_predictions_are_bounded_away_from_zero_and_one(self) -> None:
        """The bounded loss the theory assumes requires bounded probabilities.

        Asserted against `sigmoid(+/- LOGIT_BOUND)` evaluated in float32 rather
        than against `eps` itself: the bound is computed in float64 and applied
        in float32, and float32 resolution near 1.0 is about 6e-8, far coarser
        than eps.
        """
        features, labels = _dataset()
        torch.manual_seed(3)
        model = MLPDMRewardModel(D_IN, hidden_dims=HIDDEN)
        trained = train_reward_model(model, features, labels, epochs=2, device=CPU)
        # An extreme temperature would saturate the sigmoid without the clamp.
        probs = predict_calibrated(trained, features, temperature=1e-6)
        lower = torch.sigmoid(torch.tensor(-LOGIT_BOUND)).item()
        upper = torch.sigmoid(torch.tensor(LOGIT_BOUND)).item()
        self.assertGreaterEqual(probs.min().item(), lower)
        self.assertLessEqual(probs.max().item(), upper)
        self.assertLess(upper, 1.0)
        self.assertGreater(lower, 0.0)


class CalibrateTemperatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features, self.labels = _dataset()
        self.cal_features, self.cal_labels = _dataset(seed=6, n=400)
        torch.manual_seed(3)
        model = MLPDMRewardModel(D_IN, hidden_dims=HIDDEN)
        self.model = train_reward_model(
            model, self.features, self.labels, epochs=8, device=CPU
        )

    def test_temperature_is_positive(self) -> None:
        self.assertGreater(
            calibrate_temperature(self.model, self.cal_features, self.cal_labels), 0.0
        )

    def test_is_deterministic(self) -> None:
        first = calibrate_temperature(self.model, self.cal_features, self.cal_labels)
        second = calibrate_temperature(self.model, self.cal_features, self.cal_labels)
        self.assertEqual(first, second)

    def test_does_not_reduce_calibration_quality(self) -> None:
        temperature = calibrate_temperature(
            self.model, self.cal_features, self.cal_labels
        )
        calibrated = predict_calibrated(self.model, self.cal_features, temperature)
        uncalibrated = predict_calibrated(self.model, self.cal_features, 1.0)
        error_after = (calibrated - self.cal_labels).abs().mean().item()
        error_before = (uncalibrated - self.cal_labels).abs().mean().item()
        self.assertLessEqual(error_after, error_before + 1e-6)

    def test_temperature_one_leaves_predictions_unchanged(self) -> None:
        with torch.no_grad():
            expected = torch.sigmoid(self.model(self.cal_features))
        self.assertTrue(
            torch.allclose(
                expected, predict_calibrated(self.model, self.cal_features, 1.0)
            )
        )
