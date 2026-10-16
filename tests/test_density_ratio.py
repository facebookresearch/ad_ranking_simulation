# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Behavioral tests for density-ratio estimation and weight diagnostics."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from adsim.density_ratio import (
    compute_calibrated_probs,
    compute_importance_weights,
    compute_importance_weights_from_classifier,
    compute_per_domain_density_ratios,
    compute_weight_diagnostics,
)
from adsim.domain_classifier import (
    MLPDomainClassifier,
)

FEATURE_DIM = 6
N_CLASSES = 7
N_CANDIDATES = N_CLASSES - 1
N_SAMPLES = 400


def _uniform_probs() -> torch.Tensor:
    """What an uninformative classifier outputs."""
    return torch.full((N_SAMPLES, N_CLASSES), 1.0 / N_CLASSES)


def _probs(seed: int = 3, skew: float = 1.0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.softmax(
        torch.randn(N_SAMPLES, N_CLASSES, generator=gen) * skew, dim=-1
    )


class ComputeCalibratedProbsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.classifier = MLPDomainClassifier(FEATURE_DIM, N_CLASSES)
        self.embeddings = torch.randn(N_SAMPLES, FEATURE_DIM)

    def test_rows_are_distributions(self) -> None:
        probs = compute_calibrated_probs(self.classifier, self.embeddings, 1.5)
        self.assertTrue(
            torch.allclose(probs.sum(dim=1), torch.ones(N_SAMPLES), atol=1e-5)
        )

    def test_higher_temperature_softens_predictions(self) -> None:
        sharp = compute_calibrated_probs(self.classifier, self.embeddings, 0.5)
        soft = compute_calibrated_probs(self.classifier, self.embeddings, 5.0)
        self.assertGreater(
            sharp.max(dim=1).values.mean(), soft.max(dim=1).values.mean()
        )

    def test_temperature_one_is_plain_softmax(self) -> None:
        probs = compute_calibrated_probs(self.classifier, self.embeddings, 1.0)
        self.assertTrue(
            torch.allclose(probs, self.classifier.predict_proba(self.embeddings))
        )

    def test_does_not_require_grad(self) -> None:
        probs = compute_calibrated_probs(self.classifier, self.embeddings, 1.0)
        self.assertFalse(probs.requires_grad)


class ComputeImportanceWeightsTest(unittest.TestCase):
    def test_uninformative_classifier_gives_unit_weights(self) -> None:
        """p0 = 1/G makes (1-p0)/(K*p0) exactly 1 when K = G-1.

        This is the degenerate case where the classifier has learned nothing:
        the weighted model becomes the unweighted one.
        """
        weights = compute_importance_weights(
            _uniform_probs(), num_target_policies=N_CANDIDATES, clip=100.0
        )
        np.testing.assert_allclose(weights, np.ones(N_SAMPLES), rtol=1e-5)

    def test_equals_mean_of_per_candidate_ratios(self) -> None:
        """The mixture weight is by construction the average per-policy ratio."""
        probs = _probs()
        mixture = compute_importance_weights(
            probs, num_target_policies=N_CANDIDATES, clip=1e9
        )
        per_domain = compute_per_domain_density_ratios(probs, clip=1e9)
        np.testing.assert_allclose(mixture, per_domain.mean(dim=1).numpy(), rtol=1e-5)

    def test_clipping_bounds_weights(self) -> None:
        weights = compute_importance_weights(
            _probs(skew=4.0), num_target_policies=N_CANDIDATES, clip=3.0
        )
        self.assertLessEqual(weights.max(), 3.0)
        self.assertGreaterEqual(weights.min(), 0.0)

    def test_impressions_unlikely_under_logging_get_larger_weights(self) -> None:
        probs = _probs()
        weights = compute_importance_weights(
            probs, num_target_policies=N_CANDIDATES, clip=1e9
        )
        order = np.argsort(probs[:, 0].numpy())
        self.assertGreater(weights[order[0]], weights[order[-1]])

    def test_zero_logging_probability_does_not_divide_by_zero(self) -> None:
        probs = _probs()
        probs[:5, 0] = 0.0
        weights = compute_importance_weights(
            probs, num_target_policies=N_CANDIDATES, clip=50.0
        )
        self.assertTrue(np.isfinite(weights).all())


class ComputePerDomainDensityRatiosTest(unittest.TestCase):
    def test_one_column_per_candidate(self) -> None:
        ratios = compute_per_domain_density_ratios(_probs())
        self.assertEqual((N_SAMPLES, N_CANDIDATES), tuple(ratios.shape))

    def test_columns_follow_requested_class_order(self) -> None:
        probs = _probs()
        forward = compute_per_domain_density_ratios(probs, target_classes=[1, 2])
        reversed_ = compute_per_domain_density_ratios(probs, target_classes=[2, 1])
        self.assertTrue(torch.equal(forward[:, 0], reversed_[:, 1]))
        self.assertTrue(torch.equal(forward[:, 1], reversed_[:, 0]))

    def test_ratio_is_probability_quotient(self) -> None:
        probs = _probs()
        ratios = compute_per_domain_density_ratios(probs, target_classes=[1], clip=1e9)
        expected = probs[:, 1] / probs[:, 0]
        self.assertTrue(torch.allclose(ratios[:, 0], expected, atol=1e-5))

    def test_clipping_bounds_ratios(self) -> None:
        ratios = compute_per_domain_density_ratios(_probs(skew=4.0), clip=2.0)
        self.assertLessEqual(ratios.max().item(), 2.0)

    def test_excludes_the_logging_class_by_default(self) -> None:
        ratios = compute_per_domain_density_ratios(_probs(), logging_class=2)
        self.assertEqual(N_CANDIDATES, ratios.shape[1])


class ComputeWeightDiagnosticsTest(unittest.TestCase):
    def test_uniform_weights_give_full_ess(self) -> None:
        diagnostics = compute_weight_diagnostics(np.ones(100))
        self.assertAlmostEqual(100.0, diagnostics["ess"], places=6)
        self.assertAlmostEqual(1.0, diagnostics["ess_ratio"], places=6)

    def test_concentrated_weights_collapse_ess(self) -> None:
        weights = np.zeros(100)
        weights[0] = 1.0
        diagnostics = compute_weight_diagnostics(weights)
        self.assertAlmostEqual(1.0, diagnostics["ess"], places=6)
        self.assertAlmostEqual(0.01, diagnostics["ess_ratio"], places=6)

    def test_all_zero_weights_report_zero_ess(self) -> None:
        diagnostics = compute_weight_diagnostics(np.zeros(50))
        self.assertEqual(0.0, diagnostics["ess"])
        self.assertEqual(0.0, diagnostics["ess_ratio"])

    def test_reports_distribution_summary(self) -> None:
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        diagnostics = compute_weight_diagnostics(weights)
        self.assertEqual(4.0, diagnostics["max_w"])
        self.assertAlmostEqual(2.5, diagnostics["mean_w"])
        self.assertAlmostEqual(7.5, diagnostics["e_w2"])
        self.assertEqual(4.0, diagnostics["n_total"])

    def test_ess_ratio_falls_as_weights_spread(self) -> None:
        """ESS ratio is the overlap readout used throughout the paper."""
        tight = compute_weight_diagnostics(
            compute_importance_weights(
                _probs(skew=0.2), num_target_policies=N_CANDIDATES, clip=100.0
            )
        )
        spread = compute_weight_diagnostics(
            compute_importance_weights(
                _probs(skew=4.0), num_target_policies=N_CANDIDATES, clip=100.0
            )
        )
        self.assertGreater(tight["ess_ratio"], spread["ess_ratio"])


class ComputeImportanceWeightsFromClassifierTest(unittest.TestCase):
    def test_matches_the_two_step_path(self) -> None:
        torch.manual_seed(11)
        classifier = MLPDomainClassifier(FEATURE_DIM, N_CLASSES)
        embeddings = torch.randn(N_SAMPLES, FEATURE_DIM)
        combined = compute_importance_weights_from_classifier(
            classifier, embeddings, 1.7, num_target_policies=N_CANDIDATES, clip=20.0
        )
        stepwise = compute_importance_weights(
            compute_calibrated_probs(classifier, embeddings, 1.7),
            num_target_policies=N_CANDIDATES,
            clip=20.0,
        )
        np.testing.assert_array_equal(stepwise, combined)
