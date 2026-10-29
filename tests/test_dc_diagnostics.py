# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Behavioral tests for domain-classifier reliability diagnostics."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from adsim.dc_diagnostics import (
    candidate_confusion,
    logging_class_ece,
    mask_ad_features,
    multiclass_nll,
    spearman,
    true_cold_start_mass,
)
from adsim.kernels import top1_from_slate
from adsim.policy import (  # usort: skip
    make_policy,
    Policy,
)
from adsim.world import make_world

USER_DIM = 3
AD_DIM = 3
N_CLASSES = 4
TOP_K = 10


def _one_hot(labels: np.ndarray, confidence: float = 1.0) -> np.ndarray:
    """Probabilities placing `confidence` on the true class."""
    probs = np.full((len(labels), N_CLASSES), (1.0 - confidence) / (N_CLASSES - 1))
    probs[np.arange(len(labels)), labels] = confidence
    return probs


class MaskAdFeaturesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = torch.arange(24, dtype=torch.float32).reshape(4, 6)

    def test_zero_masks_nothing(self) -> None:
        self.assertTrue(torch.equal(self.features, mask_ad_features(self.features, 0)))

    def test_masks_trailing_columns_only(self) -> None:
        masked = mask_ad_features(self.features, 2)
        self.assertTrue(torch.equal(self.features[:, :4], masked[:, :4]))
        self.assertTrue(torch.all(masked[:, 4:] == 0.0))

    def test_masking_every_ad_column_leaves_the_user(self) -> None:
        """This is the policy-agnostic condition: no ad information survives."""
        masked = mask_ad_features(self.features, AD_DIM)
        self.assertTrue(torch.equal(self.features[:, :USER_DIM], masked[:, :USER_DIM]))
        self.assertTrue(torch.all(masked[:, USER_DIM:] == 0.0))

    def test_does_not_mutate_the_input(self) -> None:
        original = self.features.clone()
        mask_ad_features(self.features, 2)
        self.assertTrue(torch.equal(original, self.features))


class MulticlassNllTest(unittest.TestCase):
    def test_perfect_predictions_give_zero(self) -> None:
        labels = np.array([0, 1, 2, 3])
        self.assertAlmostEqual(0.0, multiclass_nll(_one_hot(labels), labels), places=9)

    def test_uniform_predictions_give_log_k(self) -> None:
        labels = np.array([0, 1, 2, 3])
        uniform = np.full((4, N_CLASSES), 1.0 / N_CLASSES)
        self.assertAlmostEqual(
            float(np.log(N_CLASSES)), multiclass_nll(uniform, labels), places=9
        )

    def test_confident_and_wrong_is_finite(self) -> None:
        """A zero on the true class must not produce infinity."""
        labels = np.array([0, 1])
        probs = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        self.assertTrue(np.isfinite(multiclass_nll(probs, labels)))

    def test_worse_predictions_score_higher(self) -> None:
        labels = np.array([0, 1, 2, 3])
        good = multiclass_nll(_one_hot(labels, 0.9), labels)
        bad = multiclass_nll(_one_hot(labels, 0.4), labels)
        self.assertLess(good, bad)


class LoggingClassEceTest(unittest.TestCase):
    def test_perfectly_calibrated_gives_zero(self) -> None:
        """Half the rows are logging and every row says 0.5."""
        labels = np.array([0] * 50 + [1] * 50)
        probs = np.zeros((100, N_CLASSES))
        probs[:, 0] = 0.5
        probs[:, 1] = 0.5
        self.assertAlmostEqual(0.0, logging_class_ece(probs, labels), places=6)

    def test_confidently_wrong_gives_one(self) -> None:
        labels = np.ones(100)
        probs = np.zeros((100, N_CLASSES))
        probs[:, 0] = 1.0
        self.assertAlmostEqual(1.0, logging_class_ece(probs, labels), places=6)

    def test_is_bounded(self) -> None:
        gen = np.random.RandomState(1)
        probs = np.abs(gen.randn(300, N_CLASSES))
        probs /= probs.sum(axis=1, keepdims=True)
        labels = gen.randint(0, N_CLASSES, size=300)
        error = logging_class_ece(probs, labels)
        self.assertGreaterEqual(error, 0.0)
        self.assertLessEqual(error, 1.0)


class CandidateConfusionTest(unittest.TestCase):
    def test_perfect_separation_gives_one(self) -> None:
        labels = np.array([1, 2, 3])
        accuracy, offdiag = candidate_confusion(_one_hot(labels), labels)
        self.assertAlmostEqual(1.0, accuracy)
        self.assertAlmostEqual(0.0, offdiag)

    def test_ignores_logging_rows(self) -> None:
        """Only candidate rows count, so adding logging rows changes nothing."""
        labels = np.array([1, 2, 3])
        with_logging = np.array([0, 1, 2, 3])
        self.assertEqual(
            candidate_confusion(_one_hot(labels), labels),
            candidate_confusion(_one_hot(with_logging), with_logging),
        )

    def test_no_candidate_rows_gives_nan(self) -> None:
        labels = np.zeros(10)
        accuracy, offdiag = candidate_confusion(_one_hot(labels.astype(int)), labels)
        self.assertTrue(np.isnan(accuracy))
        self.assertTrue(np.isnan(offdiag))

    def test_accuracy_and_offdiag_sum_to_one(self) -> None:
        gen = np.random.RandomState(2)
        probs = np.abs(gen.randn(200, N_CLASSES))
        probs /= probs.sum(axis=1, keepdims=True)
        labels = gen.randint(1, N_CLASSES, size=200)
        accuracy, offdiag = candidate_confusion(probs, labels)
        self.assertAlmostEqual(1.0, accuracy + offdiag)


class SpearmanTest(unittest.TestCase):
    def test_monotone_relationship_gives_one(self) -> None:
        values = np.arange(50, dtype=float)
        self.assertAlmostEqual(1.0, spearman(values, values**3), places=9)

    def test_reversed_relationship_gives_minus_one(self) -> None:
        values = np.arange(50, dtype=float)
        self.assertAlmostEqual(-1.0, spearman(values, -values), places=9)

    def test_constant_input_is_ranked_by_position_not_rejected(self) -> None:
        """Ties are broken by position, so a constant vector gets ranks 0..n-1
        and correlates perfectly with anything monotone. This is why the
        ablation's ratio correlation stays finite when the classifier collapses
        to uniform weights."""
        self.assertAlmostEqual(
            1.0, spearman(np.ones(20), np.arange(20, dtype=float)), places=9
        )

    def test_single_element_input_gives_nan(self) -> None:
        """The only case where the degenerate guard actually fires."""
        self.assertTrue(np.isnan(spearman(np.ones(1), np.ones(1))))

    def test_is_symmetric(self) -> None:
        gen = np.random.RandomState(4)
        a, b = gen.randn(100), gen.randn(100)
        self.assertAlmostEqual(spearman(a, b), spearman(b, a), places=12)


class TrueColdStartMassTest(unittest.TestCase):
    def setUp(self) -> None:
        users, ads = make_world(1_500, 300, USER_DIM, AD_DIM, seed=42)
        gen = torch.Generator().manual_seed(5000)
        self.users = users[torch.randint(0, 1_500, (600,), generator=gen)]
        self.candidate_ads = ads[torch.randint(0, 300, (600, TOP_K), generator=gen)]
        self.log_policy = self._policy(0.2, 242)
        self.log_ads = top1_from_slate(
            self.log_policy, self.users, self.candidate_ads, TOP_K
        )

    def _policy(self, offset: float, seed: int) -> Policy:
        center = torch.full((USER_DIM + AD_DIM,), offset)
        return make_policy(
            48,
            center[:USER_DIM],
            center[USER_DIM:],
            0.3,
            (0.3, 0.5),
            (0.1, 1.0),
            seed=seed,
        )

    def _mass(self, offset: float, tau: float = 0.1) -> float:
        candidate = self._policy(offset, 342)
        served = {
            "p1": top1_from_slate(candidate, self.users, self.candidate_ads, TOP_K)
        }
        return true_cold_start_mass(
            self.log_policy,
            {"p1": candidate},
            self.users,
            self.log_ads,
            served,
            "p1",
            tau=tau,
        )

    def test_is_a_fraction(self) -> None:
        mass = self._mass(0.0)
        self.assertGreaterEqual(mass, 0.0)
        self.assertLessEqual(mass, 1.0)

    def test_a_candidate_far_from_logging_has_more_cold_start(self) -> None:
        """Displacing a candidate moves it off the logged support."""
        self.assertLessEqual(self._mass(0.2), self._mass(0.95))

    def test_is_monotone_in_tau(self) -> None:
        self.assertLessEqual(self._mass(0.5, tau=0.01), self._mass(0.5, tau=0.2))

    def test_n_eval_limits_the_rows_measured(self) -> None:
        candidate = self._policy(0.5, 342)
        served = {
            "p1": top1_from_slate(candidate, self.users, self.candidate_ads, TOP_K)
        }
        args = (
            self.log_policy,
            {"p1": candidate},
            self.users,
            self.log_ads,
            served,
            "p1",
        )
        full = true_cold_start_mass(*args)
        subset = true_cold_start_mass(*args, n_eval=100)
        self.assertGreaterEqual(full, 0.0)
        self.assertGreaterEqual(subset, 0.0)
        self.assertLessEqual(subset, 1.0)
