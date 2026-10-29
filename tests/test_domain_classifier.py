# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Behavioral tests for the domain classifier and its calibration."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F
from adsim.domain_classifier import (
    CalibrationInfo,
    fit_temperature,
    get_kfold_oof_logits,
    LinearDomainClassifier,
    MLPDomainClassifier,
    train_calibrated_domain_classifier,
)

FEATURE_DIM = 6
N_PER_POLICY = 300
N_CLASSES = 4
CPU = torch.device("cpu")


def _policy_embeddings(
    separation: float = 0.6,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    gen = torch.Generator().manual_seed(7)
    names = ["logging"] + [f"p{i}" for i in range(1, N_CLASSES)]
    embeddings = {
        name: torch.randn(N_PER_POLICY, FEATURE_DIM, generator=gen) + separation * idx
        for idx, name in enumerate(names)
    }
    return embeddings, {name: idx for idx, name in enumerate(names)}


class MLPDomainClassifierTest(unittest.TestCase):
    def test_output_shape_is_one_logit_per_policy(self) -> None:
        classifier = MLPDomainClassifier(FEATURE_DIM, N_CLASSES)
        logits = classifier(torch.randn(17, FEATURE_DIM))
        self.assertEqual((17, N_CLASSES), tuple(logits.shape))

    def test_predict_proba_is_a_distribution(self) -> None:
        classifier = MLPDomainClassifier(FEATURE_DIM, N_CLASSES)
        probs = classifier.predict_proba(torch.randn(17, FEATURE_DIM))
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(17), atol=1e-5))
        self.assertGreaterEqual(probs.min().item(), 0.0)

    def test_hidden_dims_control_depth(self) -> None:
        deep = MLPDomainClassifier(FEATURE_DIM, N_CLASSES, hidden_dims=[32, 16, 8])
        self.assertEqual([32, 16, 8], deep.hidden_dims)
        self.assertEqual(8, deep.output_layer.in_features)

    def test_defaults_to_one_hidden_layer_of_64(self) -> None:
        self.assertEqual([64], MLPDomainClassifier(FEATURE_DIM, N_CLASSES).hidden_dims)

    def test_rejects_empty_hidden_dims(self) -> None:
        with self.assertRaises(ValueError):
            MLPDomainClassifier(FEATURE_DIM, N_CLASSES, hidden_dims=[])

    def test_linear_variant_has_no_hidden_layer(self) -> None:
        classifier = LinearDomainClassifier(FEATURE_DIM, N_CLASSES)
        self.assertEqual(
            (11, N_CLASSES), tuple(classifier(torch.randn(11, FEATURE_DIM)).shape)
        )


class FitTemperatureTest(unittest.TestCase):
    def test_overconfident_logits_get_temperature_above_one(self) -> None:
        """Scaling true logits up makes the model overconfident; T must undo it."""
        gen = np.random.RandomState(0)
        labels = gen.randint(0, N_CLASSES, size=800)
        logits = gen.randn(800, N_CLASSES).astype(np.float32)
        logits[np.arange(800), labels] += 1.0
        temperature = fit_temperature(logits * 5.0, labels, n_steps=300, verbose=False)
        self.assertGreater(temperature, 1.0)

    def test_underconfident_logits_get_temperature_below_one(self) -> None:
        gen = np.random.RandomState(0)
        labels = gen.randint(0, N_CLASSES, size=800)
        logits = gen.randn(800, N_CLASSES).astype(np.float32)
        logits[np.arange(800), labels] += 1.0
        temperature = fit_temperature(logits * 0.2, labels, n_steps=300, verbose=False)
        self.assertLess(temperature, 1.0)

    def test_temperature_is_positive(self) -> None:
        gen = np.random.RandomState(1)
        logits = gen.randn(200, N_CLASSES).astype(np.float32)
        labels = gen.randint(0, N_CLASSES, size=200)
        self.assertGreater(
            fit_temperature(logits, labels, n_steps=50, verbose=False), 0
        )

    def test_reduces_negative_log_likelihood(self) -> None:
        gen = np.random.RandomState(2)
        labels = gen.randint(0, N_CLASSES, size=800)
        logits = (gen.randn(800, N_CLASSES).astype(np.float32)) * 4.0
        logits[np.arange(800), labels] += 2.0
        temperature = fit_temperature(logits, labels, n_steps=300, verbose=False)
        logit_tensor = torch.tensor(logits)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        before = F.cross_entropy(logit_tensor, label_tensor).item()
        after = F.cross_entropy(logit_tensor / temperature, label_tensor).item()
        self.assertLessEqual(after, before)


class KFoldOofLogitsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        self.policy_embeddings, self.policy_to_class = _policy_embeddings()

    def test_covers_every_impression_exactly_once(self) -> None:
        logits, labels = get_kfold_oof_logits(
            self.policy_embeddings,
            self.policy_to_class,
            n_folds=5,
            epochs=1,
            device=CPU,
            verbose=False,
        )
        self.assertEqual((N_PER_POLICY * N_CLASSES, N_CLASSES), logits.shape)
        self.assertEqual(N_PER_POLICY * N_CLASSES, len(labels))
        # A row left unpredicted would stay all-zero.
        self.assertFalse(bool((np.abs(logits).sum(axis=1) == 0).any()))

    def test_labels_match_the_policy_mapping(self) -> None:
        _, labels = get_kfold_oof_logits(
            self.policy_embeddings,
            self.policy_to_class,
            n_folds=2,
            epochs=1,
            device=CPU,
            verbose=False,
        )
        self.assertEqual(
            sorted(self.policy_to_class.values()), sorted(set(labels.tolist()))
        )


class TrainCalibratedDomainClassifierTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        self.policy_embeddings, self.policy_to_class = _policy_embeddings()

    def _train(
        self, embeddings: dict[str, torch.Tensor] | None = None, epochs: int = 4
    ) -> tuple[torch.nn.Module, float, CalibrationInfo]:
        return train_calibrated_domain_classifier(
            embeddings if embeddings is not None else self.policy_embeddings,
            self.policy_to_class,
            hidden_dims=[64],
            n_folds=5,
            epochs=epochs,
            lr=0.01,
            device=CPU,
            verbose=False,
        )

    def test_returns_classifier_temperature_and_diagnostics(self) -> None:
        classifier, temperature, info = self._train()
        self.assertGreater(temperature, 0.0)
        self.assertEqual(5, info["n_folds"])
        self.assertEqual(N_PER_POLICY * N_CLASSES, info["n_samples"])
        self.assertEqual("mlp", info["classifier_type"])
        self.assertEqual(temperature, info["temperature"])
        self.assertEqual(
            (5, N_CLASSES), tuple(classifier(torch.randn(5, FEATURE_DIM)).shape)
        )

    def test_separable_policies_are_learned(self) -> None:
        """With well-separated policies the classifier must beat chance."""
        _, _, info = self._train()
        self.assertGreater(info["final_accuracy"], 1.0 / N_CLASSES)

    def test_indistinguishable_policies_collapse_to_chance(self) -> None:
        """Identical policies carry no signal, which is the degenerate regime
        the paper's reliability discussion is about."""
        embeddings, _ = _policy_embeddings(separation=0.0)
        _, _, info = self._train(embeddings=embeddings)
        self.assertLess(info["final_accuracy"], 0.5)

    def test_returns_a_frozen_cpu_model(self) -> None:
        """Weights are consumed as densities downstream, never trained further."""
        classifier, _, _ = self._train(epochs=1)
        for param in classifier.parameters():
            self.assertFalse(param.requires_grad)
            self.assertEqual("cpu", param.device.type)
        self.assertFalse(classifier.training)
