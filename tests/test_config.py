# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for experiment configuration loading.

The shipped config files are parsed here so a typo in one is a test failure
rather than a surprise several minutes into a run.
"""

from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

import yaml
from adsim.config import TrialConfig

# The configs directory sits either next to this file or at the repository
# root, one level up.
_TEST_DIR = Path(__file__).parent
CONFIG_DIR = (
    _TEST_DIR / "configs"
    if (_TEST_DIR / "configs").is_dir()
    else _TEST_DIR.parent / "configs"
)


def _load(name: str) -> TrialConfig:
    with open(CONFIG_DIR / name) as handle:
        return TrialConfig.from_dict(yaml.safe_load(handle))


class TrialConfigTest(unittest.TestCase):
    def test_defaults_are_the_paper_setting(self) -> None:
        config = TrialConfig()
        self.assertTrue(config.crossfit)
        self.assertEqual(70_000, config.n_impressions)
        self.assertEqual(30, config.epochs)
        self.assertEqual(100.0, config.clip)
        self.assertEqual(6, len(config.value_mults))
        self.assertEqual(7, len(config.ess_targets))
        self.assertEqual(50, len(config.seeds))

    def test_feature_dim_is_user_plus_ad(self) -> None:
        self.assertEqual(6, TrialConfig(user_dim=3, ad_dim=3).feature_dim)
        self.assertEqual(12, TrialConfig(user_dim=4, ad_dim=8).feature_dim)

    def test_from_dict_coerces_sequences_to_tuples(self) -> None:
        config = TrialConfig.from_dict({"ess_targets": [0.3, 0.6], "hidden": [32, 16]})
        self.assertEqual((0.3, 0.6), config.ess_targets)
        self.assertEqual((32, 16), config.hidden)

    def test_from_dict_rejects_unknown_keys(self) -> None:
        """A typo in a config must fail loudly rather than be ignored."""
        with self.assertRaises(ValueError) as caught:
            TrialConfig.from_dict({"n_impresions": 100})
        self.assertIn("n_impresions", str(caught.exception))

    def test_is_immutable(self) -> None:
        """A config cannot be modified after it is created."""
        config = TrialConfig()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.epochs = 5  # pyre-ignore[41]: intentional


# Every shipped config and the fields it is allowed to differ from the
# defaults in. Declaring a divergence here is how it gets justified; an
# undeclared one is a test failure rather than something review has to catch.
#
# None declares more than `experiment`: every row of the paper's tables is
# produced at one training setup.
DECLARED_DIVERGENCES: dict[str, set[str]] = {
    "table1.yaml": set(),
    "dc_ablation.yaml": {"experiment"},
    "clip_sensitivity.yaml": {"experiment"},
}


class SharedSetupTest(unittest.TestCase):
    """Every experiment trains the same way, so results are comparable."""

    def test_no_config_diverges_without_declaring_it(self) -> None:
        defaults = TrialConfig()
        for name, allowed in DECLARED_DIVERGENCES.items():
            config = _load(name)
            actual = {
                f
                for f in TrialConfig.__dataclass_fields__
                if getattr(config, f) != getattr(defaults, f)
            }
            with self.subTest(config=name):
                self.assertEqual(allowed, actual)

    def test_every_config_is_declared(self) -> None:
        """A new config must be added above, not silently left unchecked."""
        shipped = {p.name for p in CONFIG_DIR.glob("*.yaml")}
        self.assertEqual(shipped, set(DECLARED_DIVERGENCES))

    def test_every_experiment_runs_the_same_seeds(self) -> None:
        for name in DECLARED_DIVERGENCES:
            with self.subTest(config=name):
                self.assertEqual(TrialConfig().seeds, _load(name).seeds)

    def test_every_experiment_trains_for_the_same_epochs(self) -> None:
        for name in DECLARED_DIVERGENCES:
            with self.subTest(config=name):
                self.assertEqual(30, _load(name).epochs)
                self.assertEqual(100.0, _load(name).clip)


class ShippedConfigsTest(unittest.TestCase):
    def test_experiment_types_dispatch_correctly(self) -> None:
        self.assertEqual("trial", _load("table1.yaml").experiment)
        self.assertEqual("dc_ablation", _load("dc_ablation.yaml").experiment)
        self.assertEqual("clip_sweep", _load("clip_sensitivity.yaml").experiment)

    def test_default_is_cross_fitted(self) -> None:
        """The bound assumes the reward model's weights come from a classifier
        that never saw its rows, so that is the setting everything runs at."""
        config = _load("table1.yaml")
        self.assertTrue(config.crossfit)
        self.assertEqual((20_000, 20_000, 10_000), config.crossfit_counts)
        self.assertEqual(20_000, config.n_impressions - sum(config.crossfit_counts))

    def test_swept_axes_are_present(self) -> None:
        self.assertEqual(
            (1.0, 5.0, 10.0, 20.0, 100.0), _load("clip_sensitivity.yaml").clip_values
        )
        self.assertEqual(3000, _load("dc_ablation.yaml").dc_eval_n)
