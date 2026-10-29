# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration for one experiment.

Every setting that affects a result lives here, so a result is fully described
by a config file plus a seed list.

The defaults below are the setting every experiment shares: cross-fitted, 70k
impressions, 30 epochs, weights clipped at 100, 50 seeds. Each config file
states only the axis it sweeps, and `tests/test_config.py` asserts that each one
differs from these defaults in only its declared keys.

Cross-fitting is the default because the value-error bound assumes the reward
model's weights come from a classifier that never saw its training rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class TrialConfig:
    """Simulator, estimator, and training settings for one experiment."""

    # World. Drawn once and shared by every seed, so the population is not a
    # source of between-seed variance.
    num_users: int = 10_000
    num_ads: int = 1_000
    user_dim: int = 3
    ad_dim: int = 3
    world_seed: int = 42

    # Oracle.
    oracle_ctr: float = 0.15
    oracle_n_kernels: int = 400
    oracle_delta: float = 0.5
    oracle_sigma_range: tuple[float, float] = (0.3, 0.5)
    oracle_alpha_range: tuple[float, float] = (0.2, 1.0)

    # Candidate and logging policies.
    n_kernels: int = 200
    delta: float = 0.3
    sigma_range: tuple[float, float] = (0.3, 0.5)
    alpha_range: tuple[float, float] = (0.1, 1.0)
    value_mults: tuple[float, ...] = (0.95, 1.0, 1.05, 1.1, 1.15, 1.2)
    ess_targets: tuple[float, ...] = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)

    # Logged data. 70k rather than 50k because cross-fitting costs a fold and
    # the extra 20k pays for it, leaving the reward model, calibration, and
    # evaluation folds the same size they are without it.
    n_impressions: int = 70_000

    # Cross-fitting. The domain classifier is fitted on a fold disjoint from
    # the reward model's, sized by the absolute counts below rather than by
    # fraction. Folds are DC / reward / calibration; evaluation takes the
    # remaining 20k.
    crossfit: bool = True
    crossfit_counts: tuple[int, int, int] = (20_000, 20_000, 10_000)
    top_k: int = 10
    calib_n_samples: int = 10_000

    # Reward models.
    hidden: tuple[int, ...] = (64,)
    epochs: int = 30
    calibrate: bool = True

    # Domain classifier and weights.
    dc_epochs: int = 4
    dc_folds: int = 5
    dc_lr: float = 0.01
    clip: float = 100.0

    # Clip values swept by the clip-sensitivity experiment. 100 is effectively
    # no clip at this scale.
    clip_values: tuple[float, ...] = (1.0, 5.0, 10.0, 20.0, 100.0)

    # Which experiment to run. `trial` compares estimators at each overlap
    # level; `dc_ablation` additionally degrades the domain classifier six ways.
    experiment: str = "trial"

    # Impressions per policy used for classifier-quality measurement. Only read
    # by the ablation.
    dc_eval_n: int = 3000

    # Seeds. Each produces one record per ESS target.
    seeds: tuple[int, ...] = field(
        default_factory=lambda: tuple(42 + s * 1000 for s in range(50))
    )

    @property
    def feature_dim(self) -> int:
        """Impression features are the user concatenated with the served ad."""
        return self.user_dim + self.ad_dim

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrialConfig:
        """Build from parsed YAML, coercing sequences to the declared tuples."""
        known = {f.name: f for f in fields(cls)}
        unknown = set(values) - set(known)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        coerced: dict[str, Any] = {
            name: tuple(value) if isinstance(value, list) else value
            for name, value in values.items()
        }
        return cls(**coerced)
