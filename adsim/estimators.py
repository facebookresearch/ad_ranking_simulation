# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Off-policy value estimators.

Every estimator here answers the same question - what would this candidate
policy have been worth - from logged impressions that carry no propensity.
They differ in what they lean on:

- **Direct Method** averages a reward model's predictions over the impressions
  the candidate would have served. No weights at evaluation time, so no
  variance blow-up, but it inherits whatever bias the reward model has.
- **IPS** reweights the observed rewards by the estimated density ratio. No
  reward model, but its variance is governed by the weight distribution, which
  is why effective sample size is the diagnostic to watch.
- **Doubly robust** starts from the Direct Method value and adds a
  weighted correction of the reward model's residuals on logged impressions.

Weighted estimators come in a plain and a self-normalized form. Plain divides
by `n`; self-normalized divides by the weight sum, which trades a little bias
for lower variance when the weights do not average to one.
"""

from __future__ import annotations

import torch

from .reward_model import MLPDMRewardModel, predict_calibrated


def direct_method_value(
    model: MLPDMRewardModel,
    target_features: torch.Tensor,
    temperature: float,
) -> float:
    """Mean predicted reward over the impressions the candidate would serve."""
    return predict_calibrated(model, target_features, temperature).mean().item()


def ips_values(rewards: torch.Tensor, weights: torch.Tensor) -> tuple[float, float]:
    """Inverse propensity scoring, `(plain, self_normalized)`.

    Negative weights are floored at zero; they can only arise from estimation
    noise and a negative density ratio is meaningless.
    """
    clipped = weights.clamp(min=0.0)
    n = rewards.shape[0]
    weighted_sum = (clipped * rewards).sum().item()
    return (
        weighted_sum / n,
        weighted_sum / clipped.sum().clamp(min=1e-10).item(),
    )


def dr_values(
    rewards: torch.Tensor,
    weights: torch.Tensor,
    r_hat_logged: torch.Tensor,
    r_hat_target: torch.Tensor,
) -> tuple[float, float]:
    """Doubly robust value, `(plain, self_normalized)`.

    `r_hat_logged` and `r_hat_target` are the same reward model evaluated on the
    logged impressions and on the ones the candidate would serve. The Direct
    Method term carries the estimate and the weighted residual corrects it, so
    the estimator is consistent if either the reward model or the weights are
    right.
    """
    clipped = weights.clamp(min=0.0)
    n = rewards.shape[0]
    direct = r_hat_target.mean().item()
    correction = (clipped * (rewards - r_hat_logged)).sum().item()
    return (
        direct + correction / n,
        direct + correction / clipped.sum().clamp(min=1e-10).item(),
    )
