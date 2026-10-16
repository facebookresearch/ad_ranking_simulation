# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Is the domain classifier good enough to be believed.

Effective sample size is the usual overlap diagnostic, but it is not sufficient
on its own: a classifier that has learned nothing outputs near-uniform
probabilities, which produce near-uniform weights and therefore a *high* ESS.
Degradation looks like safety. These are the measurements that separate the two
cases.

Held-out quality:

- `multiclass_nll` on impressions the classifier never saw, and the gap against
  its training NLL, which exposes overfitting.
- `logging_class_ece`, calibration of `P(logging | z)` specifically. That
  probability is the denominator of every density ratio, so its calibration
  matters more than overall accuracy.
- `candidate_confusion`, how well candidates are told apart from each other
  rather than from the logging policy.

Because the simulator knows the true policies, two quantities that are
unavailable in production can also be computed, which is what makes the
ablation interpretable rather than merely suggestive:

- `oracle_density_ratios`, the density ratios implied by the true kernel
  scores, so estimated ratios can be ranked against them.
- `true_cold_start_mass`, the actual cold-start mass, so the estimated one can
  be checked rather than trusted.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

from .kernels import KernelParams, score_pairs

DEFAULT_ECE_BINS = 15
DEFAULT_COLD_START_TAU = 0.1


def mask_ad_features(features: torch.Tensor, n_masked: int) -> torch.Tensor:
    """Zero the last `n_masked` feature columns.

    Impression features are the user concatenated with the served ad, so
    trailing columns are ad columns. Zeroing them removes exactly the
    information that distinguishes one policy's choices from another's, driving
    the representation toward policy-agnostic without changing its shape.
    """
    if n_masked <= 0:
        return features
    masked = features.clone()
    masked[:, features.shape[1] - n_masked :] = 0.0
    return masked


def multiclass_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Negative log likelihood of the true class."""
    n = len(labels)
    true_class = np.clip(probs[np.arange(n), labels.astype(int)], 1e-12, 1.0)
    return float(-np.mean(np.log(true_class)))


def logging_class_ece(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = DEFAULT_ECE_BINS
) -> float:
    """Expected calibration error of `P(logging | z)` as a binary predictor.

    Every density ratio divides by this probability, so its calibration is what
    the weights actually depend on.
    """
    p_logging = probs[:, 0]
    is_logging = (labels == 0).astype(float)
    n = len(labels)
    error = 0.0
    for index in range(n_bins):
        low, high = index / n_bins, (index + 1) / n_bins
        in_bin = (p_logging > low) & (p_logging <= high)
        if in_bin.sum() > 0:
            error += abs(is_logging[in_bin].mean() - p_logging[in_bin].mean()) * (
                in_bin.sum() / n
            )
    return float(error)


def candidate_confusion(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Accuracy among candidate rows, and its complement.

    Restricted to rows the candidates generated, so this measures whether
    candidates are distinguishable *from each other*, which per-policy ratios
    need and overall accuracy hides.
    """
    is_candidate = labels > 0
    if is_candidate.sum() == 0:
        return float("nan"), float("nan")
    predicted = probs[is_candidate][:, 1:].argmax(1) + 1
    accuracy = float((predicted == labels[is_candidate]).mean())
    return accuracy, float(1.0 - accuracy)


def spearman(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Rank correlation, ranking ties by position.

    Ties are not averaged: `argsort` of `argsort` assigns distinct ranks to
    equal values in the order they appear. A constant input therefore receives
    ranks `0..n-1` and can correlate perfectly with anything monotone, and the
    guard below only fires for inputs of length one.

    That matters when reading the ablation's ratio correlation: a domain
    classifier that has collapsed to uniform weights produces a constant
    estimated-ratio vector, and this reports a correlation for it rather than
    declining to answer.
    """
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    rank_first = np.argsort(np.argsort(first, kind="stable")).astype(float)
    rank_second = np.argsort(np.argsort(second, kind="stable")).astype(float)
    if rank_first.std() < 1e-12 or rank_second.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rank_first, rank_second)[0, 1])


def true_cold_start_mass(
    log_policy: KernelParams,
    candidates: Mapping[str, KernelParams],
    users: torch.Tensor,
    log_ads: torch.Tensor,
    served_ads: Mapping[str, torch.Tensor],
    name: str,
    n_eval: int | None = None,
    tau: float = DEFAULT_COLD_START_TAU,
) -> float:
    """Cold-start mass computed from the true policies rather than estimated.

    Forms the posterior a perfectly calibrated classifier would output, by
    normalizing each policy's kernel scores into a distribution over its own
    served impressions and applying Bayes with equal priors, then reports the
    fraction of the candidate's impressions where that true `P(logging | z)`
    falls below `tau`.

    The normalizers are computed over the whole fold while the mass is measured
    on the first `n_eval` rows. Those are deliberately different: the
    normalizers are estimates of each policy's score mass and want all the data,
    whereas the mass only needs enough rows to be stable.

    Simulator-only: it needs the policies themselves, which production does not
    have. It is the ground truth the estimated mass is scored against.
    """
    normalizers = {
        "logging": score_pairs(log_policy, users, log_ads).sum().clamp(min=1e-10)
    }
    for other, policy in candidates.items():
        normalizers[other] = (
            score_pairs(policy, users, served_ads[other]).sum().clamp(min=1e-10)
        )

    limit = len(users) if n_eval is None else n_eval
    target_users = users[:limit]
    target_ads = served_ads[name][:limit]

    p_logging = (
        score_pairs(log_policy, target_users, target_ads) / normalizers["logging"]
    )
    denominator = p_logging.clone()
    for other, policy in candidates.items():
        denominator = (
            denominator
            + score_pairs(policy, target_users, target_ads) / (normalizers[other])
        )
    q0_star = (p_logging / denominator.clamp(min=1e-10)).numpy()
    return float((q0_star < tau).mean())
