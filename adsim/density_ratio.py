# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Density ratios from a calibrated domain classifier.

Given a classifier that predicts which policy produced an impression, the ratio
of its class probabilities estimates the ratio of the underlying densities:

    Omega_j(z) = P(j | z) / P(logging | z)  ~  p_j(z) / p_0(z)

Two forms are used. `compute_per_domain_density_ratios` keeps one ratio per
candidate, which is what a per-policy weighted reward model trains against.
`compute_importance_weights` collapses them into the single mixture weight

    Omega(z) = (1 - P(logging|z)) / (K * P(logging|z))

which is the average of the per-candidate ratios, and is what the shared
mixture-weighted reward model trains against.

Both clip, because an impression the logging policy almost never served has an
unbounded ratio and would otherwise dominate. Clipping trades bias for
variance; `compute_weight_diagnostics` reports where that trade landed.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_EPS = 1e-6
DEFAULT_CLIP = 5.0


class WeightDiagnostics(TypedDict):
    """Weight distribution summary.

    `ess` is the raw effective sample size and `ess_ratio` is that divided by
    N. The overlap diagnostic reported in the paper is the ratio.
    """

    max_w: float
    p95_w: float
    p99_w: float
    mean_w: float
    std_w: float
    e_w2: float
    ess: float
    ess_ratio: float
    n_total: float


def compute_calibrated_probs(
    classifier: nn.Module,
    embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Temperature-scaled class probabilities, `softmax(logits / T)`.

    Always use this rather than the classifier's own `predict_proba`: the raw
    probabilities are uncalibrated and are not valid densities.
    """
    classifier.eval()
    with torch.no_grad():
        logits = classifier(embeddings)
        probs = F.softmax(logits / temperature, dim=-1)
    return probs


def compute_importance_weights(
    calibrated_probs: torch.Tensor,
    logging_class: int = 0,
    num_target_policies: int = 5,
    clip: float = DEFAULT_CLIP,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """Mixture weight per impression, `(1 - p0) / (K * p0)`.

    Equal to the mean of the per-candidate density ratios, so a reward model
    trained with these weights targets the uniform mixture over candidates
    rather than any single one.
    """
    p0 = calibrated_probs[:, logging_class].numpy()
    p0 = np.maximum(p0, eps)
    weights = (1.0 - p0) / (num_target_policies * p0)
    return np.clip(weights, 0, clip)


def compute_per_domain_density_ratios(
    calibrated_probs: torch.Tensor,
    logging_class: int = 0,
    target_classes: list[int] | None = None,
    clip: float = DEFAULT_CLIP,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """One density ratio per candidate: `[N, len(target_classes)]`.

    Column `j` holds `P(target_classes[j] | z) / P(logging | z)`. Defaults to
    every non-logging class in sorted order.
    """
    n_samples, n_classes = calibrated_probs.shape

    if target_classes is None:
        target_classes = sorted(c for c in range(n_classes) if c != logging_class)

    p_logging = calibrated_probs[:, logging_class].clamp(min=eps)

    ratios = torch.zeros(n_samples, len(target_classes))
    for col_idx, cls in enumerate(target_classes):
        ratios[:, col_idx] = (calibrated_probs[:, cls] / p_logging).clamp(max=clip)
    return ratios


def compute_weight_diagnostics(weights: np.ndarray) -> WeightDiagnostics:
    """Summarize a weight vector, including effective sample size.

    ESS falls as the weights concentrate on few impressions. It is the headline
    overlap diagnostic, but it is not sufficient on its own: a domain
    classifier that has learned nothing outputs near-uniform probabilities,
    which produces near-uniform weights and therefore a high ESS.
    """
    n = len(weights)
    sum_w = float(weights.sum())
    sum_w_sq = float((weights**2).sum())
    ess = (sum_w**2) / sum_w_sq if sum_w_sq > 0 else 0.0

    return WeightDiagnostics(
        max_w=float(np.max(weights)),
        p95_w=float(np.percentile(weights, 95)),
        p99_w=float(np.percentile(weights, 99)),
        mean_w=float(np.mean(weights)),
        std_w=float(np.std(weights)),
        e_w2=float(np.mean(weights**2)),
        ess=ess,
        ess_ratio=ess / n if sum_w_sq > 0 and n > 0 else 0.0,
        n_total=float(n),
    )


def compute_importance_weights_from_classifier(
    classifier: nn.Module,
    embeddings: torch.Tensor,
    temperature: float,
    logging_class: int = 0,
    num_target_policies: int = 5,
    clip: float = DEFAULT_CLIP,
) -> np.ndarray:
    """Calibrate then reduce to mixture weights, in one call."""
    calibrated_probs = compute_calibrated_probs(classifier, embeddings, temperature)
    return compute_importance_weights(
        calibrated_probs, logging_class, num_target_policies, clip
    )
