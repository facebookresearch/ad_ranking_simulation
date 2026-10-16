# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Scoring an estimator against the oracle.

Estimators are judged on the decision they support, not only on the value they
report, so each is scored on five things:

- **acc1** - did it pick the truly best candidate.
- **regret** - how much true value was given up by picking what it picked.
- **dir_acc** - of the candidates that genuinely differ from the logging
  policy, how many did it place on the correct side of it.
- **mae_v** - absolute error in the value level.
- **mae_delta** - absolute error in the lift over the logging policy.

Everything is expressed as a lift over the logging policy rather than as a raw
value, because a shared additive bias moves every candidate together and leaves
the ranking intact. Each estimator is therefore compared against *its own*
estimate of the logging value, so a biased reward model is not penalized twice.

`cold_start_mass` is separate: it is the diagnostic the theory bounds with, not
a scoring metric.
"""

from __future__ import annotations

from typing import Mapping, Sequence, TypedDict

import numpy as np
import torch

# Candidates whose true lift is smaller than this are excluded from direction
# accuracy; their sign is not meaningfully defined.
DIRECTION_TAU = 1e-4

# Thresholds at which cold-start mass is reported.
COLD_START_TAUS: tuple[float, ...] = (0.01, 0.05, 0.1)


class DecisionMetrics(TypedDict):
    """One estimator's score on one trial."""

    acc1: int
    regret: float
    dir_acc: float
    mae_v: float
    mae_delta: float


def compute_decision_metrics(
    v_true: Mapping[str, float],
    v_hat: Mapping[str, float],
    v_log_true: float,
    v_log_hat: float | Mapping[str, float],
    policy_names: Sequence[str],
    tau: float = DIRECTION_TAU,
) -> DecisionMetrics:
    """Score one estimator's candidate values against the oracle's.

    `v_log_hat` is the estimator's own view of the logging policy's value. Pass
    a float when one shared reward model serves every candidate. Pass a mapping
    when each candidate has its own model, as the per-policy weighted baseline
    does, since then each candidate's lift must be measured against the logging
    prediction of the model that produced it.
    """
    if isinstance(v_log_hat, Mapping):
        baseline = np.array([v_log_hat[p] for p in policy_names])
    else:
        baseline = np.full(len(policy_names), v_log_hat)

    value_true = np.array([v_true[p] for p in policy_names])
    value_hat = np.array([v_hat[p] for p in policy_names])
    delta_true = value_true - v_log_true
    delta_hat = value_hat - baseline

    best_true = int(np.argmax(delta_true))
    best_hat = int(np.argmax(delta_hat))

    directional = [
        int(np.sign(delta_hat[j]) == np.sign(delta_true[j]))
        for j in range(len(policy_names))
        if abs(delta_true[j]) >= tau
    ]

    return DecisionMetrics(
        acc1=int(best_hat == best_true),
        regret=delta_true[best_true] - delta_true[best_hat],
        dir_acc=(np.mean(directional) if directional else np.nan),
        mae_v=np.mean(np.abs(value_hat - value_true)),
        mae_delta=np.mean(np.abs(delta_hat - delta_true)),
    )


def cold_start_mass(
    calibrated_probs: torch.Tensor,
    tau: float,
    logging_class: int = 0,
) -> float:
    """Fraction of a candidate's impressions the logging policy barely reaches.

    An impression whose estimated `P(logging | z)` falls below `tau` sits
    outside the region the logged data covers, so no amount of data fixes the
    reward model there. This quantity is the asymptotic ceiling on value error
    in the paper's decomposition, and unlike that error it is computable from
    logged data alone.

    `calibrated_probs` must be the temperature-corrected probabilities on the
    impressions the *candidate* would serve.
    """
    p_logging = np.asarray(calibrated_probs)[:, logging_class]
    return float((p_logging < tau).mean())


def cold_start_masses(
    calibrated_probs: torch.Tensor,
    taus: Sequence[float] = COLD_START_TAUS,
    logging_class: int = 0,
) -> dict[float, float]:
    """Cold-start mass at each threshold, since the choice of tau is a tradeoff."""
    return {tau: cold_start_mass(calibrated_probs, tau, logging_class) for tau in taus}
