# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generating the logged dataset a trial evaluates against.

One impression is a user plus a slate of `top_k` candidate ads. The logging
policy ranks the slate and serves its top-1; the oracle scores the served pair
to produce the reward. What gets logged is the impression feature
`z = [user, served_ad]` and that reward, with no record of why the ad was
chosen: selection is a hard argmax, so there is no action propensity to store.
That absence is the problem the paper addresses.

The logged impressions are then partitioned into disjoint domain-classifier,
reward-model training, calibration, and evaluation folds.

Each stage below reseeds explicitly, so the stages are independent of one
another. The seed offsets are exported as constants so that a trial can be
reproduced exactly.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch

from .kernels import score_pairs, top1_from_slate
from .oracle import Oracle
from .policy import make_policy, Policy

# Offsets applied to a trial's base seed. Distinct per stage so that changing
# one part of a trial does not perturb the others.
IMPRESSION_SEED_OFFSET = 5000
LOGGING_CENTER_SEED_OFFSET = 100
LOGGING_POLICY_SEED_OFFSET = 200
SPLIT_SEED_OFFSET = 6000

# The logging policy's kernel center is drawn from this box, which keeps it
# offset from the oracle's center and therefore imperfect.
LOGGING_CENTER_RANGE = 0.5

TRAIN_FRACTION = 0.4
CALIBRATION_FRACTION = 0.2


class LoggedData(TypedDict):
    """One trial's logged impressions.

    `rewards` is clamped to [0, 1] for use as a binary-cross-entropy target,
    while `v_log` is the mean of the unclamped oracle reward, which is the
    logging policy's true value.
    """

    served_ads: torch.Tensor
    rewards: torch.Tensor
    features: torch.Tensor
    v_log: float


class DataSplit(TypedDict):
    """Row indices for each stage of a trial.

    `dc` is the fold the domain classifier is fitted on. Under cross-fitting,
    the default, it is disjoint from `train`, so the weights a reward-model row
    receives come from a classifier that never saw that row. Without
    cross-fitting it is the same rows as `train`.
    """

    dc: torch.Tensor
    train: torch.Tensor
    calibration: torch.Tensor
    evaluation: torch.Tensor


def clamp_reward(values: torch.Tensor) -> torch.Tensor:
    """Clamp oracle scores into [0, 1] so they are valid Bernoulli targets."""
    return values.clamp(0.0, 1.0)


def sample_impressions(
    users: torch.Tensor,
    ads: torch.Tensor,
    n_impressions: int,
    top_k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw impressions: one user and a slate of `top_k` ads, both with replacement.

    Returns `(sampled_users, candidate_ads)` shaped `[N, user_dim]` and
    `[N, top_k, ad_dim]`. The two draws consume the generator in this order.
    """
    torch.manual_seed(seed)
    sampled_users = users[torch.randint(0, users.shape[0], (n_impressions,))]
    candidate_ads = ads[torch.randint(0, ads.shape[0], (n_impressions, top_k))]
    return sampled_users, candidate_ads


def make_logging_policy(
    n_kernels: int,
    user_dim: int,
    ad_dim: int,
    delta: float,
    sigma_range: tuple[float, float],
    alpha_range: tuple[float, float],
    center_seed: int,
    policy_seed: int,
) -> tuple[Policy, torch.Tensor]:
    """Draw the logging policy and return it with its center.

    The center is drawn from numpy rather than torch, on its own generator, so
    it is independent of the kernel draw. Returning it matters because the
    oracle-to-logging axis is what candidate positions are expressed in.
    """
    rng = np.random.RandomState(center_seed)
    center_u = torch.tensor(
        rng.uniform(-LOGGING_CENTER_RANGE, LOGGING_CENTER_RANGE, size=user_dim),
        dtype=torch.float32,
    )
    center_a = torch.tensor(
        rng.uniform(-LOGGING_CENTER_RANGE, LOGGING_CENTER_RANGE, size=ad_dim),
        dtype=torch.float32,
    )
    policy = make_policy(
        n_kernels, center_u, center_a, delta, sigma_range, alpha_range, seed=policy_seed
    )
    return policy, torch.cat([center_u, center_a])


def collect_logged_data(
    log_policy: Policy,
    oracle: Oracle,
    sampled_users: torch.Tensor,
    candidate_ads: torch.Tensor,
    top_k: int = 10,
) -> LoggedData:
    """Serve each slate under the logging policy and record what was observed."""
    served_ads = top1_from_slate(log_policy, sampled_users, candidate_ads, top_k)
    with torch.no_grad():
        oracle_reward = score_pairs(oracle, sampled_users, served_ads)
    return LoggedData(
        served_ads=served_ads,
        rewards=clamp_reward(oracle_reward),
        features=torch.cat([sampled_users, served_ads], dim=1),
        v_log=oracle_reward.mean().item(),
    )


def split_indices(
    n_impressions: int,
    seed: int,
    train_fraction: float = TRAIN_FRACTION,
    calibration_fraction: float = CALIBRATION_FRACTION,
) -> DataSplit:
    """Partition impressions into disjoint train, calibration, and eval folds.

    Evaluation takes the remainder, so the default split is 40/20/40.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(n_impressions)
    n_train = int(n_impressions * train_fraction)
    n_calibration = int(n_impressions * calibration_fraction)
    train = perm[:n_train]
    return DataSplit(
        dc=train,
        train=train,
        calibration=perm[n_train : n_train + n_calibration],
        evaluation=perm[n_train + n_calibration :],
    )


def crossfit_split_indices(
    n_impressions: int,
    seed: int,
    n_dc: int,
    n_train: int,
    n_calibration: int,
) -> DataSplit:
    """Four disjoint folds, sized by absolute count rather than fraction.

    Under cross-fitting the classifier and the reward model must not share
    rows. Evaluation takes the remainder.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(n_impressions)
    dc_end = n_dc
    train_end = dc_end + n_train
    cal_end = train_end + n_calibration
    if cal_end >= n_impressions:
        raise ValueError(
            f"cross-fit folds need {cal_end} rows plus an evaluation fold, "
            f"but n_impressions is {n_impressions}"
        )
    return DataSplit(
        dc=perm[:dc_end],
        train=perm[dc_end:train_end],
        calibration=perm[train_end:cal_end],
        evaluation=perm[cal_end:],
    )
