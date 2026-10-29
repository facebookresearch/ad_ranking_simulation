# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Candidate and logging policies.

A policy is drawn from the same factored-kernel family as the oracle (see
`adsim.kernels`) and ranks a slate of candidate ads for each user, serving the
top-1. Because selection is a hard argmax there is no action propensity to log,
which is the regime the paper addresses.

Policies are built unscaled. `calibrate_policy_top1` then rescales amplitudes
so a policy's own scores sum to the oracle's over the ads it selects, putting
every candidate on a comparable scale before its value is measured.
"""

from __future__ import annotations

from typing import NotRequired

import torch

from .kernels import draw_kernels, KernelModel, score_pairs, top1_from_slate
from .oracle import Oracle


class Policy(KernelModel):
    """A kernel model, optionally annotated with its true top-1 value.

    `V_top1` is set by `calibrate_policy_top1` and is the oracle-measured mean
    reward of the ads this policy selects.
    """

    V_top1: NotRequired[float]


def make_policy(
    n_kernels: int,
    center_u: torch.Tensor,
    center_a: torch.Tensor,
    delta: float,
    sigma_range: tuple[float, float],
    alpha_range: tuple[float, float],
    seed: int = 42,
) -> Policy:
    """Draw an unscaled policy. Amplitude scaling is the caller's job.

    Consumes the seeded generator identically to `make_oracle`, so a policy and
    an oracle built with the same seed and geometry share kernel placement.
    """
    torch.manual_seed(seed)
    center_u = torch.as_tensor(center_u, dtype=torch.float32)
    center_a = torch.as_tensor(center_a, dtype=torch.float32)

    kernels = draw_kernels(
        n_kernels, center_u, center_a, delta, sigma_range, alpha_range
    )
    return Policy(
        mu_u=kernels["mu_u"],
        mu_a=kernels["mu_a"],
        sigma_u=kernels["sigma_u"],
        sigma_a=kernels["sigma_a"],
        alphas=kernels["alphas"],
        K=n_kernels,
        center_u=center_u,
        center_a=center_a,
        delta=delta,
    )


def calibrate_policy_top1(
    policy: Policy,
    oracle: Oracle,
    sampled_users: torch.Tensor,
    candidate_ads: torch.Tensor,
    top_k: int = 10,
) -> Policy:
    """Rescale `policy` so its scores sum to the oracle's on its own top-1 ads.

    Mutates `policy` in place and also returns it. Records `V_top1`, the
    oracle-measured mean reward of the selected ads, which is the ground-truth
    value every estimator is scored against.
    """
    top1_ads = top1_from_slate(policy, sampled_users, candidate_ads, top_k)

    with torch.no_grad():
        policy_scores_top1 = score_pairs(policy, sampled_users, top1_ads)
    with torch.no_grad():
        oracle_scores_top1 = score_pairs(oracle, sampled_users, top1_ads)

    sum_policy = policy_scores_top1.sum().item()
    sum_oracle = oracle_scores_top1.sum().item()

    if sum_policy > 1e-10:
        policy["alphas"] = policy["alphas"] * (sum_oracle / sum_policy)

    policy["V_top1"] = oracle_scores_top1.mean().item()
    return policy


def evaluate_policy_value(
    policy: Policy,
    oracle: Oracle,
    sampled_users: torch.Tensor,
    candidate_ads: torch.Tensor,
    top_k: int = 10,
) -> float:
    """Monte-Carlo true value of `policy`: oracle reward of the ads it serves."""
    selected_ads = top1_from_slate(policy, sampled_users, candidate_ads, top_k)
    with torch.no_grad():
        oracle_rewards = score_pairs(oracle, sampled_users, selected_ads)
    return oracle_rewards.mean().item()
