# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Positioning candidate policies at a target value and overlap level.

Experiments need candidates that differ from the logging policy by a chosen
amount of value *and* a chosen amount of distribution shift, so that estimator
behavior can be read as a function of overlap. This module provides that
control.

A candidate is placed by moving its kernel center away from the oracle's along
two directions: `alpha` along the axis joining the oracle and logging centers,
and `beta` along an orthogonal direction. Sweeping that 2-D grid and recording
each point's value multiplier and effective sample size yields a lookup table;
`lookup_alpha_beta` then reads off the position achieving a requested
(value multiplier, ESS) pair.

The ESS reported here comes from kernel-score density ratios computed directly
from the two policies, not from the domain classifier. That keeps the overlap
axis of the experiment independent of the estimator being evaluated.
"""

from __future__ import annotations

from typing import Mapping, Sequence, TypedDict

import numpy as np
import torch

from .kernels import KernelParams, score_pairs, top1_from_slate
from .oracle import Oracle
from .policy import make_policy, Policy

# Grid searched by `calibration_sweep`, in units of the oracle-to-logging
# distance (alpha) and of the orthogonal direction (beta).
ALPHA_GRID: tuple[float, float, float] = (-0.5, 1.05, 0.1)
BETA_GRID: tuple[float, float, float] = (-1.0, 1.05, 0.1)


def shift_axis(
    oracle_center: torch.Tensor, log_center: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Unit vector from the oracle center toward the logging center, and the gap.

    Candidate positions are expressed in units of this gap, so that alpha means
    the same thing regardless of where the logging policy happened to land.
    Degenerate when the two coincide, in which case there is no axis to move
    along and the zero vector is returned.
    """
    diff = log_center - oracle_center
    distance = diff.norm().item()
    if distance <= 1e-10:
        return torch.zeros_like(diff), distance
    return diff / distance, distance


def orthogonal_directions(
    v_along: torch.Tensor, count: int, seed: int
) -> list[torch.Tensor]:
    """Unit vectors orthogonal to `v_along`, one per candidate policy.

    Giving each candidate its own orthogonal direction stops them from being
    collinear, so they are not merely more and less extreme versions of the
    same shift.
    """
    rng = np.random.RandomState(seed)
    directions = []
    for _ in range(count):
        r = torch.tensor(rng.randn(v_along.shape[0]), dtype=torch.float32)
        residual = r - (r @ v_along) * v_along
        directions.append(residual / residual.norm().clamp(min=1e-10))
    return directions


def sweep_orthogonal_direction(v_along: torch.Tensor, seed: int) -> torch.Tensor:
    """The single orthogonal direction the calibration grid is spanned by.

    Separate from `orthogonal_directions` because the sweep normalizes without
    the numerical floor that the per-candidate draws use.
    """
    rng = np.random.RandomState(seed)
    r = torch.tensor(rng.randn(v_along.shape[0]), dtype=torch.float32)
    residual = r - (r @ v_along) * v_along
    return residual / residual.norm()


class WeightInfo(TypedDict):
    """Diagnostics for a set of importance weights."""

    ess: float
    N: int
    w_mean: float
    w_std: float
    w_max: float


class SweepPoint(TypedDict):
    """One grid position and what it achieves."""

    alpha: float
    beta: float
    V_top1: float
    ess: float
    vmult: float


def compute_pairwise_density_weights(
    policy_1: KernelParams,
    policy_2: KernelParams,
    users_1: torch.Tensor,
    ads_1: torch.Tensor,
    users_2: torch.Tensor,
    ads_2: torch.Tensor,
    clip: float = 5.0,
    smoothing: str = "clip",
    smooth_alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, WeightInfo]:
    """Kernel-score density ratios taking `policy_1`'s samples to `policy_2`.

    Each policy's scores are normalized over its own served ads to form a
    discrete distribution, and the ratio of those is the weight. Returns the
    clipped weights, the unclipped weights, and diagnostics including ESS
    normalized to [0, 1].

    Used only to position candidates; estimator weights come from the domain
    classifier instead.
    """
    n = users_1.shape[0]
    with torch.no_grad():
        s1_own = score_pairs(policy_1, users_1, ads_1)
        s2_own = score_pairs(policy_2, users_2, ads_2)
        s2_at_1 = score_pairs(policy_2, users_1, ads_1)

    p1_i = s1_own / s1_own.sum().clamp(min=1e-10)
    p2_i = s2_at_1 / s2_own.sum().clamp(min=1e-10)
    raw_weights = p2_i / p1_i.clamp(min=1e-10)

    if smoothing == "additive":
        eps = smooth_alpha / n
        corrected = ((p2_i + eps) / (p1_i + eps)).clamp(max=clip)
    else:
        corrected = raw_weights.clamp(max=clip)

    info = WeightInfo(
        ess=(corrected.sum() ** 2 / (corrected**2).sum()).item() / n,
        N=n,
        w_mean=corrected.mean().item(),
        w_std=corrected.std().item(),
        w_max=corrected.max().item(),
    )
    return corrected, raw_weights, info


def kernel_mixture_ess(
    log_policy: KernelParams,
    candidates: Mapping[str, KernelParams],
    names: Sequence[str],
    users: torch.Tensor,
    log_ads: torch.Tensor,
    served_ads: Mapping[str, torch.Tensor],
) -> float:
    """Unclipped overlap of the candidate mixture with the logging policy.

    Averages the true kernel-score density ratios over candidates and reports
    the effective sample size of the result. Because it uses the true policies
    and applies no clip, it is a property of the experiment rather than of the
    estimator, which is what makes it the right axis to bin a clip sweep on:
    binning by the classifier's own clipped ESS would let the x-axis move with
    the quantity under study.
    """
    ratios = []
    for name in names:
        _, raw, _ = compute_pairwise_density_weights(
            log_policy,
            candidates[name],
            users,
            log_ads,
            users,
            served_ads[name],
            clip=1e10,
        )
        ratios.append(raw)
    omega = torch.stack(ratios).mean(dim=0)
    n = omega.shape[0]
    return (omega.sum() ** 2 / (omega**2).sum()).item() / n


def _subsample_impressions(
    sampled_users: torch.Tensor,
    candidate_ads: torch.Tensor,
    log_top1_ads: torch.Tensor,
    calib_n_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thin the impression set; the sweep is quadratic in grid size."""
    n_full = sampled_users.shape[0]
    if calib_n_samples >= n_full:
        return sampled_users, candidate_ads, log_top1_ads
    torch.manual_seed(seed + 7777)
    idx = torch.randperm(n_full)[:calib_n_samples]
    return sampled_users[idx], candidate_ads[idx], log_top1_ads[idx]


def _evaluate_position(
    position: torch.Tensor,
    oracle: Oracle,
    log_policy: KernelParams,
    users: torch.Tensor,
    candidate_ads: torch.Tensor,
    log_top1_ads: torch.Tensor,
    n_kernels: int,
    delta: float,
    sigma_range: tuple[float, float],
    alpha_range: tuple[float, float],
    top_k: int,
    seed: int,
    clip: float,
) -> tuple[float, float]:
    """Value and overlap of a candidate placed at `position`.

    Every grid point reseeds with the same `seed`, so all candidates share
    kernel offsets and differ only by center.
    """
    user_dim = users.shape[-1]
    candidate: Policy = make_policy(
        n_kernels,
        position[:user_dim],
        position[user_dim:],
        delta,
        sigma_range,
        alpha_range,
        seed=seed,
    )
    served = top1_from_slate(candidate, users, candidate_ads, top_k)
    with torch.no_grad():
        oracle_reward = score_pairs(oracle, users, served)
    _, _, info = compute_pairwise_density_weights(
        log_policy, candidate, users, log_top1_ads, users, served, clip=clip
    )
    return oracle_reward.mean().item(), info["ess"]


def calibration_sweep(
    oracle: Oracle,
    log_policy: KernelParams,
    sampled_users: torch.Tensor,
    candidate_ads: torch.Tensor,
    log_top1_ads: torch.Tensor,
    oracle_center: torch.Tensor,
    v_along: torch.Tensor,
    w_orth: torch.Tensor,
    distance: float,
    v_log: float,
    n_kernels: int,
    delta: float,
    sigma_range: tuple[float, float],
    alpha_range: tuple[float, float],
    top_k: int,
    seed: int,
    clip: float = 5.0,
    calib_n_samples: int = 10_000,
) -> list[SweepPoint]:
    """Map the (alpha, beta) placement grid to achieved value and overlap.

    `v_along` is the unit vector from the oracle center toward the logging
    center and `distance` their separation, so alpha is in units of that gap;
    `w_orth` is a unit vector orthogonal to it. `v_log` normalizes value into a
    multiplier of the logging policy's value.
    """
    users, ads, log_ads = _subsample_impressions(
        sampled_users, candidate_ads, log_top1_ads, calib_n_samples, seed
    )

    results: list[SweepPoint] = []
    for alpha in np.arange(*ALPHA_GRID):
        for beta in np.arange(*BETA_GRID):
            position = (
                oracle_center + float(alpha) * distance * v_along + float(beta) * w_orth
            )
            value, ess = _evaluate_position(
                position,
                oracle,
                log_policy,
                users,
                ads,
                log_ads,
                n_kernels,
                delta,
                sigma_range,
                alpha_range,
                top_k,
                seed,
                clip,
            )
            results.append(
                SweepPoint(
                    alpha=float(alpha),
                    beta=float(beta),
                    V_top1=value,
                    ess=ess,
                    vmult=value / v_log,
                )
            )
    return results


def lookup_alpha_beta(
    sweep: list[SweepPoint],
    target_vmult: float,
    target_ess: float,
    tol: float = 0.03,
) -> SweepPoint | None:
    """Closest grid point to `target_ess` among those hitting `target_vmult`.

    Value match is the hard constraint and overlap the soft one: candidates
    must sit at the requested value separation for regret to be comparable
    across ESS regimes. The tolerance is widened once before giving up.
    """
    matches = [r for r in sweep if abs(r["vmult"] - target_vmult) <= tol]
    if not matches:
        matches = [r for r in sweep if abs(r["vmult"] - target_vmult) <= tol * 2]
    if not matches:
        return None
    return min(matches, key=lambda r: abs(r["ess"] - target_ess))
