# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Factored-kernel reward oracle.

The oracle defines ground-truth reward for the simulator. Its amplitudes are
Monte-Carlo rescaled so the average reward over the user and ad populations
equals a target click-through rate, which fixes the operating point every
policy value is measured against. See `adsim.kernels` for the scoring form.
"""

from __future__ import annotations

import logging

import torch

from .kernels import draw_kernels, KernelModel, score_pairs

logger: logging.Logger = logging.getLogger(__name__)


class Oracle(KernelModel):
    """A kernel model plus `V_cf`, its realized mean reward after rescaling."""

    V_cf: float


def make_oracle(
    target_ctr: float,
    users: torch.Tensor,
    ads: torch.Tensor,
    n_kernels: int = 200,
    center: tuple[torch.Tensor, torch.Tensor] | None = None,
    delta: float = 0.5,
    sigma_range: tuple[float, float] = (0.3, 0.5),
    alpha_range: tuple[float, float] = (0.2, 1.0),
    seed: int = 42,
) -> Oracle:
    """Build a reward oracle whose average reward is `target_ctr`.

    Kernels are placed uniformly in `[center - delta, center + delta]^d`.
    Amplitudes are then scaled by `target_ctr / V_mc`, where `V_mc` is the mean
    reward over all users paired with a random ad. The scaled amplitudes are
    clamped at the top of `alpha_range`, so the realized mean undershoots the
    target whenever that clamp binds.
    """
    if center is None:
        center_u = torch.zeros(users.shape[1])
        center_a = torch.zeros(ads.shape[1])
    else:
        center_u, center_a = center

    torch.manual_seed(seed)
    center_u = torch.as_tensor(center_u, dtype=torch.float32)
    center_a = torch.as_tensor(center_a, dtype=torch.float32)

    kernels = draw_kernels(
        n_kernels, center_u, center_a, delta, sigma_range, alpha_range
    )
    alphas = kernels["alphas"]

    # Pair every user with a uniformly drawn ad; this randint consumes the
    # generator after the five kernel draws.
    with torch.no_grad():
        mc_u = users
        mc_a = ads[torch.randint(0, ads.shape[0], (users.shape[0],))]
        v_mc_before = score_pairs(kernels, mc_u, mc_a, alphas).mean().item()

    if v_mc_before > 1e-10:
        alphas = (alphas * (target_ctr / v_mc_before)).clamp(max=alpha_range[1])

    with torch.no_grad():
        v_final = score_pairs(kernels, mc_u, mc_a, alphas).mean().item()

    logger.debug(
        "oracle: K=%d delta=%.3f sigma=%.3f alpha=%.4f V_mc %.4f -> %.4f (target %.4f)",
        n_kernels,
        delta,
        kernels["sigma_u"].mean().item(),
        alphas.mean().item(),
        v_mc_before,
        v_final,
        target_ctr,
    )

    return Oracle(
        mu_u=kernels["mu_u"],
        mu_a=kernels["mu_a"],
        sigma_u=kernels["sigma_u"],
        sigma_a=kernels["sigma_a"],
        alphas=alphas,
        K=n_kernels,
        V_cf=v_final,
        center_u=center_u,
        center_a=center_a,
        delta=delta,
    )
