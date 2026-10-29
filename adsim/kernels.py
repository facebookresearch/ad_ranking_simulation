# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Factored-kernel scoring shared by the reward oracle and candidate policies.

A kernel model scores a (user, ad) pair as a sum of `K` separable Gaussian
kernels,

    s(u, a) = sum_k alpha_k
              * exp(-||u - mu_u_k||^2 / (2 sigma_u_k^2))
              * exp(-||a - mu_a_k||^2 / (2 sigma_a_k^2)).

The oracle and every candidate policy are drawn from this same family. That is
what makes a candidate placeable: moving its kernel center relative to the
oracle's trades off how much value it captures against how much its induced
impression distribution overlaps the logging one.
"""

from __future__ import annotations

from typing import TypedDict

import torch


class KernelParams(TypedDict):
    """The tensors needed to score a pair. Everything scorable satisfies this."""

    mu_u: torch.Tensor
    mu_a: torch.Tensor
    sigma_u: torch.Tensor
    sigma_a: torch.Tensor
    alphas: torch.Tensor


class KernelModel(KernelParams):
    """Scorable parameters plus the geometry they were drawn from.

    `center_u`, `center_a`, and `delta` record the region the kernels came
    from, which is what lets a candidate be positioned relative to the oracle.
    """

    K: int
    center_u: torch.Tensor
    center_a: torch.Tensor
    delta: float


def factored_kernel_eval(
    users: torch.Tensor,
    ads: torch.Tensor,
    mu_u: torch.Tensor,
    mu_a: torch.Tensor,
    sigma_u: torch.Tensor,
    sigma_a: torch.Tensor,
    alphas: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a factored kernel on paired (user, ad) samples.

    `users` and `ads` are row-aligned: row `i` is one pair. Returns one score
    per row.
    """
    du_sq = ((users.unsqueeze(1) - mu_u.unsqueeze(0)) ** 2).sum(dim=2)
    da_sq = ((ads.unsqueeze(1) - mu_a.unsqueeze(0)) ** 2).sum(dim=2)
    ku = torch.exp(-du_sq / (2 * sigma_u.unsqueeze(0) ** 2))
    ka = torch.exp(-da_sq / (2 * sigma_a.unsqueeze(0) ** 2))
    return (alphas.unsqueeze(0) * ku * ka).sum(dim=1)


def score_pairs(
    model: KernelParams,
    users: torch.Tensor,
    ads: torch.Tensor,
    alphas: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score row-aligned pairs under a kernel model.

    `alphas` overrides the model's own amplitudes, which the oracle needs while
    it is solving for the rescaling factor.
    """
    return factored_kernel_eval(
        users,
        ads,
        model["mu_u"],
        model["mu_a"],
        model["sigma_u"],
        model["sigma_a"],
        model["alphas"] if alphas is None else alphas,
    )


def draw_kernels(
    n_kernels: int,
    center_u: torch.Tensor,
    center_a: torch.Tensor,
    delta: float,
    sigma_range: tuple[float, float],
    alpha_range: tuple[float, float],
) -> KernelParams:
    """Draw kernel centers, widths, and amplitudes from the seeded generator.

    Callers seed the global generator first. The five draws below consume it in
    a fixed order, and the oracle and policies rely on that order being the
    same for both, so do not reorder them.
    """
    lo_s, hi_s = sigma_range
    lo_a, hi_a = alpha_range
    user_dim = center_u.shape[0]
    ad_dim = center_a.shape[0]
    return KernelParams(
        mu_u=center_u.unsqueeze(0) + (torch.rand(n_kernels, user_dim) * 2 - 1) * delta,
        mu_a=center_a.unsqueeze(0) + (torch.rand(n_kernels, ad_dim) * 2 - 1) * delta,
        sigma_u=torch.rand(n_kernels) * (hi_s - lo_s) + lo_s,
        sigma_a=torch.rand(n_kernels) * (hi_s - lo_s) + lo_s,
        alphas=torch.rand(n_kernels) * (hi_a - lo_a) + lo_a,
    )


def top1_from_slate(
    model: KernelParams,
    users: torch.Tensor,
    candidate_ads: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Return the ad each user's slate ranks first under `model`.

    `candidate_ads` is `[N, top_k, ad_dim]`. Selection is a hard argmax, which
    is why the logged policy has no action propensity to record.
    """
    n = users.shape[0]
    user_dim = users.shape[-1]
    ad_dim = candidate_ads.shape[-1]
    u_exp = users.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, user_dim)
    a_flat = candidate_ads.reshape(-1, ad_dim)
    with torch.no_grad():
        scores = score_pairs(model, u_exp, a_flat).reshape(n, top_k)
    return candidate_ads[torch.arange(n), scores.argmax(dim=1)]
