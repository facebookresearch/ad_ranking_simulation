# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""User and ad population for the synthetic simulator.

Users and ads are drawn uniformly from the hypercube [-1, 1]^d so that they
share support with the oracle kernel grid (see `adsim.oracle`).
"""

from __future__ import annotations

import torch


def make_world(
    num_users: int,
    num_ads: int,
    user_dim: int,
    ad_dim: int,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw the fixed user and ad populations a trial samples impressions from.

    Returns `(users, ads)` with shapes `[num_users, user_dim]` and
    `[num_ads, ad_dim]`.

    The two `torch.rand` calls consume the seeded global generator in this
    order; reordering them changes every downstream draw.
    """
    torch.manual_seed(seed)
    users = torch.rand(num_users, user_dim) * 2 - 1
    ads = torch.rand(num_ads, ad_dim) * 2 - 1
    return users, ads
