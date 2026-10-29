# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Reward model for the Direct Method estimator.

An MLP maps an impression feature `z = [user, ad]` to a click probability and
is fitted on the logged rewards. Replaying a candidate policy through it gives
that policy's estimated value without ever needing a logged propensity.

Sample weights are what distinguishes the estimators built on this model. Train
with uniform weights and the fit targets the logging distribution; train with
mixture density-ratio weights and it targets the candidate mixture; train with
one candidate's ratios and it targets that candidate alone.

Logits are clamped to `+/- LOGIT_BOUND` before the loss, which bounds the
predicted probability into `[eps, 1 - eps]` and therefore bounds the
per-example loss by `-log(eps)`. The value-error bound assumes a bounded loss,
so this clamp is what makes the reported theory apply to the trained model
rather than being an unenforced assumption.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Reward probabilities are confined to [EPS_REWARD, 1 - EPS_REWARD]; the
# corresponding logit bound is ~13.8155.
EPS_REWARD = 1e-6
LOGIT_BOUND: float = math.log((1 - EPS_REWARD) / EPS_REWARD)

DEFAULT_HIDDEN_DIMS: list[int] = [256, 128]


class MLPDMRewardModel(nn.Module):
    """MLP predicting a click logit from an impression feature.

    `hidden_dims` is a full architecture spec; its last entry is the
    penultimate width exposed by `get_penultimate_features`.
    """

    d_in: int
    hidden_dims: list[int]
    dropout_p: float
    hidden_layers: nn.Sequential
    fc_out: nn.Linear

    def __init__(
        self,
        d_in: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        dims: list[int] = (
            hidden_dims if hidden_dims is not None else list(DEFAULT_HIDDEN_DIMS)
        )
        self.d_in = d_in
        self.hidden_dims = dims
        self.dropout_p = dropout

        layers: list[nn.Module] = []
        prev_dim = d_in
        for hidden_dim in dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_dim, 1)

    @property
    def embedding_dim(self) -> int:
        """Width of the penultimate layer, or `d_in` when there are no hidden layers."""
        if len(self.hidden_dims) == 0:
            return self.d_in
        return self.hidden_dims[-1]

    def get_penultimate_features(self, x: torch.Tensor) -> torch.Tensor:
        """Activations before the output layer, for downstream uncertainty models."""
        if len(self.hidden_dims) == 0:
            return x
        return self.hidden_layers(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-sigmoid logits, shape `[N]`."""
        return self.fc_out(self.get_penultimate_features(x)).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Uncalibrated click probabilities. Prefer `predict_calibrated`."""
        return torch.sigmoid(self.forward(x))


def train_reward_model(
    model: MLPDMRewardModel,
    features: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> MLPDMRewardModel:
    """Fit the reward model with clamped, optionally weighted, BCE.

    `weights` are renormalized to sum to `N`, so weighting changes which
    impressions the fit prioritizes without changing the overall loss scale.
    Passing `None` gives the unweighted model.

    Labels may be soft (oracle probabilities) or hard. Returns the model on CPU
    in eval mode with gradients disabled; training mutates and returns the same
    object.

    Batch order comes from a shuffling `DataLoader`, so this consumes the global
    torch generator.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()
    # The single-tensor path is pinned rather than left to torch: the
    # multi-tensor implementation is selected by device and can accumulate in a
    # different order.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay, foreach=False
    )

    n_samples = len(features)
    if weights is not None:
        normalized = weights / weights.sum() * n_samples
        dataset = TensorDataset(features, labels, normalized)
    else:
        dataset = TensorDataset(features, labels)
    # Pinned memory only buys faster host-to-device copies, and these runs are
    # CPU-only.
    # @lint-ignore CITRINE CITRINECI
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _epoch in range(epochs):
        for batch in dataloader:
            batch_x = batch[0].to(device)
            batch_y = batch[1].to(device)
            optimizer.zero_grad()
            logits = model(batch_x).clamp(-LOGIT_BOUND, LOGIT_BOUND)
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, batch_y, reduction="none"
            )
            if weights is not None:
                loss = (bce * batch[2].to(device)).mean()
            else:
                loss = bce.mean()
            loss.backward()
            optimizer.step()

    model.eval()
    model.to(torch.device("cpu"))
    for param in model.parameters():
        param.requires_grad = False
    return model


def calibrate_temperature(
    model: MLPDMRewardModel,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Fit a scalar temperature on held-out impressions by LBFGS.

    A reward model fitted under sample weights is systematically miscalibrated
    on the logging distribution, so every model is temperature-corrected on the
    calibration fold before its predictions are averaged into a value.
    """
    model.eval()
    with torch.no_grad():
        logits = model(features)

    log_temperature = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.01, max_iter=20)
    targets = labels.float()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled = (logits / torch.exp(log_temperature)).clamp(-LOGIT_BOUND, LOGIT_BOUND)
        loss = nn.functional.binary_cross_entropy_with_logits(scaled, targets)
        loss.backward()
        return loss

    for _step in range(25):
        optimizer.step(closure)

    return torch.exp(log_temperature).item()


def predict_calibrated(
    model: MLPDMRewardModel,
    features: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Temperature-corrected click probabilities, bounded away from 0 and 1."""
    model.eval()
    with torch.no_grad():
        scaled = (model(features) / temperature).clamp(-LOGIT_BOUND, LOGIT_BOUND)
        return torch.sigmoid(scaled)
