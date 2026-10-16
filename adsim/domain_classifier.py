# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Domain classifier: which policy would have produced this impression.

Density ratios between the logging distribution and a candidate's are not
observable, but they are recoverable from a classifier. Train a multiclass
model to predict which policy generated an impression, and the ratio of its
class probabilities estimates the ratio of the underlying densities. This is
what replaces logged propensities.

Training runs in three steps, because the classifier's probabilities are used
as densities and therefore have to be calibrated, not merely accurate:

1. K-fold cross-fitting produces out-of-fold logits, so no impression
   contributes to the temperature fitted on it.
2. A single temperature is fitted on those out-of-fold logits by minimizing
   multiclass NLL.
3. A final classifier is trained on all the data and paired with that
   temperature.

Turning the calibrated probabilities into weights is `adsim.density_ratio`.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_HIDDEN_DIMS: list[int] = [64]


class CalibrationInfo(TypedDict):
    """Diagnostics from the calibration pipeline."""

    n_folds: int
    temperature: float
    final_accuracy: float
    n_samples: int
    classifier_type: str


class LinearDomainClassifier(nn.Module):
    """Single-layer classifier, a baseline for the MLP."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)


class MLPDomainClassifier(nn.Module):
    """MLP over the impression feature, one logit per policy."""

    input_dim: int
    num_classes: int
    hidden_dims: list[int]
    hidden_layers: nn.Sequential
    output_layer: nn.Linear

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        dims: list[int] = (
            hidden_dims if hidden_dims is not None else list(DEFAULT_HIDDEN_DIMS)
        )
        if len(dims) == 0:
            raise ValueError("hidden_dims must have at least one element")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = dims

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.hidden_layers(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)


def _build_classifier(
    classifier_type: str,
    input_dim: int,
    num_classes: int,
    hidden_dims: list[int] | None,
) -> nn.Module:
    if classifier_type == "mlp":
        return MLPDomainClassifier(input_dim, num_classes, hidden_dims=hidden_dims)
    return LinearDomainClassifier(input_dim, num_classes)


def train_domain_classifier_on_subset(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    classifier_type: str = "mlp",
    hidden_dims: list[int] | None = None,
    epochs: int = 4,
    batch_size: int = 512,
    lr: float = 0.01,
    weight_decay: float = 0.01,
    device: torch.device | None = None,
) -> nn.Module:
    """Train one classifier on one subset of impressions.

    The loss is class-weighted by inverse frequency, because a candidate that
    rarely wins the slate would otherwise be ignored. Returns the model on CPU
    in eval mode with gradients disabled.

    Batch order comes from a shuffling `DataLoader`, so this consumes the
    global torch generator.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = _build_classifier(
        classifier_type, embeddings.shape[1], num_classes, hidden_dims
    )
    classifier.to(device)
    classifier.train()

    class_counts = torch.bincount(labels, minlength=num_classes)
    class_weights = torch.reciprocal(class_counts.float() + 1e-6)
    class_weights = (class_weights / class_weights.sum() * num_classes).to(device)

    # The single-tensor path is pinned rather than left to torch: the
    # multi-tensor implementation is selected by device and can accumulate in a
    # different order.
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=lr, weight_decay=weight_decay, foreach=False
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    dataset = torch.utils.data.TensorDataset(embeddings, labels)
    # Pinned memory only buys faster host-to-device copies, and these runs are
    # CPU-only.
    # @lint-ignore CITRINE CITRINECI
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    for _epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            loss = criterion(classifier(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()

    classifier.eval()
    classifier.to(torch.device("cpu"))
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier


def _stack_policies(
    policy_embeddings: dict[str, torch.Tensor],
    policy_to_class: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten per-policy impressions into one labeled training set."""
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for policy_name, emb in policy_embeddings.items():
        embeddings.append(emb)
        labels.append(
            torch.full((len(emb),), policy_to_class[policy_name], dtype=torch.long)
        )
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def get_kfold_oof_logits(
    policy_embeddings: dict[str, torch.Tensor],
    policy_to_class: dict[str, int],
    classifier_type: str = "mlp",
    hidden_dims: list[int] | None = None,
    n_folds: int = 5,
    epochs: int = 4,
    batch_size: int = 512,
    lr: float = 0.01,
    weight_decay: float = 0.01,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold logits for every impression.

    Each fold's logits come from a classifier that never saw that fold, so the
    temperature fitted on them is not fitted on its own training data.

    The fold assignment is drawn from the global numpy generator, so a trial
    must seed numpy to be reproducible.
    """
    all_embeddings, all_labels = _stack_policies(policy_embeddings, policy_to_class)
    n_samples = len(all_labels)
    n_classes = len(policy_to_class)

    oof_logits = np.zeros((n_samples, n_classes), dtype=np.float32)
    oof_labels = all_labels.numpy()

    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    kfold = KFold(n_splits=n_folds, shuffle=False)  # indices are already shuffled

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        train_orig_idx = indices[train_idx]
        val_orig_idx = indices[val_idx]

        fold_classifier = train_domain_classifier_on_subset(
            embeddings=all_embeddings[train_orig_idx],
            labels=all_labels[train_orig_idx],
            num_classes=n_classes,
            classifier_type=classifier_type,
            hidden_dims=hidden_dims,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )
        with torch.no_grad():
            val_logits = fold_classifier(all_embeddings[val_orig_idx]).numpy()
        oof_logits[val_orig_idx] = val_logits

        if verbose:
            fold_acc = (
                np.argmax(val_logits, axis=1) == oof_labels[val_orig_idx]
            ).mean()
            logger.info(f"  Fold {fold_idx + 1}/{n_folds}: val_acc={fold_acc:.3f}")

    return oof_logits, oof_labels


def fit_temperature(
    oof_logits: np.ndarray,
    oof_labels: np.ndarray,
    n_steps: int = 500,
    lr: float = 0.01,
    verbose: bool = True,
) -> float:
    """Fit one temperature minimizing NLL of the out-of-fold logits.

    Optimizes `t` with `T = exp(t)` to keep the temperature positive, and keeps
    the best iterate rather than the last.
    """
    logits = torch.tensor(oof_logits, dtype=torch.float32)
    labels = torch.tensor(oof_labels, dtype=torch.long)

    t = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.Adam([t], lr=lr, foreach=False)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_t = 0.0
    for _step in range(n_steps):
        optimizer.zero_grad()
        loss = criterion(logits / torch.exp(t), labels)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_t = t.item()

    temperature = float(np.exp(best_t))

    if verbose:
        with torch.no_grad():
            nll_before = criterion(logits, labels).item()
            nll_after = criterion(logits / temperature, labels).item()
        logger.info(
            f"Temperature scaling: T={temperature:.4f}, "
            f"NLL {nll_before:.4f} -> {nll_after:.4f}"
        )

    return temperature


def train_calibrated_domain_classifier(
    policy_embeddings: dict[str, torch.Tensor],
    policy_to_class: dict[str, int],
    classifier_type: str = "mlp",
    hidden_dims: list[int] | None = None,
    n_folds: int = 5,
    epochs: int = 4,
    batch_size: int = 512,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    temp_lr: float = 0.01,
    temp_steps: int = 500,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, float, CalibrationInfo]:
    """Cross-fit, calibrate, then refit on everything.

    Returns the final classifier, its temperature, and diagnostics. Pass both
    to `adsim.density_ratio.compute_calibrated_probs`; the classifier's raw
    probabilities are not calibrated and must not be used as densities.
    """
    oof_logits, oof_labels = get_kfold_oof_logits(
        policy_embeddings=policy_embeddings,
        policy_to_class=policy_to_class,
        classifier_type=classifier_type,
        hidden_dims=hidden_dims,
        n_folds=n_folds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        verbose=verbose,
    )

    temperature = fit_temperature(
        oof_logits=oof_logits,
        oof_labels=oof_labels,
        n_steps=temp_steps,
        lr=temp_lr,
        verbose=verbose,
    )

    all_embeddings, all_labels = _stack_policies(policy_embeddings, policy_to_class)
    final_classifier = train_domain_classifier_on_subset(
        embeddings=all_embeddings,
        labels=all_labels,
        num_classes=len(policy_to_class),
        classifier_type=classifier_type,
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    with torch.no_grad():
        preds = torch.softmax(final_classifier(all_embeddings), dim=-1).argmax(dim=1)
        final_accuracy = (preds == all_labels).float().mean().item()

    info: CalibrationInfo = {
        "n_folds": n_folds,
        "temperature": temperature,
        "final_accuracy": final_accuracy,
        "n_samples": len(all_labels),
        "classifier_type": classifier_type,
    }
    if verbose:
        logger.info(
            f"Final classifier accuracy: {final_accuracy:.3f}, T={temperature:.4f}"
        )
    return final_classifier, temperature, info
