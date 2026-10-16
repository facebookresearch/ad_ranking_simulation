# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""How much to clip the importance weights.

Clipping is the one free parameter in the weighting scheme, and it is a
bias-variance dial rather than a detail. Left unclipped, an impression the
logging policy almost never served carries an unbounded weight and dominates
the objective. Clipped hard, the weights flatten toward uniform and the
mixture-weighted model converges on the unweighted one, giving up the
correction entirely.

This sweep holds everything else fixed and varies only the clip. Within one
overlap level the domain classifier and the unweighted reward model are each
fitted once and reused, so the only thing that differs between clip values is
the weighting itself; only the weighted reward model is refitted.

Records are binned on `kernel_ess`, the unclipped overlap computed from the
true policies, rather than on the classifier's own effective sample size. The
latter is clip-dependent, so binning on it would make the x-axis move with the
quantity being studied.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .calibration import kernel_mixture_ess
from .config import TrialConfig
from .density_ratio import (
    compute_calibrated_probs,
    compute_importance_weights,
    compute_weight_diagnostics,
)
from .domain_classifier import train_calibrated_domain_classifier
from .kernels import score_pairs
from .logging_sim import clamp_reward
from .metrics import compute_decision_metrics
from .reward_model import (
    calibrate_temperature,
    MLPDMRewardModel,
    predict_calibrated,
    train_reward_model,
)
from .trial import candidates_at_overlap, prepare_trial

logger: logging.Logger = logging.getLogger(__name__)


def run_clip_sweep(
    config: TrialConfig,
    users: torch.Tensor,
    ads: torch.Tensor,
    seed: int,
    target_seed: int | None = None,
    calib_results: list[Any] | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """One seed, every ESS target, every clip value.

    Emits one record per (ESS target, clip).
    """
    setup = prepare_trial(config, users, ads, seed, target_seed, calib_results)
    logged = setup.logged
    split = setup.split
    dc_idx, train_idx = split["dc"], split["train"]
    cal_idx, eval_idx = split["calibration"], split["evaluation"]

    train_features = logged["features"][train_idx]
    train_labels = logged["rewards"][train_idx]
    cal_features = logged["features"][cal_idx]
    cal_labels = logged["rewards"][cal_idx]
    dc_users = setup.sampled_users[dc_idx]
    dc_log_ads = logged["served_ads"][dc_idx]
    eval_users = setup.sampled_users[eval_idx]
    eval_log_ads = logged["served_ads"][eval_idx]
    log_features = torch.cat([eval_users, eval_log_ads], dim=1)

    records: list[dict[str, Any]] = []

    for ess_target in config.ess_targets:
        names, served, policies = candidates_at_overlap(config, setup, ess_target)
        if not names:
            continue

        overlap = kernel_mixture_ess(
            setup.log_policy,
            policies,
            names,
            dc_users,
            dc_log_ads,
            {n: served[n][dc_idx] for n in names},
        )

        # Clip does not affect classifier training, so fit it once.
        policy_embeddings = {"logging": torch.cat([dc_users, dc_log_ads], dim=1)}
        policy_to_class = {"logging": 0}
        for index, name in enumerate(names):
            policy_embeddings[name] = torch.cat([dc_users, served[name][dc_idx]], dim=1)
            policy_to_class[name] = index + 1
        classifier, dc_temperature, _ = train_calibrated_domain_classifier(
            policy_embeddings=policy_embeddings,
            policy_to_class=policy_to_class,
            classifier_type="mlp",
            hidden_dims=[64],
            n_folds=config.dc_folds,
            epochs=config.dc_epochs,
            batch_size=512,
            lr=config.dc_lr,
            verbose=False,
        )
        train_probs = compute_calibrated_probs(
            classifier, train_features, dc_temperature
        )

        # The unweighted model is clip-independent: fitted once, it is the
        # baseline every clip value is read against.
        unweighted = train_reward_model(
            MLPDMRewardModel(d_in=config.feature_dim, hidden_dims=list(config.hidden)),
            train_features,
            train_labels,
            epochs=config.epochs,
            device=device,
        )
        temp_unweighted = (
            calibrate_temperature(unweighted, cal_features, cal_labels)
            if config.calibrate
            else 1.0
        )

        v_true: dict[str, float] = {}
        v_hat_unweighted: dict[str, float] = {}
        target_features: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            logging_reward = score_pairs(setup.oracle, eval_users, eval_log_ads)
        v_true["logging"] = clamp_reward(logging_reward).mean().item()
        v_hat_unweighted["logging"] = (
            predict_calibrated(unweighted, log_features, temp_unweighted).mean().item()
        )
        for name in names:
            target_ads = served[name][eval_idx]
            features = torch.cat([eval_users, target_ads], dim=1)
            target_features[name] = features
            with torch.no_grad():
                reward = score_pairs(setup.oracle, eval_users, target_ads)
            v_true[name] = clamp_reward(reward).mean().item()
            v_hat_unweighted[name] = (
                predict_calibrated(unweighted, features, temp_unweighted).mean().item()
            )
        decision_unweighted = compute_decision_metrics(
            {n: v_true[n] for n in names},
            {n: v_hat_unweighted[n] for n in names},
            v_true["logging"],
            v_hat_unweighted["logging"],
            names,
        )

        for clip in config.clip_values:
            weights_np = compute_importance_weights(
                train_probs,
                logging_class=0,
                num_target_policies=len(names),
                clip=clip,
            )
            diagnostics = compute_weight_diagnostics(weights_np)
            weighted = train_reward_model(
                MLPDMRewardModel(
                    d_in=config.feature_dim, hidden_dims=list(config.hidden)
                ),
                train_features,
                train_labels,
                weights=torch.from_numpy(weights_np).float(),
                epochs=config.epochs,
                device=device,
            )
            temp_weighted = (
                calibrate_temperature(weighted, cal_features, cal_labels)
                if config.calibrate
                else 1.0
            )
            v_hat_weighted = {
                "logging": predict_calibrated(weighted, log_features, temp_weighted)
                .mean()
                .item()
            }
            for name in names:
                v_hat_weighted[name] = (
                    predict_calibrated(weighted, target_features[name], temp_weighted)
                    .mean()
                    .item()
                )
            decision_weighted = compute_decision_metrics(
                {n: v_true[n] for n in names},
                {n: v_hat_weighted[n] for n in names},
                v_true["logging"],
                v_hat_weighted["logging"],
                names,
            )

            records.append(
                {
                    "seed": seed,
                    "ess_tgt": ess_target,
                    "clip": clip,
                    "kernel_ess": overlap,
                    "dc_ess": diagnostics["ess_ratio"],
                    "V_log": logged["v_log"],
                    "n_policies": len(names),
                    "acc1_u": decision_unweighted["acc1"],
                    "regret_u": decision_unweighted["regret"],
                    "mae_v_u": decision_unweighted["mae_v"],
                    "acc1_w": decision_weighted["acc1"],
                    "regret_w": decision_weighted["regret"],
                    "mae_v_w": decision_weighted["mae_v"],
                }
            )

        if verbose:
            logger.info(
                "ess_tgt=%.2f kernel_ess=%.3f: %d clips",
                ess_target,
                overlap,
                len(config.clip_values),
            )

    return records
