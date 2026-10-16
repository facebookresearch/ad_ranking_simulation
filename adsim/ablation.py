# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Degrading the domain classifier on purpose, to see what breaks.

The whole method rests on the classifier: it supplies the density ratios, and
its logging-class probability supplies the cold-start diagnostic. So the
question to ask is what happens when that classifier is bad, and whether the
diagnostics would tell you.

Six conditions run against an otherwise identical trial:

- `clean`          classifier fitted on the reward model's fold, in-sample
                   probabilities
- `crossfit`       out-of-fold probabilities, so weights never see their own row
- `uncal`          temperature discarded, so probabilities are uncalibrated
- `mask33/67/100`  progressively zero the ad features, driving the impression
                   representation toward policy-agnostic. At `mask100` no ad
                   information survives and the policies cannot be separated
                   even in principle.

The unweighted Direct Method is fitted once per overlap level and shared across
all six conditions, because it never touches the classifier. It is the control:
movement in its numbers would mean the comparison is contaminated.

Each record pairs downstream estimator error with classifier-quality
measurements and with oracle quantities the simulator can compute but
production cannot, so degradation is traced from cause to effect rather than
merely observed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .calibration import compute_pairwise_density_weights
from .config import TrialConfig
from .dc_diagnostics import (
    candidate_confusion,
    logging_class_ece,
    mask_ad_features,
    multiclass_nll,
    spearman,
    true_cold_start_mass,
)
from .density_ratio import (
    compute_calibrated_probs,
    compute_importance_weights,
    compute_per_domain_density_ratios,
    compute_weight_diagnostics,
)
from .domain_classifier import (
    fit_temperature,
    get_kfold_oof_logits,
    train_calibrated_domain_classifier,
)
from .estimators import dr_values, ips_values
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

# (name, ad columns zeroed, probability source)
DC_CONDITIONS: tuple[tuple[str, int, str], ...] = (
    ("clean", 0, "insample"),
    ("crossfit", 0, "oof"),
    ("uncal", 0, "notemp"),
    ("mask33", 1, "insample"),
    ("mask67", 2, "insample"),
    ("mask100", 3, "insample"),
)

# Estimators reported per record. `u` and `w` are the unweighted and
# mixture-weighted Direct Method, `pp` the per-policy weighted baseline.
ABLATION_METHODS: tuple[str, ...] = (
    "u",
    "w",
    "pp",
    "ips_plain",
    "dru_plain",
    "drw_plain",
)

# Impressions per class used for classifier-quality measurement and for the
# oracle cold-start comparison.
DC_EVAL_N = 3000
COLD_START_TAU = 0.1


@dataclass(frozen=True)
class OverlapContext:
    """Everything shared by the six conditions at one overlap level.

    Computed once so the conditions differ only in the classifier.
    """

    names: list[str]
    policy_to_class: dict[str, int]
    target_classes: list[int]
    policy_embeddings: dict[str, torch.Tensor]
    train_features: torch.Tensor
    train_labels: torch.Tensor
    cal_features: torch.Tensor
    cal_labels: torch.Tensor
    log_features: torch.Tensor
    target_features: dict[str, torch.Tensor]
    eval_rewards: torch.Tensor
    reward_mean: float
    v_true: dict[str, float]
    pred_log_unweighted: torch.Tensor
    pred_unweighted: dict[str, torch.Tensor]
    oracle_ratios: dict[str, np.ndarray]
    true_masses: dict[str, float]
    holdout: tuple[torch.Tensor, np.ndarray]
    train_eval: tuple[torch.Tensor, np.ndarray]
    n_eval: int


def _multiclass_eval_set(
    users: torch.Tensor,
    log_ads: torch.Tensor,
    served: dict[str, torch.Tensor],
    names: Sequence[str],
    policy_to_class: dict[str, int],
    limit: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Stack one block of impressions per policy, labelled by policy."""
    features = [torch.cat([users[:limit], log_ads[:limit]], dim=1)]
    labels = [np.zeros(limit)]
    for name in names:
        features.append(torch.cat([users[:limit], served[name][:limit]], dim=1))
        labels.append(np.full(limit, policy_to_class[name]))
    return torch.cat(features, dim=0), np.concatenate(labels)


def _condition_probs(
    mode: str,
    classifier: Any,
    temperature: float,
    masked_train_features: torch.Tensor,
    masked_embeddings: dict[str, torch.Tensor],
    policy_to_class: dict[str, int],
    config: TrialConfig,
) -> torch.Tensor:
    """Training-fold probabilities for one condition.

    The out-of-fold branch refits cross-validation from scratch and keeps only
    the logging rows, so the weights are built from probabilities that never saw
    their own row.
    """
    if mode != "oof":
        return compute_calibrated_probs(classifier, masked_train_features, temperature)

    oof_logits, oof_labels = get_kfold_oof_logits(
        masked_embeddings,
        policy_to_class,
        classifier_type="mlp",
        hidden_dims=[64],
        n_folds=config.dc_folds,
        epochs=config.dc_epochs,
        batch_size=512,
        lr=config.dc_lr,
        verbose=False,
    )
    oof_temperature = fit_temperature(oof_logits, oof_labels, verbose=False)
    logging_rows = torch.from_numpy(np.asarray(oof_logits, dtype=np.float32))[
        np.asarray(oof_labels) == 0
    ]
    return torch.softmax(logging_rows / oof_temperature, dim=1)


def _fit_per_policy_models(
    config: TrialConfig,
    context: OverlapContext,
    train_probs: torch.Tensor,
    device: torch.device | None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """One reward model per candidate, weighted by that candidate's own ratio.

    The ratios come from the same training-fold probabilities as the mixture
    weight. Training runs under a forked random state, so the other estimators
    draw the same random numbers with or without this baseline.
    """
    per_policy_weights = torch.as_tensor(
        np.asarray(
            compute_per_domain_density_ratios(
                train_probs,
                logging_class=0,
                target_classes=context.target_classes,
                clip=config.clip,
            )
        )
    ).float()
    models: dict[str, Any] = {}
    temperatures: dict[str, float] = {}
    with torch.random.fork_rng(devices=[]):
        for index, name in enumerate(context.names):
            models[name] = train_reward_model(
                MLPDMRewardModel(
                    d_in=config.feature_dim, hidden_dims=list(config.hidden)
                ),
                context.train_features,
                context.train_labels,
                weights=per_policy_weights[:, index],
                epochs=config.epochs,
                device=device,
            )
            temperatures[name] = (
                calibrate_temperature(
                    models[name], context.cal_features, context.cal_labels
                )
                if config.calibrate
                else 1.0
            )
    return models, temperatures


def _run_condition(
    config: TrialConfig,
    context: OverlapContext,
    condition: str,
    n_masked: int,
    mode: str,
    device: torch.device | None,
) -> dict[str, Any]:
    """Fit the classifier under one condition and score everything downstream."""
    names = context.names
    masked_embeddings = {
        k: mask_ad_features(v, n_masked) for k, v in context.policy_embeddings.items()
    }
    classifier, fitted_temperature, _ = train_calibrated_domain_classifier(
        policy_embeddings=masked_embeddings,
        policy_to_class=context.policy_to_class,
        classifier_type="mlp",
        hidden_dims=[64],
        n_folds=config.dc_folds,
        epochs=config.dc_epochs,
        batch_size=512,
        lr=config.dc_lr,
        verbose=False,
    )
    temperature = 1.0 if mode == "notemp" else fitted_temperature

    train_probs = _condition_probs(
        mode,
        classifier,
        temperature,
        mask_ad_features(context.train_features, n_masked),
        masked_embeddings,
        context.policy_to_class,
        config,
    )
    mixture_weights = torch.from_numpy(
        compute_importance_weights(
            train_probs,
            logging_class=0,
            num_target_policies=len(names),
            clip=config.clip,
        )
    ).float()
    diagnostics = compute_weight_diagnostics(mixture_weights.numpy())

    weighted = train_reward_model(
        MLPDMRewardModel(d_in=config.feature_dim, hidden_dims=list(config.hidden)),
        context.train_features,
        context.train_labels,
        weights=mixture_weights,
        epochs=config.epochs,
        device=device,
    )
    temp_weighted = (
        calibrate_temperature(weighted, context.cal_features, context.cal_labels)
        if config.calibrate
        else 1.0
    )
    per_policy, per_policy_temps = _fit_per_policy_models(
        config, context, train_probs, device
    )
    pred_log_weighted = predict_calibrated(
        weighted, context.log_features, temp_weighted
    )
    eval_probs = compute_calibrated_probs(
        classifier, mask_ad_features(context.log_features, n_masked), temperature
    )
    eval_weights = torch.as_tensor(
        np.asarray(
            compute_per_domain_density_ratios(
                eval_probs,
                logging_class=0,
                target_classes=context.target_classes,
                clip=config.clip,
            )
        )
    ).float()

    v_hat: dict[str, dict[str, float]] = {m: {} for m in ABLATION_METHODS}
    per_policy_log: dict[str, float] = {}
    estimated_masses: list[float] = []
    subset_masses: list[float] = []
    ratio_correlations: list[float] = []

    for index, name in enumerate(names):
        features = context.target_features[name]
        weights = eval_weights[:, index]
        pred_weighted = predict_calibrated(weighted, features, temp_weighted)

        v_hat["u"][name] = context.pred_unweighted[name].mean().item()
        v_hat["w"][name] = pred_weighted.mean().item()
        v_hat["ips_plain"][name] = ips_values(context.eval_rewards, weights)[0]
        v_hat["dru_plain"][name] = dr_values(
            context.eval_rewards,
            weights,
            context.pred_log_unweighted,
            context.pred_unweighted[name],
        )[0]
        v_hat["drw_plain"][name] = dr_values(
            context.eval_rewards, weights, pred_log_weighted, pred_weighted
        )[0]
        v_hat["pp"][name] = (
            predict_calibrated(per_policy[name], features, per_policy_temps[name])
            .mean()
            .item()
        )
        per_policy_log[name] = (
            predict_calibrated(
                per_policy[name], context.log_features, per_policy_temps[name]
            )
            .mean()
            .item()
        )

        q_logging = np.asarray(
            compute_calibrated_probs(
                classifier, mask_ad_features(features, n_masked), temperature
            )
        )[:, 0]
        estimated_masses.append(float((q_logging < COLD_START_TAU).mean()))
        subset_masses.append(
            float((q_logging[: context.n_eval] < COLD_START_TAU).mean())
        )
        ratio_correlations.append(
            spearman(weights.numpy(), context.oracle_ratios[name])
        )

    baselines: dict[str, float | dict[str, float]] = {
        "u": context.pred_log_unweighted.mean().item(),
        "w": pred_log_weighted.mean().item(),
        "pp": per_policy_log,
    }
    for method in ABLATION_METHODS:
        baselines.setdefault(method, context.reward_mean)

    record: dict[str, Any] = {
        "dc_condition": condition,
        "n_ad_masked": n_masked,
        "mode": mode,
        "avg_ess": diagnostics["ess_ratio"],
        "n_policies": len(names),
    }
    for method in ABLATION_METHODS:
        decision = compute_decision_metrics(
            {n: context.v_true[n] for n in names},
            v_hat[method],
            context.v_true["logging"],
            baselines[method],
            names,
        )
        record[f"acc1_{method}"] = decision["acc1"]
        record[f"regret_{method}"] = decision["regret"]
        record[f"mae_v_{method}"] = decision["mae_v"]

    record.update(_classifier_quality(classifier, context, n_masked, temperature))
    record["ratio_spearman"] = float(np.nanmean(ratio_correlations))
    record["names"] = list(names)
    record["Vtrue"] = [context.v_true[n] for n in names]
    for method in ABLATION_METHODS:
        record[f"Vhat_{method}"] = [v_hat[method][n] for n in names]
    record["cs10"] = estimated_masses
    record["mu_hat10"] = subset_masses
    record["mu_star10"] = [context.true_masses[n] for n in names]
    record["cs_err10"] = [
        abs(a - b) for a, b in zip(subset_masses, record["mu_star10"])
    ]
    return record


def _classifier_quality(
    classifier: Any,
    context: OverlapContext,
    n_masked: int,
    temperature: float,
) -> dict[str, float]:
    """Held-out quality of the classifier itself, independent of any estimator."""
    holdout_x, holdout_y = context.holdout
    train_x, train_y = context.train_eval
    holdout_probs = np.asarray(
        compute_calibrated_probs(
            classifier, mask_ad_features(holdout_x, n_masked), temperature
        )
    )
    train_probs = np.asarray(
        compute_calibrated_probs(
            classifier, mask_ad_features(train_x, n_masked), temperature
        )
    )
    nll_holdout = multiclass_nll(holdout_probs, holdout_y)
    nll_train = multiclass_nll(train_probs, train_y)
    candidate_acc, candidate_offdiag = candidate_confusion(holdout_probs, holdout_y)
    return {
        "dc_nll": nll_holdout,
        "dc_nll_train": nll_train,
        "dc_nll_gap": nll_holdout - nll_train,
        "dc_ece_logging": logging_class_ece(holdout_probs, holdout_y),
        "dc_cand_acc": candidate_acc,
        "dc_cand_offdiag": candidate_offdiag,
        "dc_acc": float((holdout_probs.argmax(1) == holdout_y.astype(int)).mean()),
    }


def _build_context(
    config: TrialConfig,
    setup: Any,
    names: list[str],
    served: dict[str, torch.Tensor],
    policies: dict[str, Any],
    dc_eval_n: int,
    device: torch.device | None,
) -> OverlapContext:
    """Fit the shared control model and assemble everything condition-invariant."""
    logged = setup.logged
    split = setup.split
    train_idx, cal_idx, eval_idx = (
        split["train"],
        split["calibration"],
        split["evaluation"],
    )
    train_users = setup.sampled_users[train_idx]
    train_log_ads = logged["served_ads"][train_idx]
    eval_users = setup.sampled_users[eval_idx]
    eval_log_ads = logged["served_ads"][eval_idx]
    eval_served = {n: served[n][eval_idx] for n in names}
    train_served = {n: served[n][train_idx] for n in names}

    policy_to_class = {"logging": 0}
    for index, name in enumerate(names):
        policy_to_class[name] = index + 1

    train_features = logged["features"][train_idx]
    train_labels = logged["rewards"][train_idx]
    log_features = torch.cat([eval_users, eval_log_ads], dim=1)

    unweighted = train_reward_model(
        MLPDMRewardModel(d_in=config.feature_dim, hidden_dims=list(config.hidden)),
        train_features,
        train_labels,
        epochs=config.epochs,
        device=device,
    )
    cal_features = logged["features"][cal_idx]
    cal_labels = logged["rewards"][cal_idx]
    temp_unweighted = (
        calibrate_temperature(unweighted, cal_features, cal_labels)
        if config.calibrate
        else 1.0
    )
    pred_log_unweighted = predict_calibrated(unweighted, log_features, temp_unweighted)

    v_true = {"logging": logged["v_log"]}
    target_features: dict[str, torch.Tensor] = {}
    pred_unweighted: dict[str, torch.Tensor] = {}
    for name in names:
        features = torch.cat([eval_users, eval_served[name]], dim=1)
        target_features[name] = features
        with torch.no_grad():
            true_reward = clamp_reward(
                score_pairs(setup.oracle, eval_users, eval_served[name])
            )
        v_true[name] = true_reward.mean().item()
        pred_unweighted[name] = predict_calibrated(
            unweighted, features, temp_unweighted
        )

    n_eval = min(dc_eval_n, len(eval_idx))
    holdout = _multiclass_eval_set(
        eval_users, eval_log_ads, eval_served, names, policy_to_class, n_eval
    )
    n_train = min(dc_eval_n, train_users.shape[0])
    train_eval = _multiclass_eval_set(
        train_users, train_log_ads, train_served, names, policy_to_class, n_train
    )

    policy_embeddings = {"logging": torch.cat([train_users, train_log_ads], dim=1)}
    for name in names:
        policy_embeddings[name] = torch.cat([train_users, train_served[name]], dim=1)

    oracle_ratios: dict[str, np.ndarray] = {}
    for name in names:
        _, raw, _ = compute_pairwise_density_weights(
            setup.log_policy,
            policies[name],
            eval_users,
            eval_log_ads,
            eval_users,
            eval_served[name],
            clip=config.clip,
        )
        oracle_ratios[name] = np.asarray(raw)

    true_masses = {
        name: true_cold_start_mass(
            setup.log_policy,
            policies,
            eval_users,
            eval_log_ads,
            eval_served,
            name,
            n_eval=n_eval,
            tau=COLD_START_TAU,
        )
        for name in names
    }

    return OverlapContext(
        names=names,
        policy_to_class=policy_to_class,
        target_classes=[policy_to_class[n] for n in names],
        policy_embeddings=policy_embeddings,
        train_features=train_features,
        train_labels=train_labels,
        cal_features=cal_features,
        cal_labels=cal_labels,
        log_features=log_features,
        target_features=target_features,
        eval_rewards=logged["rewards"][eval_idx],
        reward_mean=logged["rewards"][eval_idx].mean().item(),
        v_true=v_true,
        pred_log_unweighted=pred_log_unweighted,
        pred_unweighted=pred_unweighted,
        oracle_ratios=oracle_ratios,
        true_masses=true_masses,
        holdout=holdout,
        train_eval=train_eval,
        n_eval=n_eval,
    )


def run_dc_ablation(
    config: TrialConfig,
    users: torch.Tensor,
    ads: torch.Tensor,
    seed: int,
    target_seed: int | None = None,
    calib_results: list[Any] | None = None,
    device: torch.device | None = None,
    dc_eval_n: int = DC_EVAL_N,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """One seed, every ESS target, every classifier condition.

    Emits one record per (ESS target, condition).
    """
    setup = prepare_trial(config, users, ads, seed, target_seed, calib_results)
    records: list[dict[str, Any]] = []

    for ess_target in config.ess_targets:
        names, served, policies = candidates_at_overlap(config, setup, ess_target)
        if not names:
            continue

        context = _build_context(
            config, setup, names, served, policies, dc_eval_n, device
        )
        for condition, n_masked, mode in DC_CONDITIONS:
            record = _run_condition(config, context, condition, n_masked, mode, device)
            record.update(
                {
                    "seed": seed,
                    "ess_tgt": ess_target,
                    "V_log": setup.logged["v_log"],
                }
            )
            records.append(record)

        if verbose:
            logger.info("ess_tgt=%.2f: %d conditions", ess_target, len(DC_CONDITIONS))

    return records
