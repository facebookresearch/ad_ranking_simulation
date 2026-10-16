# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""One trial: generate logged data, fit every estimator, score them all.

A trial fixes a seed and sweeps overlap. For each target effective sample size
it places candidate policies at that overlap level, simulates logging, fits the
domain classifier and the reward models, evaluates every estimator, and emits
one record. Sweeping ESS is what turns the experiment into a statement about
*when* an estimator can be trusted rather than a single average.

Ordering matters here more than anywhere else in the package. Several stages
reseed the global torch generator - `make_oracle`, `make_policy`, the
calibration sweep - and several others consume it, including every shuffling
`DataLoader`, so the order of the stages below fixes every random draw.
Reordering two stages that look independent will silently change every number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .calibration import (
    calibration_sweep,
    lookup_alpha_beta,
    orthogonal_directions,
    shift_axis,
    sweep_orthogonal_direction,
    SweepPoint,
)
from .config import TrialConfig
from .density_ratio import (
    compute_calibrated_probs,
    compute_importance_weights,
    compute_per_domain_density_ratios,
    compute_weight_diagnostics,
)
from .domain_classifier import train_calibrated_domain_classifier
from .estimators import dr_values, ips_values
from .kernels import score_pairs, top1_from_slate
from .logging_sim import (
    clamp_reward,
    collect_logged_data,
    crossfit_split_indices,
    DataSplit,
    IMPRESSION_SEED_OFFSET,
    LoggedData,
    LOGGING_CENTER_SEED_OFFSET,
    LOGGING_POLICY_SEED_OFFSET,
    make_logging_policy,
    sample_impressions,
    split_indices,
    SPLIT_SEED_OFFSET,
)
from .metrics import cold_start_masses, COLD_START_TAUS, compute_decision_metrics
from .oracle import make_oracle, Oracle
from .policy import make_policy, Policy
from .reward_model import (
    calibrate_temperature,
    MLPDMRewardModel,
    predict_calibrated,
    train_reward_model,
)

logger: logging.Logger = logging.getLogger(__name__)

# Estimators reported per record. `u` and `w` are the unweighted and
# mixture-weighted Direct Method, `pp` the per-policy weighted baseline.
# Weighted estimators appear in plain and self-normalized form.
METHODS: tuple[str, ...] = (
    "u",
    "w",
    "pp",
    "ips_plain",
    "ips_sn",
    "dru_plain",
    "dru_sn",
    "drw_plain",
    "drw_sn",
)

SWEEP_SEED_OFFSET = 500
SWEEP_DIRECTION_SEED_OFFSET = 999
CANDIDATE_SEED_OFFSET = 300


def _candidates_at_overlap(
    config: TrialConfig,
    sweep: list[SweepPoint],
    ess_target: float,
    oracle_center: torch.Tensor,
    v_along: torch.Tensor,
    distance: float,
    orth_dirs: list[torch.Tensor],
    target_seed: int,
    sampled_users: torch.Tensor,
    candidate_ads: torch.Tensor,
) -> tuple[list[str], dict[str, torch.Tensor], dict[str, Any]]:
    """Place one candidate per value multiplier at the requested overlap.

    Returns the names, the ads each candidate would serve, and the policy
    objects themselves. The policies are returned rather than rebuilt on demand
    because `make_policy` reseeds the global torch generator, so redrawing one
    later would shift every subsequent draw.

    A multiplier with no reachable grid point is skipped, so a trial can report
    fewer candidates than were asked for.
    """
    names: list[str] = []
    served: dict[str, torch.Tensor] = {}
    policies: dict[str, Any] = {}
    for index, value_mult in enumerate(config.value_mults):
        match = lookup_alpha_beta(sweep, value_mult, ess_target)
        if match is None:
            continue
        position = (
            oracle_center
            + match["alpha"] * distance * v_along
            + match["beta"] * orth_dirs[index]
        )
        candidate = make_policy(
            config.n_kernels,
            position[: config.user_dim],
            position[config.user_dim :],
            config.delta,
            config.sigma_range,
            config.alpha_range,
            seed=target_seed + CANDIDATE_SEED_OFFSET + index,
        )
        name = f"p{index + 1}"
        names.append(name)
        policies[name] = candidate
        served[name] = top1_from_slate(
            candidate, sampled_users, candidate_ads, config.top_k
        )
    return names, served, policies


def _fit_domain_classifier(
    config: TrialConfig,
    names: Sequence[str],
    train_users: torch.Tensor,
    train_log_ads: torch.Tensor,
    served: dict[str, torch.Tensor],
    train_idx: torch.Tensor,
) -> tuple[Any, float, list[int]]:
    """Train the classifier that separates logging from each candidate."""
    policy_embeddings = {"logging": torch.cat([train_users, train_log_ads], dim=1)}
    policy_to_class = {"logging": 0}
    for index, name in enumerate(names):
        policy_embeddings[name] = torch.cat(
            [train_users, served[name][train_idx]], dim=1
        )
        policy_to_class[name] = index + 1

    classifier, temperature, _ = train_calibrated_domain_classifier(
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
    return classifier, temperature, [policy_to_class[n] for n in names]


def _fit_reward_models(
    config: TrialConfig,
    names: Sequence[str],
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    cal_features: torch.Tensor,
    cal_labels: torch.Tensor,
    mixture_weights: torch.Tensor,
    per_policy_weights: torch.Tensor,
    device: torch.device | None,
) -> tuple[Any, Any, dict[str, Any], float, float, dict[str, float]]:
    """Fit the unweighted, mixture-weighted, and per-policy reward models."""
    hidden = list(config.hidden)
    d_in = config.feature_dim

    unweighted = train_reward_model(
        MLPDMRewardModel(d_in=d_in, hidden_dims=hidden),
        train_features,
        train_labels,
        epochs=config.epochs,
        device=device,
    )
    weighted = train_reward_model(
        MLPDMRewardModel(d_in=d_in, hidden_dims=hidden),
        train_features,
        train_labels,
        weights=mixture_weights,
        epochs=config.epochs,
        device=device,
    )
    temp_unweighted = (
        calibrate_temperature(unweighted, cal_features, cal_labels)
        if config.calibrate
        else 1.0
    )
    temp_weighted = (
        calibrate_temperature(weighted, cal_features, cal_labels)
        if config.calibrate
        else 1.0
    )

    per_policy: dict[str, Any] = {}
    per_policy_temps: dict[str, float] = {}
    for index, name in enumerate(names):
        per_policy[name] = train_reward_model(
            MLPDMRewardModel(d_in=d_in, hidden_dims=hidden),
            train_features,
            train_labels,
            weights=per_policy_weights[:, index],
            epochs=config.epochs,
            device=device,
        )
        per_policy_temps[name] = (
            calibrate_temperature(per_policy[name], cal_features, cal_labels)
            if config.calibrate
            else 1.0
        )
    return (
        unweighted,
        weighted,
        per_policy,
        temp_unweighted,
        temp_weighted,
        per_policy_temps,
    )


@dataclass(frozen=True)
class TrialSetup:
    """Everything a trial needs before any estimator is fitted.

    Shared verbatim by the main experiment and the domain-classifier ablation,
    which begin identically. Keeping one implementation is what guarantees the
    two experiments see the same world, the same logged data, and the same
    candidate placements for a given seed.
    """

    oracle: Oracle
    sampled_users: torch.Tensor
    candidate_ads: torch.Tensor
    log_policy: Policy
    log_center: torch.Tensor
    logged: LoggedData
    split: DataSplit
    oracle_center: torch.Tensor
    v_along: torch.Tensor
    distance: float
    orth_dirs: list[torch.Tensor]
    calib_results: list[SweepPoint]
    target_seed: int


def prepare_trial(
    config: TrialConfig,
    users: torch.Tensor,
    ads: torch.Tensor,
    seed: int,
    target_seed: int | None = None,
    calib_results: list[SweepPoint] | None = None,
) -> TrialSetup:
    """Seed, build the world, simulate logging, and position the candidates.

    The order of the stages below is load-bearing; see the module docstring.
    """
    if target_seed is None:
        target_seed = seed

    # numpy's global generator is read by the classifier's K-fold shuffle, so
    # this seeding is load-bearing rather than defensive.
    torch.manual_seed(seed)
    np.random.seed(seed)

    oracle: Oracle = make_oracle(
        config.oracle_ctr,
        users,
        ads,
        n_kernels=config.oracle_n_kernels,
        delta=config.oracle_delta,
        sigma_range=config.oracle_sigma_range,
        alpha_range=config.oracle_alpha_range,
        seed=seed,
    )

    sampled_users, candidate_ads = sample_impressions(
        users, ads, config.n_impressions, config.top_k, seed + IMPRESSION_SEED_OFFSET
    )
    log_policy, log_center = make_logging_policy(
        config.n_kernels,
        config.user_dim,
        config.ad_dim,
        config.delta,
        config.sigma_range,
        config.alpha_range,
        center_seed=seed + LOGGING_CENTER_SEED_OFFSET,
        policy_seed=seed + LOGGING_POLICY_SEED_OFFSET,
    )
    logged = collect_logged_data(
        log_policy, oracle, sampled_users, candidate_ads, config.top_k
    )
    if config.crossfit:
        n_dc, n_train, n_cal = config.crossfit_counts
        split = crossfit_split_indices(
            config.n_impressions, seed + SPLIT_SEED_OFFSET, n_dc, n_train, n_cal
        )
    else:
        split = split_indices(config.n_impressions, seed + SPLIT_SEED_OFFSET)

    oracle_center = torch.cat([oracle["center_u"], oracle["center_a"]])
    v_along, distance = shift_axis(oracle_center, log_center)
    orth_dirs = orthogonal_directions(v_along, len(config.value_mults), target_seed)

    if calib_results is None:
        w_orth = sweep_orthogonal_direction(v_along, seed + SWEEP_DIRECTION_SEED_OFFSET)
        calib_results = calibration_sweep(
            oracle,
            log_policy,
            sampled_users,
            candidate_ads,
            logged["served_ads"],
            oracle_center,
            v_along,
            w_orth,
            distance,
            logged["v_log"],
            config.n_kernels,
            config.delta,
            config.sigma_range,
            config.alpha_range,
            config.top_k,
            seed=seed + SWEEP_SEED_OFFSET,
            clip=config.clip,
            calib_n_samples=config.calib_n_samples,
        )

    return TrialSetup(
        oracle=oracle,
        sampled_users=sampled_users,
        candidate_ads=candidate_ads,
        log_policy=log_policy,
        log_center=log_center,
        logged=logged,
        split=split,
        oracle_center=oracle_center,
        v_along=v_along,
        distance=distance,
        orth_dirs=orth_dirs,
        calib_results=calib_results,
        target_seed=target_seed,
    )


def candidates_at_overlap(
    config: TrialConfig,
    setup: TrialSetup,
    ess_target: float,
) -> tuple[list[str], dict[str, torch.Tensor], dict[str, Any]]:
    """Candidates placed at one overlap level. Shared with the ablation."""
    return _candidates_at_overlap(
        config,
        setup.calib_results,
        ess_target,
        setup.oracle_center,
        setup.v_along,
        setup.distance,
        setup.orth_dirs,
        setup.target_seed,
        setup.sampled_users,
        setup.candidate_ads,
    )


def run_trial(
    config: TrialConfig,
    users: torch.Tensor,
    ads: torch.Tensor,
    seed: int,
    target_seed: int | None = None,
    calib_results: list[SweepPoint] | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Run one seed across every ESS target, returning one record per target.

    `users` and `ads` are drawn once and shared across seeds. `target_seed`
    seeds the candidate directions
    independently of the trial, and `calib_results` lets a caller reuse a
    positioning sweep.
    """
    setup = prepare_trial(config, users, ads, seed, target_seed, calib_results)
    target_seed = setup.target_seed
    oracle = setup.oracle
    sampled_users = setup.sampled_users
    logged = setup.logged
    split = setup.split

    train_idx, cal_idx, eval_idx = (
        split["train"],
        split["calibration"],
        split["evaluation"],
    )
    train_features = logged["features"][train_idx]
    train_labels = logged["rewards"][train_idx]
    cal_features = logged["features"][cal_idx]
    cal_labels = logged["rewards"][cal_idx]
    eval_users = sampled_users[eval_idx]
    # The classifier reads its own fold, which is the reward model's rows
    # unless cross-fitting is on.
    dc_idx = split["dc"]
    dc_users = sampled_users[dc_idx]
    dc_log_ads = logged["served_ads"][dc_idx]

    records: list[dict[str, Any]] = []

    for ess_target in config.ess_targets:
        names, served, _policies = candidates_at_overlap(config, setup, ess_target)
        if not names:
            continue

        # Fitted on the classifier's own fold, which equals the reward
        # model's unless cross-fitting is on.
        classifier, dc_temperature, target_classes = _fit_domain_classifier(
            config, names, dc_users, dc_log_ads, served, dc_idx
        )
        train_probs = compute_calibrated_probs(
            classifier, train_features, dc_temperature
        )
        mixture_weights = torch.from_numpy(
            compute_importance_weights(
                train_probs,
                logging_class=0,
                num_target_policies=len(names),
                clip=config.clip,
            )
        ).float()
        weight_diagnostics = compute_weight_diagnostics(mixture_weights.numpy())
        per_policy_weights = torch.as_tensor(
            np.asarray(
                compute_per_domain_density_ratios(
                    train_probs,
                    logging_class=0,
                    target_classes=target_classes,
                    clip=config.clip,
                )
            )
        ).float()

        (
            unweighted,
            weighted,
            per_policy,
            temp_unweighted,
            temp_weighted,
            per_policy_temps,
        ) = _fit_reward_models(
            config,
            names,
            train_features,
            train_labels,
            cal_features,
            cal_labels,
            mixture_weights,
            per_policy_weights,
            device,
        )

        record = _evaluate(
            config=config,
            names=names,
            served=served,
            oracle=oracle,
            logged=logged,
            eval_idx=eval_idx,
            eval_users=eval_users,
            classifier=classifier,
            dc_temperature=dc_temperature,
            target_classes=target_classes,
            models=(unweighted, weighted, per_policy),
            temperatures=(temp_unweighted, temp_weighted, per_policy_temps),
        )
        record.update(
            {
                "seed": seed,
                "target_seed": target_seed,
                "ess_tgt": ess_target,
                "avg_ess": weight_diagnostics["ess_ratio"],
                "dc_mix_ess": weight_diagnostics["ess_ratio"],
                "V_log": logged["v_log"],
                "n_policies": len(names),
            }
        )
        records.append(record)
        if verbose:
            logger.info(
                "ess_tgt=%.2f realized=%.3f n=%d",
                ess_target,
                weight_diagnostics["ess_ratio"],
                len(names),
            )

    return records


def _evaluate(
    config: TrialConfig,
    names: Sequence[str],
    served: dict[str, torch.Tensor],
    oracle: Oracle,
    logged: LoggedData,
    eval_idx: torch.Tensor,
    eval_users: torch.Tensor,
    classifier: Any,
    dc_temperature: float,
    target_classes: list[int],
    models: tuple[Any, Any, dict[str, Any]],
    temperatures: tuple[float, float, dict[str, float]],
) -> dict[str, Any]:
    """Score every estimator on the evaluation fold."""
    unweighted, weighted, per_policy = models
    temp_unweighted, temp_weighted, per_policy_temps = temperatures

    log_features = torch.cat([eval_users, logged["served_ads"][eval_idx]], dim=1)
    rewards = logged["rewards"][eval_idx]
    pred_log_unweighted = predict_calibrated(unweighted, log_features, temp_unweighted)
    pred_log_weighted = predict_calibrated(weighted, log_features, temp_weighted)
    eval_probs = compute_calibrated_probs(classifier, log_features, dc_temperature)
    eval_weights = torch.as_tensor(
        np.asarray(
            compute_per_domain_density_ratios(
                eval_probs,
                logging_class=0,
                target_classes=target_classes,
                clip=config.clip,
            )
        )
    ).float()

    with torch.no_grad():
        logging_reward = score_pairs(oracle, eval_users, logged["served_ads"][eval_idx])
    v_true = {"logging": clamp_reward(logging_reward).mean().item()}
    v_hat: dict[str, dict[str, float]] = {m: {} for m in METHODS}
    per_policy_log: dict[str, float] = {}
    cold_start: dict[float, dict[str, float]] = {t: {} for t in COLD_START_TAUS}

    for index, name in enumerate(names):
        target_ads = served[name][eval_idx]
        target_features = torch.cat([eval_users, target_ads], dim=1)
        target_probs = compute_calibrated_probs(
            classifier, target_features, dc_temperature
        )
        for tau, mass in cold_start_masses(target_probs).items():
            cold_start[tau][name] = mass

        with torch.no_grad():
            true_reward = clamp_reward(score_pairs(oracle, eval_users, target_ads))
        v_true[name] = true_reward.mean().item()

        pred_unweighted = predict_calibrated(
            unweighted, target_features, temp_unweighted
        )
        pred_weighted = predict_calibrated(weighted, target_features, temp_weighted)
        weights = eval_weights[:, index]

        v_hat["u"][name] = pred_unweighted.mean().item()
        v_hat["w"][name] = pred_weighted.mean().item()
        v_hat["ips_plain"][name], v_hat["ips_sn"][name] = ips_values(rewards, weights)
        v_hat["dru_plain"][name], v_hat["dru_sn"][name] = dr_values(
            rewards, weights, pred_log_unweighted, pred_unweighted
        )
        v_hat["drw_plain"][name], v_hat["drw_sn"][name] = dr_values(
            rewards, weights, pred_log_weighted, pred_weighted
        )

        pred_per_policy = predict_calibrated(
            per_policy[name], target_features, per_policy_temps[name]
        )
        v_hat["pp"][name] = pred_per_policy.mean().item()
        per_policy_log[name] = (
            predict_calibrated(per_policy[name], log_features, per_policy_temps[name])
            .mean()
            .item()
        )

    return _assemble_record(
        names,
        v_true,
        v_hat,
        per_policy_log,
        rewards,
        pred_log_unweighted,
        pred_log_weighted,
        logged["v_log"],
        cold_start,
    )


def _assemble_record(
    names: Sequence[str],
    v_true: dict[str, float],
    v_hat: dict[str, dict[str, float]],
    per_policy_log: dict[str, float],
    rewards: torch.Tensor,
    pred_log_unweighted: torch.Tensor,
    pred_log_weighted: torch.Tensor,
    v_log: float,
    cold_start: dict[float, dict[str, float]],
) -> dict[str, Any]:
    """Score each estimator against its own logging baseline and flatten."""
    reward_mean = rewards.mean().item()
    baselines: dict[str, float | dict[str, float]] = {
        "u": pred_log_unweighted.mean().item(),
        "w": pred_log_weighted.mean().item(),
        "pp": per_policy_log,
    }
    for method in METHODS:
        baselines.setdefault(method, reward_mean)

    candidate_true = {n: v_true[n] for n in names}
    record: dict[str, Any] = {}
    for method in METHODS:
        decision = compute_decision_metrics(
            candidate_true, v_hat[method], v_true["logging"], baselines[method], names
        )
        record[f"acc1_{method}"] = decision["acc1"]
        record[f"regret_{method}"] = decision["regret"]
        record[f"dir_{method}"] = decision["dir_acc"]
        record[f"mae_v_{method}"] = decision["mae_v"]
        record[f"mae_d_{method}"] = decision["mae_delta"]

    record["names"] = list(names)
    record["vk"] = [v_true[n] / v_log for n in names]
    record["Vtrue"] = [v_true[n] for n in names]
    for method in METHODS:
        record[f"Vhat_{method}"] = [v_hat[method][n] for n in names]
    record["cs01"] = [cold_start[0.01][n] for n in names]
    record["cs05"] = [cold_start[0.05][n] for n in names]
    record["cs10"] = [cold_start[0.1][n] for n in names]
    return record
