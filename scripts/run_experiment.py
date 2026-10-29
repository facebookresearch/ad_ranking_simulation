# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run an experiment from a config file.

    python -m scripts.run_experiment --config configs/table1.yaml --out results/

Each seed is written to its own file as soon as it finishes and skipped on a
rerun, so a multi-hour sweep survives interruption and can be split across
machines by passing disjoint `--seeds`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import yaml
from adsim.ablation import run_dc_ablation
from adsim.clip_sweep import run_clip_sweep
from adsim.config import TrialConfig
from adsim.trial import run_trial
from adsim.world import make_world

logger: logging.Logger = logging.getLogger(__name__)


def _run_seed(
    config: TrialConfig,
    users: torch.Tensor,
    ads: torch.Tensor,
    seed: int,
    device: torch.device | None,
) -> list[dict]:
    """Dispatch to the experiment the config names."""
    if config.experiment == "trial":
        return run_trial(config, users, ads, seed=seed, device=device)
    if config.experiment == "clip_sweep":
        return run_clip_sweep(config, users, ads, seed=seed, device=device)
    if config.experiment == "dc_ablation":
        return run_dc_ablation(
            config, users, ads, seed=seed, device=device, dc_eval_n=config.dc_eval_n
        )
    raise ValueError(f"unknown experiment: {config.experiment!r}")


def load_config(path: Path) -> TrialConfig:
    """Read a YAML config into a `TrialConfig`."""
    with open(path) as handle:
        return TrialConfig.from_dict(yaml.safe_load(handle) or {})


def seed_output_path(out_dir: Path, config_name: str, seed: int) -> Path:
    return out_dir / f"{config_name}_seed{seed}.json"


def run(
    config: TrialConfig,
    out_dir: Path,
    config_name: str,
    seeds: tuple[int, ...] | None = None,
    device: torch.device | None = None,
) -> list[dict]:
    """Run every seed, caching each to disk, and return all records."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # The population is drawn once and shared across seeds.
    users, ads = make_world(
        config.num_users,
        config.num_ads,
        config.user_dim,
        config.ad_dim,
        seed=config.world_seed,
    )

    all_records: list[dict] = []
    for seed in seeds if seeds is not None else config.seeds:
        cache_path = seed_output_path(out_dir, config_name, seed)
        if cache_path.exists():
            records = json.loads(cache_path.read_text())
            logger.info("seed %d: %d cached records", seed, len(records))
            all_records.extend(records)
            continue

        started = time.monotonic()
        records = _run_seed(config, users, ads, seed, device)
        cache_path.write_text(json.dumps(records))
        logger.info(
            "seed %d: %d records in %.0fs -> %s",
            seed,
            len(records),
            time.monotonic() - started,
            cache_path,
        )
        all_records.extend(records)

    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override the config's seeds, to shard a run across machines.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run on CPU.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(args.config)
    records = run(
        config,
        args.out,
        config_name=args.config.stem,
        seeds=tuple(args.seeds) if args.seeds else None,
        device=torch.device("cpu") if args.cpu else None,
    )
    combined = args.out / f"{args.config.stem}.json"
    combined.write_text(json.dumps(records))
    logger.info("%d records -> %s", len(records), combined)


if __name__ == "__main__":
    main()
