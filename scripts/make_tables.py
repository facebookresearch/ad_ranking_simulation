# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Render the paper's tables from a completed run.

    python -m scripts.make_tables --results results/table1 \\
        --config configs/table1.yaml

`--results` takes the combined file `run_experiment` writes or directories of
per-seed files, and several of either, so shards can be read together. The
config's experiment decides what is rendered: Table 1 and the four Appendix D
tables for `trial`, the two Appendix E tables for `dc_ablation`. Writes text
to stdout, and JSON alongside it with `--json`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml
from adsim.config import TrialConfig
from adsim.tables import (
    by_realized_overlap,
    dc_ablation_tables,
    DcAblation,
    format_binned,
    format_dc_diagnostics,
    format_dc_performance,
    format_main_comparison,
    format_per_candidate,
    main_comparison,
    per_candidate_value_error,
)

logger: logging.Logger = logging.getLogger(__name__)


def load_records(paths: list[Path]) -> list[dict]:
    """Records from combined result files and directories of per-seed files.

    A directory is read for `*_seed*.json` only: `run_experiment` writes its
    combined file into the same directory, and reading both would count every
    record twice.
    """
    records: list[dict] = []
    for path in paths:
        files = sorted(path.glob("*_seed*.json")) if path.is_dir() else [path]
        for file in files:
            records.extend(json.loads(file.read_text()))
    return records


def build(records: list[dict], config: TrialConfig) -> dict[str, object]:
    """Table 1 and the Appendix D tables, as plain data."""
    return {
        "main": main_comparison(records),
        "accuracy_by_overlap": by_realized_overlap(
            records, lambda r, m: r[f"acc1_{m}"]
        ),
        "regret_by_overlap": by_realized_overlap(
            records, lambda r, m: r[f"regret_{m}"]
        ),
        "value_mae_by_overlap": by_realized_overlap(
            records, lambda r, m: r[f"mae_v_{m}"]
        ),
        "per_candidate": per_candidate_value_error(records, config.value_mults),
    }


def render(tables: dict[str, object]) -> str:
    """Table 1 and the Appendix D tables as text, in the paper's order."""
    return "\n\n".join(
        [
            "===== Main comparison, by overlap regime =====",
            format_main_comparison(tables["main"]),  # pyre-ignore[6]
            format_binned(
                tables["accuracy_by_overlap"],  # pyre-ignore[6]
                "===== Top-1 accuracy, by realized overlap =====",
            ),
            format_binned(
                tables["regret_by_overlap"],  # pyre-ignore[6]
                "===== Top-1 regret, by realized overlap =====",
                decimals=4,
            ),
            format_binned(
                tables["value_mae_by_overlap"],  # pyre-ignore[6]
                "===== Value MAE, by realized overlap =====",
                decimals=4,
            ),
            format_per_candidate(tables["per_candidate"]),  # pyre-ignore[6]
        ]
    )


def render_dc_ablation(tables: DcAblation) -> str:
    """Both Appendix E tables as text."""
    return "\n\n".join(
        [
            "===== Estimator performance under each domain-classifier condition =====",
            format_dc_performance(tables),
            "===== Domain-classifier reliability diagnostics =====",
            format_dc_diagnostics(tables),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = TrialConfig.from_dict(yaml.safe_load(args.config.read_text()) or {})
    records = load_records(args.results)
    if not records:
        raise SystemExit(f"no records found in {args.results}")
    logger.info("loaded %d records", len(records))

    data: object
    if config.experiment == "trial":
        trial_tables = build(records, config)
        text = render(trial_tables)
        data = trial_tables
    elif config.experiment == "dc_ablation":
        dc_tables = dc_ablation_tables(records, config.ad_dim)
        text = render_dc_ablation(dc_tables)
        data = dc_tables
    else:
        raise SystemExit(f"no tables are rendered for {config.experiment!r} runs")

    print(text)
    if args.json is not None:
        args.json.write_text(json.dumps(data, indent=2, default=str))
        logger.info("wrote %s", args.json)


if __name__ == "__main__":
    main()
