# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Render the paper's clip-sensitivity figure from a completed run.

    python -m scripts.make_figures --out figures/ \\
        --clip-sensitivity results/clip_sensitivity

`--clip-sensitivity` takes the combined file `run_experiment` writes or
directories of per-seed files, and several of either, so shards can be read
together.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from adsim.figures import plot_clip_sensitivity

logger: logging.Logger = logging.getLogger(__name__)

CLIP_FIGURE = "clip_sensitivity.png"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clip-sensitivity", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    records = load_records(args.clip_sensitivity)
    path = args.out / CLIP_FIGURE
    plot_clip_sensitivity(records, str(path))
    logger.info("%d records -> %s", len(records), path)


if __name__ == "__main__":
    main()
