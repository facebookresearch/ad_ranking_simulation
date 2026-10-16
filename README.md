# Ad Ranking Simulation

A synthetic ad-ranking simulator and off-policy evaluation harness for
counterfactual evaluation of ranking policies when logged action propensities
are unavailable.

This code accompanies the NeurIPS 2026 paper *Beyond IPS: Reliable
Counterfactual Evaluation in Multi-Stage Ad Systems without Logged
Propensities* and reproduces its synthetic results. Production data,
embeddings, and trained production models will not be released.

## What this reproduces

| Paper artifact | Config | Rendered by |
| --- | --- | --- |
| Table 1: estimator comparison across ESS regimes | `configs/table1.yaml` | `scripts/make_tables.py` |
| Appendix D: Top-1 accuracy, Top-1 regret and Value MAE by realized-ESS bin, and per-policy Value MAE | `configs/table1.yaml` | `scripts/make_tables.py` |
| Appendix E: domain-classifier reliability (both tables) | `configs/dc_ablation.yaml` | `scripts/make_tables.py` |
| Appendix F: sensitivity to importance-weight clipping (figure) | `configs/clip_sensitivity.yaml` | `scripts/make_figures.py` |

## Requirements

Python 3.10 or 3.11 and the packages in `requirements.txt`. Everything runs on
CPU. Run all commands from the repository root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests
```

## Reproducing the results

`run_experiment` writes one JSON file per seed into `--out` as soon as the seed
finishes, skips seeds whose file already exists, and writes all records to
`<out>/<config name>.json` at the end.

**Table 1 and Appendix D**

```bash
python -m scripts.run_experiment --config configs/table1.yaml --out results/table1 --cpu
python -m scripts.make_tables --config configs/table1.yaml --results results/table1
```

**Appendix E**

```bash
python -m scripts.run_experiment --config configs/dc_ablation.yaml --out results/dc_ablation --cpu
python -m scripts.make_tables --config configs/dc_ablation.yaml --results results/dc_ablation
```

**Appendix F**

```bash
python -m scripts.run_experiment --config configs/clip_sensitivity.yaml --out results/clip_sensitivity --cpu
python -m scripts.make_figures --out figures --clip-sensitivity results/clip_sensitivity
```

`make_tables` prints the tables; add `--json <file>` to also save them as JSON.
`make_figures` writes `figures/clip_sensitivity.png`.

### Running in parallel

Seeds are independent. Give each process a disjoint set of seeds and its own
output directory, then pass every directory to the renderer:

```bash
for i in 0 1 2 3; do
  seeds=$(python -c "s = [42 + 1000 * k for k in range(50)]; print(*s[$i::4])")
  python -m scripts.run_experiment --config configs/table1.yaml \
    --out results/table1_s$i --cpu --seeds $seeds &
done
wait
python -m scripts.make_tables --config configs/table1.yaml \
  --results results/table1_s0 results/table1_s1 results/table1_s2 results/table1_s3
```

The 50 seeds are 42, 1042, ..., 49042, as set in `adsim/config.py`.

## Layout

```
adsim/     simulator, policies, domain classifier, reward models, estimators, tables, figures
configs/   one config per experiment
scripts/   run_experiment.py, make_tables.py, make_figures.py
tests/     unit tests
```

## Citation

```bibtex
@inproceedings{malmir2026beyond,
  title     = {Beyond {IPS}: Reliable Counterfactual Evaluation in Multi-Stage Ad Systems without Logged Propensities},
  author    = {Malmir, Mohsen and Radwan, Mohammed A. and Nassif, Houssam and Bayir, Murat Ali},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

This code is released under the CC BY-NC 4.0 license; see the `LICENSE` file in
the root directory of this source tree.
