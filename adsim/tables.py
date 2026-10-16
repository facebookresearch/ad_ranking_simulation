# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Turning experiment records into the paper's tables.

Five tables come out of the Table 1 experiment's records:

- The main comparison (Table 1), binned into three overlap regimes by *target*
  effective sample size. Targets are used rather than realized values so the
  regimes have the same meaning across seeds.
- Three Appendix D tables, binned in width-0.1 bins of *realized* overlap.
  Those answer a different question, how estimators behave at the overlap
  actually achieved, so they use the achieved value and report a standard
  error.
- A per-candidate table (Appendix D), showing value error against how much
  better than the logging policy each candidate was asked to be, with the
  number of trials that contained the candidate.

Two more come out of the domain-classifier ablation's records (Appendix E):
estimator performance under each classifier condition, and the classifier's
reliability diagnostics.

Table 1 and the ablation report value error as a percentage of true value,
because the candidates sit within a few percent of each other; the Appendix D
tables report it in absolute terms, as the paper does.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, TypedDict

import numpy as np

# Reporting order and display names, as the paper prints them. The keys are
# record fields; `plain` rather than self-normalized forms are reported.
METHOD_LABELS: tuple[tuple[str, str], ...] = (
    ("ips_plain", "IPS"),
    ("u", "Unweighted DM"),
    ("w", "Weighted DM (ours)"),
    ("pp", "Per-policy Weighted DM"),
    ("dru_plain", "Unweighted DR"),
    ("drw_plain", "Weighted DR (ours)"),
)

# Classifier conditions in the order and wording of the Appendix E tables.
DC_CONDITION_LABELS: tuple[tuple[str, str], ...] = (
    ("clean", "Clean DC"),
    ("uncal", "Uncalibrated"),
    ("crossfit", "Cross-fit (5-fold OOF)"),
    ("mask33", "Noisy DC (33% ad masked)"),
    ("mask67", "Noisy DC (67% ad masked)"),
    ("mask100", "Policy-agnostic (100% ad masked)"),
)

# The estimators that depend on the classifier. Unweighted DM never touches it,
# so it is reported once rather than per condition.
DC_METHOD_LABELS: tuple[tuple[str, str], ...] = (
    ("w", "Weighted DM"),
    ("pp", "Per-policy Weighted DM"),
    ("ips_plain", "IPS"),
    ("dru_plain", "Unweighted DR"),
    ("drw_plain", "Weighted DR"),
)

REGIME_BOUNDS: tuple[float, float] = (0.4, 0.7)
REGIMES: tuple[str, ...] = ("Low", "Mid", "High")

LABEL_WIDTH = 22
CONDITION_WIDTH = 34


class RegimeInfo(TypedDict):
    """How many trials landed in a regime, and how cold-start-heavy it was."""

    n_trials: int
    cold_start_mass: float


class MethodCell(TypedDict):
    """One method's score in one regime."""

    acc1: float
    regret: float
    mae_pct: float


class MainComparison(TypedDict):
    """The main table: regime descriptions plus one row per method."""

    regimes: dict[str, RegimeInfo]
    methods: dict[str, dict[str, MethodCell | None]]


class Stat(TypedDict):
    """A mean and its standard error."""

    mean: float
    se: float


class OverlapBin(TypedDict):
    """One realized-overlap bin: its size and one statistic per method."""

    n: int
    methods: dict[str, Stat]


class CandidateRow(TypedDict):
    """One value multiplier: the trials containing it and each method's error."""

    n: int
    methods: dict[str, Stat]


class DcDiagnostics(TypedDict):
    """Classifier and representation reliability under one condition."""

    nll: float
    candidate_accuracy: float
    ratio_correlation: float | None
    estimated_ess: float
    cold_start_error: float
    weighted_dm_mae_pct: float
    weighted_dr_mae_pct: float


class DcAblation(TypedDict):
    """Both Appendix E tables, plus the condition-independent Unweighted DM."""

    performance: dict[str, dict[str, MethodCell]]
    unweighted_dm: MethodCell
    diagnostics: dict[str, DcDiagnostics]


def regime(ess_target: float) -> str:
    """Overlap regime of a target effective sample size."""
    low, mid = REGIME_BOUNDS
    if ess_target < low:
        return "Low"
    return "Mid" if ess_target < mid else "High"


def realized_bin(ess: float) -> float:
    """Realized overlap rounded into a width-0.1 bin, clamped to [0.1, 1.0].

    Uses Python's banker's rounding, so exact midpoints go to the even bin:
    0.45 lands in 0.4 while 0.55 lands in 0.6.
    """
    return round(min(max(ess, 0.1), 1.0) * 10) / 10


def value_mae_percent(record: Mapping[str, Any], method: str) -> float:
    """Mean absolute value error as a percentage of true value.

    Candidates with a true value of zero are skipped rather than dividing by
    it.
    """
    predicted = record[f"Vhat_{method}"]
    truth = record["Vtrue"]
    return float(
        np.mean([abs(p - t) / t * 100 for p, t in zip(predicted, truth) if t > 0])
    )


def _mean_and_se(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), 0.0
    mean = float(np.mean(values))
    se = float(np.std(values) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, se


def _method_cell(records: Sequence[Mapping[str, Any]], key: str) -> MethodCell:
    """Mean accuracy, regret, and percentage value error of one method."""
    return MethodCell(
        acc1=float(np.mean([r[f"acc1_{key}"] for r in records])),
        regret=float(np.mean([r[f"regret_{key}"] for r in records])),
        mae_pct=float(np.mean([value_mae_percent(r, key) for r in records])),
    )


def main_comparison(
    records: Sequence[Mapping[str, Any]],
    methods: Sequence[tuple[str, str]] = METHOD_LABELS,
) -> MainComparison:
    """Accuracy, regret, and value error per method in each overlap regime.

    Also reports how many trials fell in each regime and the mean cold-start
    mass there, which is what makes the regimes interpretable rather than
    arbitrary cut points.
    """
    regimes: dict[str, RegimeInfo] = {}
    for name in REGIMES:
        subset = [r for r in records if regime(float(r["ess_tgt"])) == name]
        cold = [c for r in subset for c in r.get("cs10", [])]
        regimes[name] = RegimeInfo(
            n_trials=len(subset),
            cold_start_mass=float(np.mean(cold)) if cold else float("nan"),
        )

    rows: dict[str, dict[str, MethodCell | None]] = {}
    for key, label in methods:
        cells: dict[str, MethodCell | None] = {}
        for name in REGIMES:
            subset = [r for r in records if regime(float(r["ess_tgt"])) == name]
            cells[name] = _method_cell(subset, key) if subset else None
        rows[label] = cells
    return MainComparison(regimes=regimes, methods=rows)


def by_realized_overlap(
    records: Sequence[Mapping[str, Any]],
    field: Callable[[Mapping[str, Any], str], float],
    methods: Sequence[tuple[str, str]] = METHOD_LABELS,
) -> dict[float, OverlapBin]:
    """One metric per method, in width-0.1 bins of realized overlap."""
    bins: dict[float, OverlapBin] = {}
    for value in sorted({realized_bin(float(r["avg_ess"])) for r in records}):
        subset = [r for r in records if realized_bin(float(r["avg_ess"])) == value]
        cells: dict[str, Stat] = {}
        for key, label in methods:
            mean, se = _mean_and_se([field(r, key) for r in subset])
            cells[label] = Stat(mean=mean, se=se)
        bins[value] = OverlapBin(n=len(subset), methods=cells)
    return bins


def per_candidate_value_error(
    records: Sequence[Mapping[str, Any]],
    value_mults: Sequence[float],
    methods: Sequence[tuple[str, str]] = METHOD_LABELS,
) -> dict[float, CandidateRow]:
    """Absolute value error per method, per candidate value multiplier.

    Shows whether error concentrates on the candidates furthest from the
    logging policy, which is what the cold-start story predicts. Candidates
    are matched to multipliers by their position in a record. A trial omits a
    candidate when no grid point reaches its value, and `n` counts the trials
    that contained it.
    """
    table: dict[float, CandidateRow] = {}
    for index, multiplier in enumerate(value_mults):
        subset = [r for r in records if len(r["Vtrue"]) > index]
        cells: dict[str, Stat] = {}
        for key, label in methods:
            mean, se = _mean_and_se(
                [abs(r[f"Vhat_{key}"][index] - r["Vtrue"][index]) for r in subset]
            )
            cells[label] = Stat(mean=mean, se=se)
        table[multiplier] = CandidateRow(n=len(subset), methods=cells)
    return table


def dc_ablation_tables(records: Sequence[Mapping[str, Any]], ad_dim: int) -> DcAblation:
    """Both Appendix E tables from the domain-classifier ablation's records.

    Cold-start error is averaged within each record first and then across
    records. The oracle ratio
    correlation is undefined once every ad feature is masked, because the
    estimated weights are then constant, so it is reported as `None` there.
    """
    performance: dict[str, dict[str, MethodCell]] = {}
    diagnostics: dict[str, DcDiagnostics] = {}
    for condition, label in DC_CONDITION_LABELS:
        subset = [r for r in records if r["dc_condition"] == condition]
        if not subset:
            continue
        performance[label] = {
            name: _method_cell(subset, key) for key, name in DC_METHOD_LABELS
        }
        fully_masked = all(int(r["n_ad_masked"]) >= ad_dim for r in subset)
        diagnostics[label] = DcDiagnostics(
            nll=float(np.mean([r["dc_nll"] for r in subset])),
            candidate_accuracy=float(np.mean([r["dc_cand_acc"] for r in subset])),
            ratio_correlation=(
                None
                if fully_masked
                else float(np.mean([r["ratio_spearman"] for r in subset]))
            ),
            estimated_ess=float(np.mean([r["avg_ess"] for r in subset])),
            cold_start_error=float(np.mean([np.mean(r["cs_err10"]) for r in subset])),
            weighted_dm_mae_pct=float(
                np.mean([value_mae_percent(r, "w") for r in subset])
            ),
            weighted_dr_mae_pct=float(
                np.mean([value_mae_percent(r, "drw_plain") for r in subset])
            ),
        )
    clean = [r for r in records if r["dc_condition"] == "clean"]
    return DcAblation(
        performance=performance,
        unweighted_dm=_method_cell(clean, "u"),
        diagnostics=diagnostics,
    )


def _format_cell(cell: MethodCell) -> str:
    return f"{cell['acc1']:.2f}/{cell['regret']:.4f}/{cell['mae_pct']:.2f}"


def format_main_comparison(table: MainComparison) -> str:
    """Render the main comparison as fixed-width text."""
    lines = ["Overlap regimes:"]
    regimes = table["regimes"]
    for name in REGIMES:
        info = regimes[name]
        lines.append(
            f"  {name:>4} ESS: n_trials={info['n_trials']}, "
            f"mean cold-start mass={info['cold_start_mass'] * 100:.0f}%"
        )
    header = f"{'Method':>{LABEL_WIDTH}} | " + " ".join(
        f"{name + ' Acc/Reg/MAE%':>24}" for name in REGIMES
    )
    lines.extend(["", header])
    for label, cells in table["methods"].items():
        rendered = []
        for name in REGIMES:
            cell = cells[name]
            rendered.append("n/a" if cell is None else _format_cell(cell))
        lines.append(
            f"{label:>{LABEL_WIDTH}} | " + " ".join(f"{c:>24}" for c in rendered)
        )
    return "\n".join(lines)


def format_binned(
    table: Mapping[float, OverlapBin],
    title: str,
    decimals: int = 3,
    methods: Sequence[tuple[str, str]] = METHOD_LABELS,
) -> str:
    """Render a realized-overlap table as fixed-width text."""
    lines = [
        title,
        "ESS   n | " + " ".join(f"{label:>{LABEL_WIDTH}}" for _, label in methods),
    ]
    for value, row in table.items():
        cells = []
        for _, label in methods:
            stats = row["methods"][label]
            cells.append(f"{stats['mean']:.{decimals}f}+/-{stats['se']:.{decimals}f}")
        lines.append(
            f"{value:>3} {row['n']:>3} | "
            + " ".join(f"{c:>{LABEL_WIDTH}}" for c in cells)
        )
    return "\n".join(lines)


def format_per_candidate(
    table: Mapping[float, CandidateRow],
    methods: Sequence[tuple[str, str]] = METHOD_LABELS,
) -> str:
    """Render the per-candidate table as fixed-width text."""
    lines = [
        "Per-candidate value error by requested value multiplier (mean+/-SE)",
        f"{'v_k':>5} {'n':>4} | "
        + " ".join(f"{label:>{LABEL_WIDTH}}" for _, label in methods),
    ]
    for multiplier, row in table.items():
        rendered = [
            f"{row['methods'][label]['mean']:.4f}+/-{row['methods'][label]['se']:.4f}"
            for _, label in methods
        ]
        lines.append(
            f"{multiplier:>5} {row['n']:>4} | "
            + " ".join(f"{c:>{LABEL_WIDTH}}" for c in rendered)
        )
    return "\n".join(lines)


def format_dc_performance(table: DcAblation) -> str:
    """Render the Appendix E estimator-performance table as fixed-width text."""
    names = [name for _, name in DC_METHOD_LABELS]
    lines = [
        f"{'DC condition':>{CONDITION_WIDTH}} | "
        + " ".join(f"{name + ' Acc/Reg/MAE%':>36}" for name in names)
    ]
    for label, cells in table["performance"].items():
        lines.append(
            f"{label:>{CONDITION_WIDTH}} | "
            + " ".join(f"{_format_cell(cells[name]):>36}" for name in names)
        )
    lines.extend(
        [
            "",
            "Unweighted DM, identical in every condition: "
            + _format_cell(table["unweighted_dm"]),
        ]
    )
    return "\n".join(lines)


def format_dc_diagnostics(table: DcAblation) -> str:
    """Render the Appendix E reliability-diagnostics table as fixed-width text."""
    header = (
        "NLL",
        "Cand. acc.",
        "Ratio corr.",
        "Est. ESS",
        "Cold-start err.",
        "Wtd. DM MAE%",
        "Wtd. DR MAE%",
    )
    lines = [
        f"{'DC condition':>{CONDITION_WIDTH}} | "
        + " ".join(f"{name:>15}" for name in header)
    ]
    for label, row in table["diagnostics"].items():
        ratio = row["ratio_correlation"]
        values = (
            f"{row['nll']:.2f}",
            f"{row['candidate_accuracy']:.2f}",
            "--" if ratio is None else f"{ratio:.2f}",
            f"{row['estimated_ess']:.2f}",
            f"{row['cold_start_error']:.2f}",
            f"{row['weighted_dm_mae_pct']:.2f}",
            f"{row['weighted_dr_mae_pct']:.2f}",
        )
        lines.append(
            f"{label:>{CONDITION_WIDTH}} | " + " ".join(f"{v:>15}" for v in values)
        )
    return "\n".join(lines)
