# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for table construction from trial records."""

from __future__ import annotations

import unittest
from typing import Any, Mapping

import numpy as np
from adsim.tables import (
    by_realized_overlap,
    dc_ablation_tables,
    format_binned,
    format_dc_diagnostics,
    format_dc_performance,
    format_main_comparison,
    format_per_candidate,
    main_comparison,
    per_candidate_value_error,
    realized_bin,
    regime,
    value_mae_percent,
)

METHODS = (("u", "Unweighted DM"), ("w", "Weighted DM"))


def _acc1(record: Mapping[str, Any], method: str) -> float:
    return float(record[f"acc1_{method}"])


VALUE_MULTS = (0.95, 1.05)


def _record(ess_target: float, avg_ess: float, **overrides) -> dict:
    record = {
        "ess_tgt": ess_target,
        "avg_ess": avg_ess,
        "acc1_u": 1,
        "acc1_w": 0,
        "regret_u": 0.001,
        "regret_w": 0.002,
        "mae_v_u": 0.01,
        "mae_v_w": 0.02,
        "Vtrue": [0.10, 0.20],
        "Vhat_u": [0.11, 0.20],
        "Vhat_w": [0.10, 0.22],
        "cs10": [0.3, 0.5],
    }
    record.update(overrides)
    return record


class RegimeTest(unittest.TestCase):
    def test_boundaries(self) -> None:
        self.assertEqual("Low", regime(0.25))
        self.assertEqual("Low", regime(0.35))
        self.assertEqual("Mid", regime(0.4))
        self.assertEqual("Mid", regime(0.65))
        self.assertEqual("High", regime(0.7))
        self.assertEqual("High", regime(0.85))

    def test_realized_bin_rounds_to_tenths(self) -> None:
        self.assertEqual(0.4, realized_bin(0.42))
        self.assertEqual(0.5, realized_bin(0.54))
        self.assertEqual(0.6, realized_bin(0.61))

    def test_realized_bin_breaks_midpoint_ties_to_even(self) -> None:
        """Python rounds halves to even, so 0.45 and 0.55 go opposite ways.
        This decides which bin an edge-case trial is counted in."""
        self.assertEqual(0.4, realized_bin(0.45))
        self.assertEqual(0.6, realized_bin(0.55))
        self.assertEqual(0.4, realized_bin(0.35))

    def test_realized_bin_is_clamped(self) -> None:
        self.assertEqual(0.1, realized_bin(0.0))
        self.assertEqual(1.0, realized_bin(1.7))


class ValueMaePercentTest(unittest.TestCase):
    def test_is_relative_to_true_value(self) -> None:
        record = _record(0.45, 0.45, Vtrue=[0.10, 0.20], Vhat_u=[0.11, 0.22])
        # 10% on both candidates.
        self.assertAlmostEqual(10.0, value_mae_percent(record, "u"), places=9)

    def test_perfect_estimate_is_zero(self) -> None:
        record = _record(0.45, 0.45, Vhat_u=[0.10, 0.20])
        self.assertAlmostEqual(0.0, value_mae_percent(record, "u"), places=12)

    def test_skips_candidates_with_zero_true_value(self) -> None:
        record = _record(0.45, 0.45, Vtrue=[0.0, 0.20], Vhat_u=[0.5, 0.22])
        self.assertAlmostEqual(10.0, value_mae_percent(record, "u"), places=9)


class MainComparisonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            _record(0.25, 0.2),
            _record(0.35, 0.3),
            _record(0.45, 0.5),
            _record(0.85, 0.9),
        ]
        self.table = main_comparison(self.records, METHODS)

    def test_counts_trials_per_regime(self) -> None:
        regimes = self.table["regimes"]
        self.assertEqual(2, regimes["Low"]["n_trials"])
        self.assertEqual(1, regimes["Mid"]["n_trials"])
        self.assertEqual(1, regimes["High"]["n_trials"])

    def test_reports_cold_start_mass_per_regime(self) -> None:
        self.assertAlmostEqual(0.4, self.table["regimes"]["Low"]["cold_start_mass"])

    def test_reports_every_method(self) -> None:
        self.assertEqual({"Unweighted DM", "Weighted DM"}, set(self.table["methods"]))

    def test_averages_within_a_regime(self) -> None:
        cell = self.table["methods"]["Unweighted DM"]["Low"]
        assert cell is not None
        self.assertAlmostEqual(1.0, cell["acc1"])
        self.assertAlmostEqual(0.001, cell["regret"])

    def test_empty_regime_is_none_not_an_error(self) -> None:
        table = main_comparison([_record(0.25, 0.2)], METHODS)
        self.assertIsNone(table["methods"]["Unweighted DM"]["High"])


class ByRealizedOverlapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            _record(0.45, 0.42, acc1_u=1),
            _record(0.45, 0.44, acc1_u=0),
            _record(0.65, 0.71, acc1_u=1),
        ]

    def test_bins_by_realized_not_target(self) -> None:
        table = by_realized_overlap(self.records, _acc1, METHODS)
        self.assertEqual({0.4, 0.7}, set(table))
        self.assertEqual(2, table[0.4]["n"])

    def test_reports_mean_and_standard_error(self) -> None:
        table = by_realized_overlap(self.records, _acc1, METHODS)
        cell = table[0.4]["methods"]["Unweighted DM"]
        self.assertAlmostEqual(0.5, cell["mean"])
        self.assertGreater(cell["se"], 0.0)

    def test_single_record_bin_has_zero_standard_error(self) -> None:
        table = by_realized_overlap(self.records, _acc1, METHODS)
        self.assertEqual(0.0, table[0.7]["methods"]["Unweighted DM"]["se"])


class PerCandidateTest(unittest.TestCase):
    def test_reports_error_per_value_multiplier(self) -> None:
        records = [_record(0.45, 0.45), _record(0.55, 0.55)]
        table = per_candidate_value_error(records, VALUE_MULTS, METHODS)
        self.assertEqual(set(VALUE_MULTS), set(table))
        # Unweighted is off by 0.01 on the first candidate, exact on the second.
        first = table[0.95]["methods"]["Unweighted DM"]
        second = table[1.05]["methods"]["Unweighted DM"]
        self.assertAlmostEqual(0.01, first["mean"], places=9)
        self.assertAlmostEqual(0.0, second["mean"], places=9)

    def test_counts_the_trials_containing_each_candidate(self) -> None:
        """A trial can drop a candidate when no grid point reaches its value."""
        short = _record(0.45, 0.45, Vtrue=[0.10], Vhat_u=[0.11], Vhat_w=[0.10])
        table = per_candidate_value_error(
            [_record(0.45, 0.45), short], VALUE_MULTS, METHODS
        )
        self.assertEqual(2, table[0.95]["n"])
        self.assertEqual(1, table[1.05]["n"])
        first = table[0.95]["methods"]["Unweighted DM"]
        second = table[1.05]["methods"]["Unweighted DM"]
        self.assertAlmostEqual(0.01, first["mean"], places=9)
        self.assertFalse(np.isnan(second["mean"]))


def _dc_record(condition: str, n_ad_masked: int = 0, **overrides) -> dict:
    record = {
        "dc_condition": condition,
        "n_ad_masked": n_ad_masked,
        "Vtrue": [0.10, 0.20],
        "dc_nll": 1.7,
        "dc_cand_acc": 0.3,
        "ratio_spearman": 0.5,
        "avg_ess": 0.6,
        "cs_err10": [0.1, 0.3],
    }
    for method in ("u", "w", "pp", "ips_plain", "dru_plain", "drw_plain"):
        record[f"acc1_{method}"] = 1
        record[f"regret_{method}"] = 0.001
        record[f"Vhat_{method}"] = [0.11, 0.20]
    record.update(overrides)
    return record


class DcAblationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            _dc_record("clean"),
            _dc_record("clean", cs_err10=[0.5]),
            _dc_record("mask100", n_ad_masked=3, ratio_spearman=-0.01),
        ]
        self.tables = dc_ablation_tables(self.records, ad_dim=3)

    def test_reports_the_conditions_present_in_the_paper_order(self) -> None:
        self.assertEqual(
            ["Clean DC", "Policy-agnostic (100% ad masked)"],
            list(self.tables["performance"]),
        )

    def test_scores_each_classifier_dependent_estimator(self) -> None:
        cell = self.tables["performance"]["Clean DC"]["Weighted DR"]
        self.assertAlmostEqual(1.0, cell["acc1"])
        self.assertAlmostEqual(0.001, cell["regret"])
        # Off by 10% on the first candidate and exact on the second.
        self.assertAlmostEqual(5.0, cell["mae_pct"])

    def test_scores_the_per_policy_baseline_from_its_own_fields(self) -> None:
        tables = dc_ablation_tables(
            [_dc_record("clean", acc1_pp=0, Vhat_pp=[0.12, 0.20])], ad_dim=3
        )
        cell = tables["performance"]["Clean DC"]["Per-policy Weighted DM"]
        self.assertAlmostEqual(0.0, cell["acc1"])
        # Off by 20% on the first candidate and exact on the second.
        self.assertAlmostEqual(10.0, cell["mae_pct"])

    def test_cold_start_error_averages_within_each_record_first(self) -> None:
        """Record means 0.2 and 0.5 average to 0.35; pooling would give 0.3."""
        diagnostics = self.tables["diagnostics"]["Clean DC"]
        self.assertAlmostEqual(0.35, diagnostics["cold_start_error"])

    def test_ratio_correlation_is_undefined_once_every_ad_feature_is_masked(
        self,
    ) -> None:
        masked = self.tables["diagnostics"]["Policy-agnostic (100% ad masked)"]
        self.assertIsNone(masked["ratio_correlation"])
        clean = self.tables["diagnostics"]["Clean DC"]["ratio_correlation"]
        assert clean is not None
        self.assertAlmostEqual(0.5, clean)

    def test_unweighted_dm_is_reported_once_from_the_clean_condition(self) -> None:
        self.assertAlmostEqual(5.0, self.tables["unweighted_dm"]["mae_pct"])

    def test_formatting_shows_every_condition_and_the_undefined_cell(self) -> None:
        performance = format_dc_performance(self.tables)
        diagnostics = format_dc_diagnostics(self.tables)
        self.assertIn("Clean DC", performance)
        self.assertIn("Per-policy Weighted DM", performance)
        self.assertIn("Unweighted DM", performance)
        self.assertIn("Policy-agnostic (100% ad masked)", diagnostics)
        self.assertIn("--", diagnostics)


class FormattingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [_record(0.25, 0.2), _record(0.45, 0.5), _record(0.85, 0.9)]

    def test_main_comparison_renders_every_method_and_regime(self) -> None:
        text = format_main_comparison(main_comparison(self.records, METHODS))
        for label in ("Unweighted DM", "Weighted DM", "Low", "Mid", "High"):
            self.assertIn(label, text)

    def test_binned_table_renders_bins(self) -> None:
        table = by_realized_overlap(self.records, _acc1, METHODS)
        text = format_binned(table, "TITLE", methods=METHODS)
        self.assertIn("TITLE", text)
        self.assertIn("+/-", text)

    def test_per_candidate_table_renders_multipliers(self) -> None:
        table = per_candidate_value_error(self.records, VALUE_MULTS, METHODS)
        text = format_per_candidate(table, METHODS)
        self.assertIn("0.95", text)
        self.assertIn("1.05", text)
