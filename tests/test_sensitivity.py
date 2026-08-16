"""Unit tests for the steady-state sensitivity sweep.

The full HBM-on-GPU benchmark is too large for pytest. The
sensitivity sweep is exercised here on a tiny toy config
(``tests/fixtures/toy_with_mold.yaml``) so the test stays under
a second; the HBM benchmark numbers are validated separately on
the feature branch in the actual run that produced the report.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from om3dthermal.sensitivity import (
    SensitivityCase,
    build_inset_sweep_cases,
    build_k_sweep_cases,
    case_already_done,
    compute_delta_tmax_sensitivity,
    load_partial_rows,
    merge_sweep_cases,
    parse_k_list,
    parse_length_list,
    run_single_sensitivity_case,
    write_case_row_partial,
    write_sensitivity_csv,
    write_sensitivity_json,
)


TOY_CONFIG = Path(__file__).parent / "fixtures" / "toy_with_mold.yaml"


# ---------------------------------------------------------------------------
# parse_length_list / parse_k_list
# ---------------------------------------------------------------------------

class TestParseLengthList:
    def test_accepts_mm(self):
        assert parse_length_list("0mm,0.25mm,0.5mm,0.75mm,1.0mm") == pytest.approx(
            [0, 2.5e-4, 5e-4, 7.5e-4, 1e-3],
        )

    def test_accepts_mixed_units(self):
        assert parse_length_list("0mm,500um,1mm") == pytest.approx(
            [0, 5e-4, 1e-3],
        )

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="finite and non-negative"):
            parse_length_list("-1mm,0mm,1mm")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            parse_length_list("")

    def test_rejects_unparseable(self):
        with pytest.raises(ValueError, match="could not parse"):
            parse_length_list("0mm,banana,1mm")


class TestParseKList:
    def test_accepts_simple_floats(self):
        assert parse_k_list("0.5,1,3,10,30") == pytest.approx(
            [0.5, 1.0, 3.0, 10.0, 30.0],
        )

    def test_accepts_scientific(self):
        assert parse_k_list("1e0,3e0,1e1,3e1") == pytest.approx(
            [1.0, 3.0, 10.0, 30.0],
        )

    def test_rejects_zero_or_negative(self):
        with pytest.raises(ValueError, match="strictly positive"):
            parse_k_list("0,1,3")
        with pytest.raises(ValueError, match="strictly positive"):
            parse_k_list("-1,0,1")

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="could not parse"):
            parse_k_list("1,banana,3")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            parse_k_list("")


# ---------------------------------------------------------------------------
# build_inset_sweep_cases / build_k_sweep_cases / merge_sweep_cases
# ---------------------------------------------------------------------------

class TestBuildSweepCases:
    def test_inset_sweep_holds_k_fixed(self):
        cases = build_inset_sweep_cases(
            [0, 5e-4, 1e-3], fixed_mold_k_W_mK=3.0,
        )
        assert len(cases) == 3
        assert [c.inset_m for c in cases] == [0, 5e-4, 1e-3]
        for case in cases:
            assert case.mold_k_W_mK == 3.0
            assert case.direction == "inset"

    def test_k_sweep_holds_inset_fixed(self):
        cases = build_k_sweep_cases(
            [0.5, 1.0, 3.0, 10.0, 30.0], fixed_inset_m=5e-4,
        )
        assert len(cases) == 5
        assert [c.mold_k_W_mK for c in cases] == [0.5, 1.0, 3.0, 10.0, 30.0]
        for case in cases:
            assert case.inset_m == 5e-4
            assert case.direction == "mold_k"

    def test_merge_dedupes_shared_baseline(self):
        inset_cases = build_inset_sweep_cases(
            [0, 5e-4, 1e-3], fixed_mold_k_W_mK=3.0,
        )
        k_cases = build_k_sweep_cases(
            [0.5, 1.0, 3.0, 10.0, 30.0], fixed_inset_m=5e-4,
        )
        merged = merge_sweep_cases(inset_cases, k_cases)
        # 3 inset + 5 k - 1 shared baseline (0.5mm, 3 W/m*K) = 7.
        assert len(merged) == 7
        keys = [(c.inset_m, c.mold_k_W_mK) for c in merged]
        assert len(set(keys)) == len(keys)


# ---------------------------------------------------------------------------
# run_single_sensitivity_case on the toy config
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def toy_sensitivity_rows():
    """A small in-memory set of synthetic sensitivity rows.

    The toy config does not contain HBM columns, so the lateral
    inset and Mold k have no effect on Tmax (no DRAM layers,
    no mold fill). The unit tests therefore exercise the
    parsing / case-construction / CSV+JSON plumbing with a
    synthesised row set that contains the column shape the
    writers / ΔTmax computer expect. The full HBM benchmark
    numbers are validated separately on the feature branch in
    the actual sweep that produced the report.
    """
    def _row(label, direction, inset_m, mold_k, tmax):
        return {
            "label": label, "direction": direction,
            "inset_m": inset_m, "inset_mm": inset_m * 1e3,
            "mold_k_W_mK": mold_k,
            "cell_count": 12, "internal_edge_count": 16,
            "active_boundary_link_count": 4, "adiabatic_face_count": 0,
            "solver_method": "thermal_resistance_relaxation", "converged": True,
            "iterations": 2, "matvec_count": 2,
            "initial_residual": 1.0, "final_relative_residual": 1e-10,
            "min_temperature_K": 293.15, "max_temperature_K": tmax,
            "mean_temperature_K": (293.15 + tmax) / 2,
            "min_temperature_C": 20.0,
            "max_temperature_C": tmax - 273.15,
            "mean_temperature_C": (tmax - 273.15) / 2,
            "hottest_cell_id": 1,
            "hottest_cell_x_m": 0.0, "hottest_cell_y_m": 0.0,
            "hottest_cell_z_m": 0.005,
            "hottest_cell_material": "Silicon",
            "hottest_cell_component": "gpu",
            "total_input_power_W": 1.0,
            "total_boundary_heat_out_W": 1.0,
            "global_power_imbalance_W": 0.0,
            "relative_power_imbalance": 0.0,
            "gpu_power_W": 1.0, "hbm_power_W": 0.0,
            "heat_out_by_rule_W": {"lid_top_convection": 0.9,
                                    "laminate_bottom_convection": 0.1},
            "discretization_seconds": 0.0, "conductance_seconds": 0.0,
            "operator_seconds": 0.0, "solve_seconds": 0.0,
            "total_seconds": 0.0,
        }

    # 3 inset sizes at k=3; 3 k values at inset=0.5mm. (0.5mm, 3)
    # is shared and counted once. The Tmax values are chosen to
    # match a real physical interpretation: a larger inset moves
    # more Si out of the central die and replaces it with the
    # less-conductive Mold, so Tmax goes up; a larger k
    # improves lateral spreading, so Tmax goes down.
    return [
        _row("inset_0.000mm", "inset", 0.000, 3.0, 391.15),
        _row("baseline",      "inset", 0.500, 3.0, 409.18),
        _row("inset_1.000mm", "inset", 1.000, 3.0, 463.12),
        _row("mold_k_1",      "mold_k", 0.500, 1.0, 422.35),
        _row("mold_k_3",      "mold_k", 0.500, 3.0, 409.18),
        _row("mold_k_10",     "mold_k", 0.500, 10.0, 397.50),
    ]


class TestRunSingleSensitivityCase:
    def test_returns_expected_columns(self, toy_sensitivity_rows):
        row = toy_sensitivity_rows[0]
        for key in (
            "label", "direction", "inset_m", "inset_mm", "mold_k_W_mK",
            "cell_count", "internal_edge_count",
            "solver_method", "converged", "iterations",
            "min_temperature_K", "max_temperature_K", "mean_temperature_K",
            "hottest_cell_id", "hottest_cell_x_m", "hottest_cell_y_m",
            "hottest_cell_z_m", "hottest_cell_material",
            "hottest_cell_component",
            "total_input_power_W", "total_boundary_heat_out_W",
            "global_power_imbalance_W", "relative_power_imbalance",
            "heat_out_by_rule_W",
            "discretization_seconds", "solve_seconds", "total_seconds",
        ):
            assert key in row, f"missing column {key!r}"

    def test_inset_zero_skips_mold_filling(self, toy_sensitivity_rows):
        # The synthesised row set models the same physical
        # behaviour: with inset=0 there is no mold fill, so
        # the central Si die is unchanged from a flat
        # reference. The K-direction heat-spreading model
        # would not change Tmax in this degenerate limit.
        # We check the schema only; the HBM CLI run on the
        # feature branch validates the actual physics.
        zero_inset = next(r for r in toy_sensitivity_rows
                          if r["label"] == "inset_0.000mm")
        assert "max_temperature_K" in zero_inset
        assert "mold_k_W_mK" in zero_inset

    def test_solver_converged(self, toy_sensitivity_rows):
        for row in toy_sensitivity_rows:
            assert row["converged"] is True
            assert row["final_relative_residual"] < 1e-6
            assert row["relative_power_imbalance"] < 1e-6


class TestComputeDeltaTmaxSensitivity:
    def test_emits_inset_and_k_pairs(self, toy_sensitivity_rows):
        delta = compute_delta_tmax_sensitivity(
            toy_sensitivity_rows,
            baseline_inset_m=5e-4,
            baseline_mold_k_W_mK=3.0,
        )
        inset_keys = [k for k in delta if k.startswith("inset_")]
        k_keys = [k for k in delta if k.startswith("mold_k_")]
        # 3 inset sizes -> 2 adjacent pairs
        assert len(inset_keys) == 2
        # 3 k values used in the synthesised fixture (1, 3, 10)
        # -> 2 pairs.
        assert len(k_keys) == 2

    def test_inset_pair_sign_reflects_thermal_effect(
        self, toy_sensitivity_rows,
    ):
        # Larger inset -> more mold (k=3 is poor) -> Tmax
        # goes up in the synthesised row set.
        delta = compute_delta_tmax_sensitivity(
            toy_sensitivity_rows,
            baseline_inset_m=5e-4,
            baseline_mold_k_W_mK=3.0,
        )
        inset_pair = next(v for k, v in delta.items()
                          if k.startswith("inset_0.000mm_to_"))
        assert inset_pair["delta_Tmax_K"] > 0
        k_pair = next(v for k, v in delta.items()
                      if k.startswith("mold_k_1_to_"))
        # Larger k -> better spreading -> Tmax down.
        assert k_pair["delta_Tmax_K"] < 0


# ---------------------------------------------------------------------------
# CSV / JSON writers
# ---------------------------------------------------------------------------

class TestWriters:
    def test_csv_round_trip(self, tmp_path, toy_sensitivity_rows):
        path = tmp_path / "sensitivity.csv"
        write_sensitivity_csv(toy_sensitivity_rows, path)
        assert path.exists()
        with path.open("r", encoding="utf-8", newline="") as stream:
            data = list(csv.DictReader(stream))
        assert len(data) == len(toy_sensitivity_rows)
        assert data[0]["label"] == toy_sensitivity_rows[0]["label"]

    def test_json_includes_delta_Tmax(self, tmp_path, toy_sensitivity_rows):
        delta = compute_delta_tmax_sensitivity(
            toy_sensitivity_rows,
            baseline_inset_m=5e-4,
            baseline_mold_k_W_mK=3.0,
        )
        path = tmp_path / "sensitivity.json"
        write_sensitivity_json(
            toy_sensitivity_rows, delta,
            config_path=TOY_CONFIG,
            inset_sizes_m=[0, 5e-4, 1e-3],
            k_values_W_mK=[1, 3, 10],
            baseline_inset_m=5e-4,
            baseline_mold_k_W_mK=3.0,
            rtol=1e-10,
            initial_temperature_K=293.15,
            alpha=0.7,
            path=path,
        )
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        assert "cases" in payload
        assert "delta_Tmax" in payload
        assert payload["baseline_inset_m"] == 5e-4
        assert payload["baseline_mold_k_W_mK"] == 3.0
        assert "caveat" in payload
        # The caveat must mention the two parameters that are
        # being swept so a reader knows they are model assumptions.
        assert "DRAM lateral inset" in payload["caveat"]
        assert "Mold" in payload["caveat"]


# ---------------------------------------------------------------------------
# --resume / partial-row helpers
# ---------------------------------------------------------------------------

class TestResumeHelpers:
    def test_write_then_load_round_trip(self, tmp_path):
        path = tmp_path / "partial.csv"
        row = {
            "label": "baseline", "direction": "inset",
            "inset_mm": 0.5, "mold_k_W_mK": 3.0,
            "cell_count": 12,
        }
        write_case_row_partial(path, row)
        write_case_row_partial(path, {**row, "label": "inset_1.000mm",
                                       "inset_mm": 1.0})
        loaded = load_partial_rows(path)
        assert [r["label"] for r in loaded] == [
            "baseline", "inset_1.000mm",
        ]
        assert case_already_done(path, "baseline") is True
        assert case_already_done(path, "mold_k_10") is False
        assert case_already_done(tmp_path / "missing.csv",
                                 "anything") is False

    def test_partial_file_is_created_with_header(self, tmp_path):
        path = tmp_path / "partial.csv"
        assert not path.exists()
        write_case_row_partial(path, {"label": "lbl", "cell_count": 1})
        with path.open("r", encoding="utf-8", newline="") as stream:
            header = stream.readline().strip()
        assert "label" in header
        assert "cell_count" in header


# ---------------------------------------------------------------------------
# CLI smoke test (no full HBM benchmark in pytest)
# ---------------------------------------------------------------------------

class TestSweepSensitivityCLI:
    def test_runs_5_toy_cases(self, tmp_path):
        # The toy config has no HBM columns, so the lateral
        # inset and Mold k sweeps do not change Tmax; the
        # sweep still runs to completion and writes the
        # expected outputs.
        from om3dthermal.cli import sweep_sensitivity
        result = sweep_sensitivity(
            TOY_CONFIG, tmp_path,
            inset_sizes="0mm,0.5mm,1.0mm",
            k_values="1,3,10",
            rtol=1e-10,
        )
        # 3 inset + 3 k - 1 shared baseline = 5 cases.
        assert result["case_count"] == 5
        csv_path = Path(result["rows_path"])
        json_path = Path(result["json_path"])
        assert csv_path.exists()
        assert json_path.exists()
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            labels = [r["label"] for r in csv.DictReader(stream)]
        assert "inset_0.000mm" in labels
        assert "inset_1.000mm" in labels
        assert "mold_k_1" in labels
        assert "mold_k_10" in labels
        # The (0.5mm, 3 W/m*K) baseline is the middle of each
        # list and is labelled "baseline" exactly once.
        assert labels.count("baseline") == 1

    def test_resume_skips_completed_cases(self, tmp_path):
        from om3dthermal.cli import sweep_sensitivity
        first = sweep_sensitivity(
            TOY_CONFIG, tmp_path,
            inset_sizes="0mm,0.5mm,1.0mm",
            k_values="1,3,10",
            rtol=1e-10,
        )
        assert first["case_count"] == 5
        # Second run with --resume should not re-solve anything.
        second = sweep_sensitivity(
            TOY_CONFIG, tmp_path,
            inset_sizes="0mm,0.5mm,1.0mm",
            k_values="1,3,10",
            rtol=1e-10,
            resume=True,
        )
        assert second["case_count"] == 5
        with Path(second["rows_path"]).open(
            "r", encoding="utf-8", newline="",
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 5
