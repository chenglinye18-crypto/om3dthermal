"""Unit tests for the steady-state mesh-convergence sweep.

The full HBM-on-GPU benchmark is too large for pytest. The
canonical 5-case sweep is exercised here on a tiny toy config
(``tests/fixtures/toy_1box.yaml``) so the test stays under a
second; the HBM benchmark numbers are validated separately on the
feature branch in the actual run that produced the report.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from om3dthermal.config import load_config
from om3dthermal.case_runner import run_steady_pipeline
from om3dthermal.mesh_convergence import (
    CaseSpec,
    build_sweep_cases,
    case_already_done,
    compute_delta_tmax,
    load_partial_rows,
    parse_mesh_sizes,
    run_single_case,
    write_case_row_partial,
    write_mesh_convergence_csv,
    write_mesh_convergence_json,
)


TOY_CONFIG = Path(__file__).parent / "fixtures" / "toy_1box.yaml"


# ---------------------------------------------------------------------------
# parse_mesh_sizes
# ---------------------------------------------------------------------------

class TestParseMeshSizes:
    def test_accepts_mm(self):
        assert parse_mesh_sizes("1.0mm,0.5mm,0.25mm") == pytest.approx(
            [1e-3, 5e-4, 2.5e-4],
        )

    def test_accepts_um(self):
        # Pint's unit conversion introduces float noise
        # (``parse_length("200um")`` = 1.999...e-4) so compare
        # with ``pytest.approx``.
        assert parse_mesh_sizes("200um,100um,50um") == pytest.approx(
            [200e-6, 100e-6, 50e-6],
        )

    def test_accepts_mixed_units(self):
        # 0.001 m = 1 mm, 0.5 mm = 500 um, 250 um = 0.25 mm.
        # Use three distinct sizes so the strict-decreasing
        # check still passes.
        assert parse_mesh_sizes("0.001m,0.5mm,250um") == pytest.approx(
            [1e-3, 5e-4, 2.5e-4],
        )

    def test_rejects_non_decreasing(self):
        with pytest.raises(ValueError, match="strictly decreasing"):
            parse_mesh_sizes("1.0mm,2.0mm,3.0mm")

    def test_rejects_equal(self):
        with pytest.raises(ValueError, match="strictly decreasing"):
            parse_mesh_sizes("1.0mm,1.0mm")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            parse_mesh_sizes("")

    def test_rejects_unparseable_token(self):
        with pytest.raises(ValueError, match="could not parse"):
            parse_mesh_sizes("1.0mm,banana,0.5mm")


# ---------------------------------------------------------------------------
# build_sweep_cases
# ---------------------------------------------------------------------------

class TestBuildSweepCases:
    def test_5_case_shape_with_3_x_3(self):
        cases = build_sweep_cases(
            [1e-3, 5e-4, 2.5e-4], [200e-6, 100e-6, 50e-6],
        )
        labels = [c.label for c in cases]
        # 3 xy cases, 3 z cases minus the shared baseline (0.5mm, 100um)
        # = 5 total.
        assert len(cases) == 5
        assert labels == [
            "xy_1.0000mm",
            "xy_0.5000mm",
            "xy_0.2500mm",
            "z_200.00um",
            "z_50.00um",
        ]

    def test_xy_cases_hold_z_fixed_at_middle(self):
        cases = build_sweep_cases(
            [1e-3, 5e-4, 2.5e-4], [200e-6, 100e-6, 50e-6],
        )
        for c in cases:
            if c.label.startswith("xy_"):
                assert c.dz_m == 100e-6, c

    def test_z_cases_hold_xy_fixed_at_middle(self):
        cases = build_sweep_cases(
            [1e-3, 5e-4, 2.5e-4], [200e-6, 100e-6, 50e-6],
        )
        for c in cases:
            if c.label.startswith("z_"):
                assert c.dx_m == 5e-4, c
                assert c.dy_m == 5e-4, c

    def test_baseline_is_deduplicated(self):
        # xy_0.5mm @ dz=100um and z_100um @ dxy=0.5mm are the same
        # physical case; only one of them should appear in the
        # case list.
        cases = build_sweep_cases(
            [1e-3, 5e-4, 2.5e-4], [200e-6, 100e-6, 50e-6],
        )
        keys = [c.tuple for c in cases]
        assert len(keys) == len(set(keys))

    def test_5_case_shape_with_2_x_2(self):
        # Only two sizes per direction; shared baseline is the only
        # overlap.
        cases = build_sweep_cases(
            [1e-3, 5e-4], [200e-6, 100e-6],
        )
        labels = [c.label for c in cases]
        assert labels == [
            "xy_1.0000mm", "xy_0.5000mm", "z_200.00um",
        ]


# ---------------------------------------------------------------------------
# run_single_case + compute_delta_tmax on the toy config
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def toy_config():
    return load_config(TOY_CONFIG)


@pytest.fixture(scope="module")
def toy_case_rows(toy_config):
    """Run the canonical 5-case sweep on the toy config once and
    share the per-case rows across tests in this module."""
    xy_list = parse_mesh_sizes("10mm,5mm,2.5mm")
    z_list = parse_mesh_sizes("10mm,5mm,2.5mm")
    cases = build_sweep_cases(xy_list, z_list)
    rows = []
    for spec in cases:
        rows.append(run_single_case(toy_config, spec, rtol=1e-10))
    return rows, xy_list, z_list


class TestRunSingleCase:
    def test_returns_expected_columns(self, toy_config):
        spec = CaseSpec(
            label="xy_5.0000mm", dx_m=5e-3, dy_m=5e-3, dz_m=5e-3,
        )
        row = run_single_case(toy_config, spec, rtol=1e-10)
        for key in (
            "label", "dx_m", "dy_m", "dz_m",
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

    def test_mesh_override_actually_changes_cell_count(self, toy_config):
        coarse = run_single_case(
            toy_config,
            CaseSpec(label="xy_10.0000mm", dx_m=10e-3, dy_m=10e-3,
                     dz_m=5e-3),
            rtol=1e-10,
        )
        fine = run_single_case(
            toy_config,
            CaseSpec(label="xy_2.5000mm", dx_m=2.5e-3, dy_m=2.5e-3,
                     dz_m=5e-3),
            rtol=1e-10,
        )
        # Refining 4x in each lateral direction should produce
        # significantly more cells.
        assert fine["cell_count"] > coarse["cell_count"]

    def test_solver_converged(self, toy_config):
        row = run_single_case(
            toy_config,
            CaseSpec(label="xy_5.0000mm", dx_m=5e-3, dy_m=5e-3,
                     dz_m=5e-3),
            rtol=1e-10,
        )
        assert row["converged"] is True
        assert row["final_relative_residual"] < 1e-6
        assert row["relative_power_imbalance"] < 1e-6


class TestComputeDeltaTmax:
    def test_produces_four_pairs_for_canonical_sweep(
        self, toy_case_rows,
    ):
        rows, xy_list, z_list = toy_case_rows
        deltas = compute_delta_tmax(
            rows, xy_sizes_m=xy_list, z_sizes_m=z_list,
        )
        keys = list(deltas.keys())
        # xy has 2 adjacent pairs (3 sizes), z has 2 adjacent pairs
        # (3 sizes).
        assert sum(1 for k in keys if k.startswith("xy_")) == 2
        assert sum(1 for k in keys if k.startswith("z_")) == 2

    def test_uses_shared_baseline_for_z_pairs(self, toy_case_rows):
        rows, xy_list, z_list = toy_case_rows
        deltas = compute_delta_tmax(
            rows, xy_sizes_m=xy_list, z_sizes_m=z_list,
        )
        # The two z deltas should chain through the shared baseline
        # (0.5mm / 5mm); the fine value of the first z delta and
        # the coarse value of the second z delta must agree.
        z_keys = sorted(k for k in deltas if k.startswith("z_"))
        if len(z_keys) >= 2:
            first = deltas[z_keys[0]]
            second = deltas[z_keys[1]]
            assert math.isclose(
                first["fine_max_temperature_K"],
                second["coarse_max_temperature_K"],
                rel_tol=1e-12,
            )

    def test_delta_keys_are_monotonically_refining(self, toy_case_rows):
        rows, xy_list, z_list = toy_case_rows
        deltas = compute_delta_tmax(
            rows, xy_sizes_m=xy_list, z_sizes_m=z_list,
        )
        for entry in deltas.values():
            coarse, fine = entry["coarse_label"], entry["fine_label"]
            # Coarse should be larger than fine in its own direction.
            if entry["direction"] == "xy":
                c = float(coarse.split("_", 1)[1].rstrip("mm"))
                f = float(fine.split("_", 1)[1].rstrip("mm"))
                assert c > f
            else:
                c = float(coarse.split("_", 1)[1].rstrip("um"))
                f = float(fine.split("_", 1)[1].rstrip("um"))
                assert c > f


# ---------------------------------------------------------------------------
# CSV / JSON writers
# ---------------------------------------------------------------------------

class TestWriters:
    def test_csv_round_trip(self, tmp_path, toy_case_rows):
        rows, _, _ = toy_case_rows
        path = tmp_path / "mesh_convergence.csv"
        write_mesh_convergence_csv(rows, path)
        assert path.exists()
        with path.open("r", encoding="utf-8", newline="") as stream:
            data = list(csv.DictReader(stream))
        assert len(data) == len(rows)
        assert data[0]["label"] == rows[0]["label"]

    def test_json_includes_delta_tmax(self, tmp_path, toy_case_rows):
        rows, xy_list, z_list = toy_case_rows
        deltas = compute_delta_tmax(
            rows, xy_sizes_m=xy_list, z_sizes_m=z_list,
        )
        path = tmp_path / "mesh_convergence.json"
        write_mesh_convergence_json(
            rows, deltas,
            config_path=TOY_CONFIG,
            xy_sizes_m=xy_list,
            z_sizes_m=z_list,
            rtol=1e-10,
            initial_temperature_K=293.15,
            method="pcg",
            path=path,
        )
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        assert "cases" in payload
        assert "delta_Tmax" in payload
        assert len(payload["cases"]) == len(rows)
        # Hot cells and the per-rule heat-out are preserved.
        for case in payload["cases"]:
            assert "hottest_cell_id" in case
            assert "heat_out_by_rule_W" in case
            assert isinstance(case["heat_out_by_rule_W"], dict)
        assert payload["benchmark_label"].startswith(
            "paper-parameter-aligned uniform-power baseline")


# ---------------------------------------------------------------------------
# --resume / partial-row helpers
# ---------------------------------------------------------------------------

class TestResumeHelpers:
    def test_write_then_load_round_trip(self, tmp_path):
        path = tmp_path / "partial.csv"
        row = {
            "label": "xy_5.0000mm",
            "dx_mm": 5.0,
            "dy_mm": 5.0,
            "dz_um": 5e3,
            "cell_count": 12,
        }
        write_case_row_partial(path, row)
        write_case_row_partial(path, {**row, "label": "z_5000.00um",
                                       "dz_um": 5e3, "dx_mm": 5.0,
                                       "dy_mm": 5.0})
        loaded = load_partial_rows(path)
        assert [r["label"] for r in loaded] == [
            "xy_5.0000mm", "z_5000.00um",
        ]
        assert case_already_done(path, "xy_5.0000mm") is True
        assert case_already_done(path, "z_200.00um") is False
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
# CLI smoke test (no full HBM benchmark)
# ---------------------------------------------------------------------------

class TestSweepMeshCLI:
    def test_runs_5_toy_cases(self, tmp_path):
        from om3dthermal.cli import sweep_mesh
        result = sweep_mesh(
            TOY_CONFIG, tmp_path,
            xy_sizes="10mm,5mm,2.5mm",
            z_sizes="10mm,5mm,2.5mm",
            rtol=1e-10,
        )
        assert result["case_count"] == 5
        csv_path = Path(result["rows_path"])
        json_path = Path(result["json_path"])
        assert csv_path.exists()
        assert json_path.exists()
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            labels = [r["label"] for r in csv.DictReader(stream)]
        assert labels == [
            "xy_10.0000mm", "xy_5.0000mm", "xy_2.5000mm",
            "z_10000.00um", "z_2500.00um",
        ]

    def test_resume_skips_completed_cases(self, tmp_path):
        from om3dthermal.cli import sweep_mesh
        # First run.
        first = sweep_mesh(
            TOY_CONFIG, tmp_path,
            xy_sizes="10mm,5mm,2.5mm",
            z_sizes="10mm,5mm,2.5mm",
            rtol=1e-10,
        )
        assert first["case_count"] == 5
        # Second run with --resume should not re-solve anything.
        second = sweep_mesh(
            TOY_CONFIG, tmp_path,
            xy_sizes="10mm,5mm,2.5mm",
            z_sizes="10mm,5mm,2.5mm",
            rtol=1e-10,
            resume=True,
        )
        assert second["case_count"] == 5
        with Path(second["rows_path"]).open(
            "r", encoding="utf-8", newline="",
        ) as stream:
            rows = list(csv.DictReader(stream))
        # No duplicate rows; the CSV still has exactly 5 cases.
        assert len(rows) == 5
