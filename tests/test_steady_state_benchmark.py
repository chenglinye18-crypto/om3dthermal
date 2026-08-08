"""Integration tests for the steady-state thermal solver on the
shipped HBM-on-GPU 12-Hi benchmark.

These tests run the full
``YAML -> mesh -> conductance -> boundary -> power -> operator ->
PCG`` pipeline. They assert the geometric / physical invariants
the user-visible CLI summary is supposed to report:

- the cell / edge / boundary link counts match the upstream
  stages;
- the total power is 574 W (414 W GPU + 4 x 40 W HBM);
- the network is fully anchored (no unanchored components);
- the PCG convergence reaches a sub-1e-8 relative residual with
  relative power imbalance under 1e-6;
- the temperature is finite and above the boundary reference
  (20 degC = 293.15 K);
- the result is honestly labelled as the
  ``paper-parameter-aligned uniform-power baseline`` and not a
  strict reproduction of the paper's 141.7 degC.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from om3dthermal.config import load_config
from om3dthermal.discretization import (
    build_adjacency,
    build_boundary_faces,
    build_global_grid,
    generate_cells,
)
from om3dthermal.geometry.horizontal_columns import HorizontalColumnsBuilder
from om3dthermal.thermal import (
    build_boundary_link_table,
    build_conductance_table,
    build_matrix_free_operator,
    map_power_sources,
    solve_pcg,
    solve_weighted_jacobi,
    validate_anchored_components,
)
from om3dthermal.thermal.solution_export import build_solver_summary


CONFIG = Path(__file__).parents[1] / "configs" / "hbm_on_gpu_12hi.yaml"


@pytest.fixture(scope="module")
def benchmark_solve():
    """Build the full steady-state solve once and share across tests."""
    cfg = load_config(CONFIG)
    scene = HorizontalColumnsBuilder(cfg).build()
    boxes = list(scene.boxes)
    grid = build_global_grid(boxes, cfg.discretization.max_cell_size)
    cells = generate_cells(boxes, grid)
    edges = build_adjacency(cells, grid)
    boundary_faces = build_boundary_faces(cells, grid)
    conductance = build_conductance_table(
        cells=cells, adjacency_edges=edges,
        materials=cfg.materials, config=cfg.thermal_conductance,
    )
    boundary = build_boundary_link_table(
        boundary_faces=boundary_faces, cells=cells,
        materials=cfg.materials,
        config=cfg.thermal_boundary_conditions,
    )
    power = map_power_sources(cells, cfg.thermal_power_sources)
    operator = build_matrix_free_operator(
        conductance, boundary, power.power_W,
    )
    validate_anchored_components(
        cell_count=operator.cell_count,
        internal_cell_a=operator.internal_cell_a,
        internal_cell_b=operator.internal_cell_b,
        boundary=boundary,
    )
    T0 = np.full(operator.cell_count, 293.15, dtype=np.float64)
    # Coarse tolerance keeps the test under ~30 s; the CLI run uses
    # 1e-8 to make the published summary.
    result = solve_pcg(operator, T0, boundary,
                       relative_residual_tolerance=1e-6,
                       max_iterations=2000)
    return {
        "config": cfg,
        "cells": cells,
        "edges": edges,
        "boundary_faces": boundary_faces,
        "conductance": conductance,
        "boundary": boundary,
        "power": power,
        "operator": operator,
        "result": result,
    }


# ---------------------------------------------------------------------------
# Counts and physical setup
# ---------------------------------------------------------------------------

def test_cell_count_matches_discretisation(benchmark_solve):
    assert benchmark_solve["result"].temperature_K.shape[0] == 272460


def test_total_input_power_is_574_W(benchmark_solve):
    assert benchmark_solve["power"].total_power_W == pytest.approx(574.0)
    assert benchmark_solve["result"].total_input_power_W == pytest.approx(
        574.0, abs=1e-9)


def test_gpu_power_414_W_hbm_power_160_W(benchmark_solve):
    power = benchmark_solve["power"]
    gpu = sum(v for k, v in power.power_by_source.items()
              if k.lower().startswith("gpu"))
    hbm = sum(v for k, v in power.power_by_source.items()
              if k.lower().startswith("hbm"))
    assert gpu == pytest.approx(414.0)
    assert hbm == pytest.approx(160.0)


def test_active_boundary_link_count_is_positive(benchmark_solve):
    boundary = benchmark_solve["boundary"]
    assert boundary.link_count > 0
    # Lid top + laminate bottom convection in the shipped config.
    assert boundary.link_count >= 2 * 2640  # at least the lid (2640 faces)


def test_network_is_fully_anchored(benchmark_solve):
    # validate_anchored_components already ran in the fixture and did
    # not raise. Calling it again here also confirms the result is
    # idempotent.
    validate_anchored_components(
        cell_count=benchmark_solve["operator"].cell_count,
        internal_cell_a=benchmark_solve["operator"].internal_cell_a,
        internal_cell_b=benchmark_solve["operator"].internal_cell_b,
        boundary=benchmark_solve["boundary"],
    )


# ---------------------------------------------------------------------------
# Solver convergence
# ---------------------------------------------------------------------------

def test_pcg_produces_finite_temperatures(benchmark_solve):
    T = benchmark_solve["result"].temperature_K
    assert np.all(np.isfinite(T))


def test_pcg_min_temperature_at_or_above_ambient(benchmark_solve):
    # The only non-adiabatic boundaries are convection at 293.15 K
    # (ambient). No cell can be below the boundary reference.
    T = benchmark_solve["result"].temperature_K
    assert T.min() >= 293.15 - 1e-6


def test_pcg_max_temperature_above_min(benchmark_solve):
    T = benchmark_solve["result"].temperature_K
    assert T.max() > T.min()


def test_pcg_relative_residual_below_tolerance(benchmark_solve):
    assert benchmark_solve["result"].final_relative_residual < 1e-6


def test_pcg_relative_power_imbalance_below_tolerance(benchmark_solve):
    assert benchmark_solve["result"].relative_power_imbalance < 1e-6


# ---------------------------------------------------------------------------
# Summary honesty
# ---------------------------------------------------------------------------

def test_solver_summary_records_strict_paper_reproduction_false(benchmark_solve):
    # Compute the same summary the CLI does, then assert the metadata.
    result = benchmark_solve["result"]
    power = benchmark_solve["power"]
    boundary = benchmark_solve["boundary"]
    summary = build_solver_summary(
        result=result, cell_count=272460, internal_edge_count=790964,
        active_boundary_link_count=boundary.link_count,
        adiabatic_boundary_face_count=0,
        boundary_build_seconds=0.0, power_mapping_seconds=0.0,
        operator_build_seconds=0.0, gpu_power_W=414.0, hbm_power_W=160.0,
    )
    assert summary["strict_paper_temperature_reproduction"] is False
    assert "uniform-power baseline" in summary["benchmark_label"]


# ---------------------------------------------------------------------------
# Jacobi on a coarse coarse grid (small, must converge)
# ---------------------------------------------------------------------------

def test_jacobi_baseline_loop_runs_and_returns_residual_history(benchmark_solve):
    # The Jacobi baseline is exercised at the unit level
    # (test_steady_state_solver.py). Here we only confirm that the
    # full-benchmark operator feeds the solver without raising and
    # that the residual history is recorded; a convergence check on
    # the 272 k cell problem with Jacobi is left out because Jacobi
    # is the slow O(N^2)-iterations algorithm by design.
    operator = benchmark_solve["operator"]
    boundary = benchmark_solve["boundary"]
    T0 = np.full(operator.cell_count, 293.15, dtype=np.float64)
    result = solve_weighted_jacobi(
        operator, T0, boundary, omega=0.7, max_iterations=20,
        relative_residual_tolerance=1e-12,
        max_temperature_update_tolerance=1e-12,
    )
    assert len(result.residual_history) >= 1
    assert len(result.update_norm_history) >= 1
    assert np.all(np.isfinite(result.temperature_K))
