"""Shared end-to-end steady-state pipeline runner.

Both ``om3dthermal.cli.solve_steady`` and the per-case runners
in ``om3dthermal.sensitivity`` need to run the same sequence

    SimulationConfig
      -> geometry
      -> mesh
      -> conductance / boundary links / power
      -> matrix-free operator
      -> PCG (or weighted Jacobi)

without ever materialising a dense matrix. This module isolates
that pipeline as a single function so the two call sites cannot
drift.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import SimulationConfig
from .discretization import (
    build_adjacency,
    build_boundary_faces,
    build_global_grid,
    generate_cells,
    validate_cell_surface_partition,
    validate_volume_conservation,
)
from .geometry.horizontal_columns import HorizontalColumnsBuilder
from .thermal import (
    PowerVector,
    build_boundary_link_table,
    build_conductance_table,
    build_matrix_free_operator,
    map_power_sources,
    solve_pcg,
    solve_weighted_jacobi,
    validate_anchored_components,
)
from .thermal.boundary import BoundaryLinkTable, select_boundary_rule
from .thermal.operator import MatrixFreeThermalOperator
from .thermal.steady_state import SteadyStateResult


@dataclass
class PipelineResult:
    """Everything produced by one end-to-end solve, plus timing."""

    config: SimulationConfig
    result: SteadyStateResult
    cells: list
    edges: list
    boundary_faces: list
    boundary_table: BoundaryLinkTable
    power: PowerVector
    operator: MatrixFreeThermalOperator
    # Per-stage wall time (seconds).
    discretization_seconds: float
    conductance_seconds: float
    operator_seconds: float
    solve_seconds: float
    # Aggregates that are useful to both the per-case writeout and
    # any sensitivity summary.
    cell_count: int
    internal_edge_count: int
    active_boundary_link_count: int
    adiabatic_face_count: int
    gpu_power_W: float
    hbm_power_W: float
    # Per-source / per-rule breakdown.
    power_by_source: dict[str, float]
    heat_out_by_rule_W: dict[str, float]
    hottest_cell_id: int
    hottest_cell_xyz_m: tuple[float, float, float]
    hottest_cell_material: str
    hottest_cell_component: str


def _heat_out_by_rule(
    boundary: BoundaryLinkTable,
    result: SteadyStateResult,
    config: SimulationConfig,
    cells_by_id: dict,
) -> dict[str, float]:
    """Per-rule boundary heat outflow (W), keyed by rule name."""
    if config.thermal_boundary_conditions is None:
        return {}
    T = result.temperature_K
    rules = config.thermal_boundary_conditions.rules
    out: dict[str, float] = {}
    if boundary.link_count == 0:
        return out
    fluxes = boundary.conductance_W_K * (
        T[boundary.cell_id] - boundary.reference_temperature_K
    )
    for idx in np.unique(boundary.rule_index):
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= len(rules):
            continue
        name = rules[idx_i].name
        mask = boundary.rule_index == idx_i
        out[name] = float(np.sum(fluxes[mask]))
    return out


def run_steady_pipeline(
    config: SimulationConfig,
    *,
    method: str = "pcg",
    omega: float = 0.7,
    rtol: float = 1e-8,
    max_iterations: int = 10_000,
    initial_temperature_K: float = 293.15,
) -> PipelineResult:
    """Run the full steady-state pipeline and return all
    artifacts. The caller is responsible for writing any
    per-case output files."""
    if config.discretization is None:
        raise ValueError(
            "config has no 'discretization' block; add one before running "
            "the steady-state pipeline")
    if config.thermal_conductance is None:
        raise ValueError(
            "config has no 'thermal_conductance' block; add one before "
            "running the steady-state pipeline")
    if config.thermal_boundary_conditions is None:
        raise ValueError(
            "config has no 'thermal_boundary_conditions' block; add one "
            "before running the steady-state pipeline")
    if config.thermal_power_sources is None:
        raise ValueError(
            "config has no 'thermal_power_sources' block; add one before "
            "running the steady-state pipeline")

    # Geometry.
    scene = HorizontalColumnsBuilder(config).build()
    boxes = list(scene.boxes)

    # Discretise.
    t0 = time.perf_counter()
    grid = build_global_grid(boxes, config.discretization.max_cell_size)
    cells = generate_cells(boxes, grid)
    edges = build_adjacency(cells, grid)
    boundary_faces = build_boundary_faces(cells, grid)
    validate_volume_conservation(cells, boxes)
    validate_cell_surface_partition(cells, edges, boundary_faces)
    t1 = time.perf_counter()

    # Conductance + boundary links + power.
    t2 = time.perf_counter()
    conductance_table = build_conductance_table(
        cells=cells, adjacency_edges=edges,
        materials=config.materials,
        config=config.thermal_conductance,
    )
    boundary_table = build_boundary_link_table(
        boundary_faces=boundary_faces, cells=cells,
        materials=config.materials,
        config=config.thermal_boundary_conditions,
    )
    power = map_power_sources(cells=cells, config=config.thermal_power_sources)
    t3 = time.perf_counter()

    # Operator + anchored check.
    t4 = time.perf_counter()
    operator = build_matrix_free_operator(
        conductance=conductance_table, boundary=boundary_table,
        power_W=power.power_W,
    )
    validate_anchored_components(
        cell_count=operator.cell_count,
        internal_cell_a=operator.internal_cell_a,
        internal_cell_b=operator.internal_cell_b,
        boundary=boundary_table,
    )
    t5 = time.perf_counter()

    # Solve.
    initial_T = np.full(operator.cell_count, initial_temperature_K,
                        dtype=np.float64)
    if method == "pcg":
        result = solve_pcg(
            operator, initial_T, boundary_table,
            relative_residual_tolerance=rtol,
            max_iterations=max_iterations,
        )
    elif method == "jacobi":
        result = solve_weighted_jacobi(
            operator, initial_T, boundary_table,
            omega=omega,
            relative_residual_tolerance=rtol,
            max_iterations=max_iterations,
        )
    else:
        raise ValueError(
            f"unknown method {method!r}; expected 'pcg' or 'jacobi'")

    # Power-by-source breakdown.
    gpu_power = 0.0
    hbm_power = 0.0
    for source_name, distributed in power.power_by_source.items():
        if source_name.lower().startswith("gpu"):
            gpu_power += distributed
        elif source_name.lower().startswith("hbm"):
            hbm_power += distributed

    # Per-rule boundary heat outflow.
    cells_by_id = {c.id: c for c in cells}
    heat_out = _heat_out_by_rule(boundary_table, result, config, cells_by_id)
    adiabatic_face_count = sum(
        1 for f in boundary_faces
        if not _face_matches_rule(f, cells_by_id[f.cell_id], config)
    )

    # Hottest cell.
    T = result.temperature_K
    hottest_idx = int(np.argmax(T))
    hottest_cell = cells_by_id[cells[hottest_idx].id]
    hottest_xyz = (
        float(hottest_cell.center_x),
        float(hottest_cell.center_y),
        float(hottest_cell.center_z),
    )

    return PipelineResult(
        config=config,
        result=result,
        cells=cells,
        edges=edges,
        boundary_faces=boundary_faces,
        boundary_table=boundary_table,
        power=power,
        operator=operator,
        discretization_seconds=t1 - t0,
        conductance_seconds=t3 - t2,
        operator_seconds=t5 - t4,
        solve_seconds=float(result.solve_seconds),
        cell_count=len(cells),
        internal_edge_count=len(edges),
        active_boundary_link_count=boundary_table.link_count,
        adiabatic_face_count=adiabatic_face_count,
        gpu_power_W=gpu_power,
        hbm_power_W=hbm_power,
        power_by_source=dict(power.power_by_source),
        heat_out_by_rule_W=heat_out,
        hottest_cell_id=cells[hottest_idx].id,
        hottest_cell_xyz_m=hottest_xyz,
        hottest_cell_material=str(hottest_cell.material),
        hottest_cell_component=str(
            hottest_cell.component
            if hasattr(hottest_cell, "component") else ""
        ),
    )


def _face_matches_rule(face, cell, config) -> bool:
    """Heuristic for the adiabatic-face count: a face is considered
    *covered by a rule* (and therefore non-adiabatic) iff
    :func:`select_boundary_rule` returns a match."""
    if config.thermal_boundary_conditions is None:
        return False
    return select_boundary_rule(
        face, cell,
        config.thermal_boundary_conditions.rules) is not None
