"""Shared end-to-end steady-state pipeline runner.

The CLI's ``solve-steady`` command and the mesh-convergence sweep
both need to run the same sequence

    config
      -> geometry
      -> mesh
      -> conductance / boundary links / power
      -> matrix-free operator
      -> thermal-resistance-network relaxation (CPU or GPU)

without ever materialising a dense matrix. This module isolates that
pipeline as a single function so the two call sites cannot drift.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import CellSizeConfig, DiscretizationConfig, SimulationConfig
from .discretization import (
    build_adjacency,
    build_boundary_faces,
    build_global_grid,
    generate_cells,
    validate_cell_surface_partition,
    validate_volume_conservation,
)
from .geometry.horizontal_columns import HorizontalColumnsBuilder
from .geometry.orthogonal_hbm import OrthogonalHBMBuilder
from .thermal import (
    PowerVector,
    build_boundary_link_table,
    build_conductance_table,
    build_matrix_free_operator,
    map_power_sources,
    solve_thermal_resistance_relaxation,
    solve_thermal_resistance_relaxation_gpu,
    validate_anchored_components,
)
from .thermal.boundary import BoundaryLinkTable
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
    # the mesh-convergence sweep summary.
    cell_count: int
    internal_edge_count: int
    active_boundary_link_count: int
    adiabatic_face_count: int
    gpu_power_W: float
    hbm_power_W: float
    max_cell_size_m: tuple[float, float, float]
    # Per-source / per-rule breakdown for the summary.
    power_by_source: dict[str, float]
    heat_out_by_rule_W: dict[str, float]
    hottest_cell_id: int
    hottest_cell_xyz_m: tuple[float, float, float]
    hottest_cell_material: str
    hottest_cell_component: str


def _override_discretization(
    config: SimulationConfig,
    max_cell_size_m: tuple[float, float, float],
) -> SimulationConfig:
    """Return a deep-copied config with ``discretization.max_cell_size``
    replaced by ``max_cell_size_m``. The original config (and the
    on-disk YAML it was loaded from) is left untouched, so the
    mesh-convergence sweep can run 5+ cases without writing any
    intermediate YAML."""
    if config.discretization is None:
        raise ValueError(
            "config has no 'discretization' block; add one before running "
            "the steady-state pipeline")
    dx, dy, dz = max_cell_size_m
    new_max = CellSizeConfig(x=dx, y=dy, z=dz)
    new_discr = DiscretizationConfig(
        max_cell_size=new_max,
        preserve_box_boundaries=config.discretization.preserve_box_boundaries,
    )
    return config.model_copy(update={"discretization": new_discr})


def _heat_out_by_rule(
    boundary: BoundaryLinkTable,
    result: SteadyStateResult,
    config: SimulationConfig,
    cells_by_id: dict,
) -> dict[str, float]:
    """Per-rule boundary heat outflow (W), keyed by rule name.

    Only rules that have at least one link in the active table
    contribute; rules that produced zero links are omitted.
    """
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


def _face_matches_rule(face, cell, config) -> bool:
    """Heuristic for the adiabatic-face count: a face is considered
    *covered by a rule* (and therefore non-adiabatic) iff
    :func:`select_boundary_rule` returns a match."""
    from .thermal.boundary import select_boundary_rule
    if config.thermal_boundary_conditions is None:
        return False
    return select_boundary_rule(
        face, cell,
        config.thermal_boundary_conditions.rules) is not None


def run_steady_pipeline(
    config: SimulationConfig,
    *,
    max_cell_size_m: tuple[float, float, float] | None = None,
    alpha: float = 0.7,
    rtol: float = 1e-8,
    max_delta_t_K: float = 1e-6,
    max_iterations: int = 100_000,
    check_interval: int = 10,
    initial_temperature_K: float = 293.15,
    backend: str = "cpu",
) -> PipelineResult:
    """Run the full steady-state pipeline and return all artifacts.

    ``max_cell_size_m`` is a 3-tuple ``(dx, dy, dz)`` in metres that
    overrides ``config.discretization.max_cell_size`` for this run.
    Pass ``None`` (the default) to use whatever the config declares.

    ``backend`` selects between two implementations of the
    *same* thermal-resistance-network relaxation equation:

    * ``"cpu"`` (default) calls
      :func:`solve_thermal_resistance_relaxation` (NumPy on host).
    * ``"gpu"`` calls
      :func:`solve_thermal_resistance_relaxation_gpu` (CuPy / NVRTC).

    Convergence is measured on two physical quantities:
    ``relative_heat_flow_residual`` and ``max_abs_delta_T``.  Both
    must drop below the configured tolerance at the same
    ``check_interval`` boundary.
    """
    if backend not in {"cpu", "gpu"}:
        raise ValueError(
            f"unknown backend {backend!r}; expected 'cpu' or 'gpu'")
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
    if max_cell_size_m is not None:
        config = _override_discretization(config, max_cell_size_m)

    # Geometry.
    scene = (
        OrthogonalHBMBuilder(config).build()
        if config.orthogonal_hbm is not None
        else HorizontalColumnsBuilder(config).build())
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
    if backend == "cpu":
        result = solve_thermal_resistance_relaxation(
            operator, initial_T, boundary_table,
            alpha=alpha,
            relative_residual_tolerance=rtol,
            max_temperature_update_tolerance=max_delta_t_K,
            max_iterations=max_iterations,
            check_interval=check_interval,
        )
    else:
        result = solve_thermal_resistance_relaxation_gpu(
            operator, initial_T, boundary_table,
            alpha=alpha,
            relative_residual_tolerance=rtol,
            max_temperature_update_tolerance=max_delta_t_K,
            max_iterations=max_iterations,
            check_interval=check_interval,
        )

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

    discr = config.discretization.max_cell_size
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
        max_cell_size_m=(float(discr.x), float(discr.y), float(discr.z)),
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
