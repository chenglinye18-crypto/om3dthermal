"""Geometry, power mapping, cached operator, and FP64 GPU-PCG pipeline.
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
from .thermal import (
    PowerVector,
    build_boundary_link_table,
    build_conductance_table,
    build_matrix_free_operator,
    map_power_sources,
    solve_pcg_gpu,
    validate_anchored_components,
)
from .thermal.boundary import BoundaryLinkTable
from .thermal.operator import MatrixFreeThermalOperator
from .thermal.setup_cache import (
    ThermalSetupArtifacts,
    build_scene_and_signature,
    load_setup_cache,
    save_setup_cache,
)
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
    reusable_setup: ThermalSetupArtifacts
    # Per-stage wall time (seconds).
    discretization_seconds: float
    conductance_seconds: float
    operator_seconds: float
    solve_seconds: float
    setup_build_seconds: float
    cache_serialization_seconds: float
    cache_load_seconds: float
    total_pipeline_seconds: float
    cache_status: str
    cache_path: str | None
    cache_size_bytes: int | None
    cache_physical_signature: str | None
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
    rtol: float = 1e-8,
    max_delta_t_K: float = 1e-6,
    max_iterations: int = 100_000,
    check_interval: int = 10,
    initial_temperature_K: float = 293.15,
    backend: str = "gpu_pcg",
    setup_cache_path: str | Path | None = None,
    reusable_setup: ThermalSetupArtifacts | None = None,
) -> PipelineResult:
    """Run the full steady-state pipeline and return all artifacts.

    ``max_cell_size_m`` is a 3-tuple ``(dx, dy, dz)`` in metres that
    overrides ``config.discretization.max_cell_size`` for this run.
    Pass ``None`` (the default) to use whatever the config declares.

    ``setup_cache_path`` optionally persists the power-independent mesh,
    mappings, graph, boundary links, matrix-free operator and Jacobi diagonal.
    Its physical signature excludes workload power and the RHS.

    The production backend is GPU-PCG with Jacobi preconditioning.
    """
    if backend != "gpu_pcg":
        raise ValueError(f"unknown backend {backend!r}; expected 'gpu_pcg'")
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

    pipeline_started = time.perf_counter()
    boxes = []
    physical_signature = None
    cache_path = Path(setup_cache_path) if setup_cache_path is not None else None
    artifacts = reusable_setup
    cache_load_seconds = 0.0
    cache_serialization_seconds = 0.0
    cache_status = "MEMORY_REUSE" if artifacts is not None else "DISABLED"
    if artifacts is None:
        boxes, physical_signature = build_scene_and_signature(config)
    if artifacts is None and cache_path is not None:
        artifacts, cache_load_seconds, cache_status = load_setup_cache(
            cache_path, physical_signature)

    discretization_seconds = conductance_seconds = operator_seconds = 0.0
    setup_build_seconds = 0.0
    if artifacts is None:
        build_started = time.perf_counter()
        t0 = time.perf_counter()
        grid = build_global_grid(boxes, config.discretization.max_cell_size)
        cells = generate_cells(boxes, grid)
        edges = build_adjacency(cells, grid)
        boundary_faces = build_boundary_faces(cells, grid)
        validate_volume_conservation(cells, boxes)
        validate_cell_surface_partition(cells, edges, boundary_faces)
        t1 = time.perf_counter()
        discretization_seconds = t1 - t0

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
        t3 = time.perf_counter()
        conductance_seconds = t3 - t2

        t4 = time.perf_counter()
        operator_template = build_matrix_free_operator(
            conductance=conductance_table, boundary=boundary_table,
            power_W=np.zeros(len(cells), dtype=np.float64),
        )
        validate_anchored_components(
            cell_count=operator_template.cell_count,
            internal_cell_a=operator_template.internal_cell_a,
            internal_cell_b=operator_template.internal_cell_b,
            boundary=boundary_table,
        )
        t5 = time.perf_counter()
        operator_seconds = t5 - t4
        artifacts = ThermalSetupArtifacts(
            grid=grid,
            cells=cells,
            edges=edges,
            boundary_faces=boundary_faces,
            conductance_table=conductance_table,
            boundary_table=boundary_table,
            operator_template=operator_template,
        )
        setup_build_seconds = time.perf_counter() - build_started
        if cache_path is not None:
            cache_serialization_seconds = save_setup_cache(
                cache_path, physical_signature, artifacts)
            cache_status = (
                "REBUILT" if cache_status == "INVALIDATED" else "BUILT")

    cells = artifacts.cells
    edges = artifacts.edges
    boundary_faces = artifacts.boundary_faces
    boundary_table = artifacts.boundary_table
    power = map_power_sources(cells=cells, config=config.thermal_power_sources)
    operator = artifacts.operator_template.with_power(power.power_W)

    # Solve.
    initial_T = np.full(operator.cell_count, initial_temperature_K,
                        dtype=np.float64)
    result = solve_pcg_gpu(
        operator, initial_T, boundary_table,
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
        reusable_setup=artifacts,
        discretization_seconds=discretization_seconds,
        conductance_seconds=conductance_seconds,
        operator_seconds=operator_seconds,
        solve_seconds=float(result.solve_seconds),
        setup_build_seconds=setup_build_seconds,
        cache_serialization_seconds=cache_serialization_seconds,
        cache_load_seconds=cache_load_seconds,
        total_pipeline_seconds=time.perf_counter() - pipeline_started,
        cache_status=cache_status,
        cache_path=str(cache_path) if cache_path is not None else None,
        cache_size_bytes=(
            cache_path.stat().st_size
            if cache_path is not None and cache_path.exists() else None),
        cache_physical_signature=(
            physical_signature if cache_path is not None else None),
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
