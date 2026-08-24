"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import load_config
from .discretization import (
    build_adjacency,
    build_boundary_faces,
    build_global_grid,
    generate_cells,
    validate_cell_surface_partition,
    validate_volume_conservation,
)
from .discretization.export import (
    build_mesh_summary,
    write_boundary_faces_csv,
    write_cells_csv,
    write_edges_csv,
    write_mesh_summary_json,
)
from .geometry.horizontal_columns import HorizontalColumnsBuilder
from .geometry.orthogonal_hbm import OrthogonalHBMBuilder
from .mesh_convergence import (
    build_sweep_cases as build_mesh_sweep_cases,
    case_already_done as mesh_case_already_done,
    compute_delta_tmax as compute_mesh_delta_tmax,
    load_partial_rows as load_mesh_partial_rows,
    parse_mesh_sizes,
    run_single_case as run_single_mesh_case,
    write_case_row_partial as write_mesh_case_row_partial,
    write_mesh_convergence_csv,
    write_mesh_convergence_json,
)
from .power import run_case_system_power, run_memory_power
from .sensitivity import (
    build_inset_sweep_cases,
    build_k_sweep_cases,
    case_already_done as sensitivity_case_already_done,
    compute_delta_tmax_sensitivity,
    load_partial_rows as load_sensitivity_partial_rows,
    merge_sweep_cases,
    parse_k_list,
    parse_length_list,
    run_single_sensitivity_case,
    write_case_row_partial as write_sensitivity_case_row_partial,
    write_sensitivity_csv,
    write_sensitivity_json,
)
from .thermal import (
    build_boundary_link_table,
    build_conductance_table,
    build_matrix_free_operator,
    build_power_breakdown,
    map_power_sources,
    solve_thermal_resistance_relaxation,
    solve_thermal_resistance_relaxation_gpu,
    validate_anchored_components,
)
from .thermal.export import (
    build_conductance_summary,
    write_conductance_csv,
    write_conductance_npz,
    write_conductance_summary_json,
)
from .thermal.solution_export import (
    build_solver_summary,
    write_boundary_heat_flows_csv,
    write_solver_history_csv,
    write_solver_summary_json,
    write_temperature_csv,
    write_temperature_npz,
)
from .units import parse_temperature
from .visualization import write_visualizations


def build_scene(config):
    """Select the geometry template while keeping all downstream stages shared."""
    if config.orthogonal_hbm is not None:
        return OrthogonalHBMBuilder(config).build()
    return HorizontalColumnsBuilder(config).build()


def build(config_path: str | Path, output_dir: str | Path):
    config = load_config(config_path)
    scene = build_scene(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene.write_csv(output_dir / "regions.csv")
    scene.write_summary(output_dir / "geometry_summary.json")
    write_visualizations(scene, config, output_dir)
    return scene


def discretize(config_path: str | Path, output_dir: str | Path):
    """Build the block-structured mesh and write the four output files."""
    config = load_config(config_path)
    if config.discretization is None:
        raise ValueError(
            "config has no 'discretization' block; add one before running "
            "om3dthermal.cli discretize")
    scene = build_scene(config)
    boxes = list(scene.boxes)

    t0 = time.perf_counter()
    grid = build_global_grid(boxes, config.discretization.max_cell_size)
    cells = generate_cells(boxes, grid)
    t1 = time.perf_counter()

    validate_volume_conservation(cells, boxes)

    edges = build_adjacency(cells, grid)
    boundary_faces = build_boundary_faces(cells, grid)
    t2 = time.perf_counter()

    validate_cell_surface_partition(cells, edges, boundary_faces)

    total_box_volume = sum(
        (b.x1 - b.x0) * (b.y1 - b.y0) * (b.z1 - b.z0) for b in boxes)
    total_cell_volume = sum(c.volume for c in cells)

    summary = build_mesh_summary(
        scene_boxes=len(boxes), grid=grid, cells=cells, edges=edges,
        boundary_faces=boundary_faces,
        total_box_volume=total_box_volume,
        total_cell_volume=total_cell_volume,
        build_seconds=t1 - t0, adjacency_seconds=t2 - t1,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_cells_csv(cells, output_dir / "thermal_cells.csv")
    write_edges_csv(edges, output_dir / "adjacency_edges.csv")
    write_boundary_faces_csv(boundary_faces, output_dir / "boundary_faces.csv")
    write_mesh_summary_json(summary, output_dir / "mesh_summary.json")
    return summary


def conductance(config_path: str | Path, output_dir: str | Path,
                *, write_csv: bool = False):
    """Run the discretiser and compute the per-edge conductance table.

    The benchmark re-emits the three discretisation CSVs and the mesh
    summary so ``runs/<name>_conductance`` is self-contained; the new
    artefacts are ``conductance_edges.npz`` (always) and optionally
    ``conductance_edges.csv`` (when ``--write-conductance-csv`` is set).
    """
    config = load_config(config_path)
    if config.discretization is None:
        raise ValueError(
            "config has no 'discretization' block; add one before running "
            "om3dthermal.cli conductance")
    if config.thermal_conductance is None:
        raise ValueError(
            "config has no 'thermal_conductance' block; add one before "
            "running om3dthermal.cli conductance")
    scene = build_scene(config)
    boxes = list(scene.boxes)

    t0 = time.perf_counter()
    grid = build_global_grid(boxes, config.discretization.max_cell_size)
    cells = generate_cells(boxes, grid)
    t1 = time.perf_counter()
    validate_volume_conservation(cells, boxes)
    edges = build_adjacency(cells, grid)
    boundary_faces = build_boundary_faces(cells, grid)
    t2 = time.perf_counter()
    validate_cell_surface_partition(cells, edges, boundary_faces)

    t3 = time.perf_counter()
    table = build_conductance_table(
        cells=cells, adjacency_edges=edges,
        materials=config.materials,
        config=config.thermal_conductance,
    )
    t4 = time.perf_counter()

    total_box_volume = sum(
        (b.x1 - b.x0) * (b.y1 - b.y0) * (b.z1 - b.z0) for b in boxes)
    total_cell_volume = sum(c.volume for c in cells)
    mesh_summary = build_mesh_summary(
        scene_boxes=len(boxes), grid=grid, cells=cells, edges=edges,
        boundary_faces=boundary_faces,
        total_box_volume=total_box_volume,
        total_cell_volume=total_cell_volume,
        build_seconds=t1 - t0, adjacency_seconds=t2 - t1,
    )
    # We need the k_n cache size, but build_conductance_table does not
    # return it. Re-compute via the (deterministic) cache key lookup on
    # the same cell/edge enumeration: that is more brittle than
    # exposing the cache from build_conductance_table, but acceptable
    # for the summary because the cache is purely a memoisation
    # detail. The benchmark test does not depend on the exact entry
    # count beyond ``entries << edge_count``.
    unique_materials = sorted({c.material for c in cells})
    from .thermal.tensors import canonical_rotation_key
    cache_keys: set[tuple[str, tuple[int, ...], int]] = set()
    from .thermal.conductance import _AXIS_CODE
    for cell in cells:
        rot_key = canonical_rotation_key(cell.rotation)
        for axis_int in (0, 1, 2):
            cache_keys.add((cell.material, rot_key, axis_int))
    summary = build_conductance_summary(
        table=table,
        scene_box_count=len(boxes),
        cells=cells, edges=edges, boundary_faces=boundary_faces,
        unique_materials=unique_materials,
        k_n_cache_entries=len(cache_keys),
        default_interface_areal_resistance=(
            config.thermal_conductance.default_interface_areal_resistance),
        interface_rule_count=len(config.thermal_conductance.interfaces),
        discretization_seconds=(t1 - t0) + (t2 - t1),
        conductance_build_seconds=t4 - t3,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_cells_csv(cells, output_dir / "thermal_cells.csv")
    write_edges_csv(edges, output_dir / "adjacency_edges.csv")
    write_boundary_faces_csv(boundary_faces, output_dir / "boundary_faces.csv")
    write_mesh_summary_json(mesh_summary, output_dir / "mesh_summary.json")
    write_conductance_npz(table, output_dir / "conductance_edges.npz")
    if write_csv:
        write_conductance_csv(table, edges, output_dir / "conductance_edges.csv")
    write_conductance_summary_json(summary, output_dir / "conductance_summary.json")
    return summary


def solve_steady(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    alpha: float = 0.7,
    max_iterations: int = 100_000,
    rtol: float = 1e-8,
    max_delta_t_K: float = 1e-6,
    initial_temperature: float = 293.15,
    backend: str | None = None,
) -> dict:
    """End-to-end steady-state thermal solve without ever
    materialising a dense or sparse matrix on the production path.
    The only production solver is the thermal-resistance-network
    relaxation (CPU or GPU).
    """
    import numpy as np
    config = load_config(config_path)
    configured_solver = config.metadata.get("solver", {})
    resolved_backend = backend or configured_solver.get("backend", "cpu")
    if resolved_backend not in {"cpu", "gpu"}:
        raise ValueError(
            f"unknown solver backend {resolved_backend!r}; expected 'cpu' or 'gpu'")
    if config.discretization is None:
        raise ValueError(
            "config has no 'discretization' block; add one before running "
            "om3dthermal.cli solve-steady")
    if config.thermal_conductance is None:
        raise ValueError(
            "config has no 'thermal_conductance' block; add one before "
            "running om3dthermal.cli solve-steady")
    if config.thermal_boundary_conditions is None:
        raise ValueError(
            "config has no 'thermal_boundary_conditions' block; add one "
            "before running om3dthermal.cli solve-steady")
    if config.thermal_power_sources is None:
        raise ValueError(
            "config has no 'thermal_power_sources' block; add one before "
            "running om3dthermal.cli solve-steady")
    scene = build_scene(config)
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
    power_breakdown = build_power_breakdown(
        power=power, config=config.thermal_power_sources)
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
    initial_T = np.full(operator.cell_count, initial_temperature,
                        dtype=np.float64)
    if resolved_backend == "cpu":
        result = solve_thermal_resistance_relaxation(
            operator, initial_T, boundary_table,
            alpha=alpha,
            relative_residual_tolerance=rtol,
            max_temperature_update_tolerance=max_delta_t_K,
            max_iterations=max_iterations,
        )
    else:
        result = solve_thermal_resistance_relaxation_gpu(
            operator, initial_T, boundary_table,
            alpha=alpha,
            relative_residual_tolerance=rtol,
            max_temperature_update_tolerance=max_delta_t_K,
            max_iterations=max_iterations,
        )

    package_power = power_breakdown["whole_package"]
    gpu_power = float(package_power["gpu_total_W"])
    hbm_power = float(package_power["hbm_total_W"])

    # Write outputs.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_temperature_npz(result, cells, output_dir / "temperature_cells.npz")
    write_temperature_csv(result, cells, power,
                          output_dir / "temperature_cells.csv")
    write_boundary_heat_flows_csv(
        boundary_table, result, cells, boundary_faces,
        output_dir / "boundary_heat_flows.csv",
    )
    write_solver_history_csv(result, output_dir / "solver_history.csv")
    # Power per cell.
    np.savez(
        output_dir / "power_cells.npz",
        cell_id=np.array([c.id for c in cells], dtype=np.int64),
        power_W=power.power_W,
    )
    cell_by_id = {c.id: c for c in cells}
    adiabatic_face_count = sum(
        1 for f in boundary_faces
        if not _face_matches_selector_for_summary(f, cell_by_id, config)
    )
    summary = build_solver_summary(
        result=result,
        cell_count=len(cells),
        internal_edge_count=len(edges),
        active_boundary_link_count=boundary_table.link_count,
        adiabatic_boundary_face_count=adiabatic_face_count,
        boundary_build_seconds=t3 - t2,
        power_mapping_seconds=0.0,
        operator_build_seconds=t5 - t4,
        gpu_power_W=gpu_power,
        hbm_power_W=hbm_power,
    )
    # Surface the individual stage timings for diagnostics.
    summary["discretization_seconds"] = t1 - t0
    summary["power_mapping_seconds"] = 0.0
    summary["case_id"] = config.metadata.get("case_id", config.name)
    summary["solver_backend"] = resolved_backend
    summary["power_model"] = power_breakdown["power_model"]
    summary["power_breakdown"] = power_breakdown
    summary["power_by_source_W"] = dict(power.power_by_source)
    if "architecture_bookkeeping" in config.metadata:
        summary["architecture_bookkeeping"] = dict(
            config.metadata["architecture_bookkeeping"])
    if "power_provenance" in config.metadata:
        summary["power_provenance"] = dict(
            config.metadata["power_provenance"])
    summary["benchmark_label"] = (
        "paper-parameter-aligned Son23 component-power experiment"
        if power_breakdown["power_model"] == "son23split"
        else (
            "M3D-v1 matched-bandwidth array-read baseline"
            if power_breakdown["power_model"] == "m3d_operation_energy"
            else "paper-parameter-aligned uniform-power baseline"))
    write_solver_summary_json(summary, output_dir / "steady_state_summary.json")
    return summary


def _face_matches_selector_for_summary(face, cell_by_id, config):
    """Heuristic: a face is ``adiabatic`` if no rule matches it and
    the default is ``adiabatic``. Used for the count only.
    """
    from .thermal.boundary import select_boundary_rule
    if config.thermal_boundary_conditions is None:
        return False
    cell = cell_by_id.get(face.cell_id)
    if cell is None:
        return False
    return select_boundary_rule(
        face, cell,
        config.thermal_boundary_conditions.rules) is not None


def sweep_sensitivity(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    inset_sizes: str,
    k_values: str,
    alpha: float = 0.7,
    rtol: float = 1e-6,
    max_iterations: int = 100_000,
    initial_temperature: float = 293.15,
    resume: bool = False,
) -> dict:
    """Run the legacy single-factor inset/Mold-k sensitivity sweep."""
    inset_list = parse_length_list(inset_sizes)
    k_list = parse_k_list(k_values)
    baseline_inset = inset_list[len(inset_list) // 2]
    baseline_k = k_list[len(k_list) // 2]
    cases = merge_sweep_cases(
        build_inset_sweep_cases(inset_list, baseline_k),
        build_k_sweep_cases(k_list, baseline_inset),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = output_dir / "sensitivity.csv"
    json_path = output_dir / "sensitivity.json"
    rows = list(load_sensitivity_partial_rows(rows_csv)) if resume else []
    for case in cases:
        if resume and sensitivity_case_already_done(rows_csv, case.label):
            print(f"[sweep-sensitivity] skip {case.label} (cached)")
            continue
        print(
            f"[sweep-sensitivity] running {case.label}: "
            f"inset={case.inset_m*1e3:.3f}mm, "
            f"k={case.mold_k_W_mK:g} W/(m*K)")
        row = run_single_sensitivity_case(
            config_path, case, alpha=alpha, rtol=rtol,
            max_iterations=max_iterations,
            initial_temperature_K=initial_temperature)
        write_sensitivity_case_row_partial(rows_csv, row)
        rows.append(row)
    delta = compute_delta_tmax_sensitivity(
        rows, baseline_inset_m=baseline_inset,
        baseline_mold_k_W_mK=baseline_k)
    write_sensitivity_csv(rows, rows_csv)
    write_sensitivity_json(
        rows, delta, config_path=config_path,
        inset_sizes_m=inset_list, k_values_W_mK=k_list,
        baseline_inset_m=baseline_inset,
        baseline_mold_k_W_mK=baseline_k, rtol=rtol,
        initial_temperature_K=initial_temperature, alpha=alpha,
        path=json_path)
    return {
        "case_count": len(rows),
        "rows_path": str(rows_csv),
        "json_path": str(json_path),
        "delta_Tmax": delta,
    }


def sweep_mesh(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    xy_sizes: str,
    z_sizes: str,
    alpha: float = 0.7,
    rtol: float = 1e-6,
    max_iterations: int = 100_000,
    initial_temperature: float = 293.15,
    resume: bool = False,
) -> dict:
    """Run a single-factor steady-state mesh-convergence sweep."""
    config = load_config(config_path)
    xy_list = parse_mesh_sizes(xy_sizes)
    z_list = parse_mesh_sizes(z_sizes)
    cases = build_mesh_sweep_cases(xy_list, z_list)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = output_dir / "mesh_convergence.csv"
    json_path = output_dir / "mesh_convergence.json"
    rows = list(load_mesh_partial_rows(rows_csv)) if resume else []
    for spec in cases:
        if resume and mesh_case_already_done(rows_csv, spec.label):
            print(f"[sweep-mesh] skip {spec.label} (cached)")
            continue
        print(
            f"[sweep-mesh] running {spec.label}: "
            f"dx={spec.dx_m*1e3:.4f}mm dy={spec.dy_m*1e3:.4f}mm "
            f"dz={spec.dz_m*1e6:.2f}um")
        row = run_single_mesh_case(
            config, spec, alpha=alpha, rtol=rtol,
            max_iterations=max_iterations,
            initial_temperature_K=initial_temperature)
        write_mesh_case_row_partial(rows_csv, row)
        rows.append(row)
    delta = compute_mesh_delta_tmax(
        rows, xy_sizes_m=xy_list, z_sizes_m=z_list)
    write_mesh_convergence_csv(rows, rows_csv)
    write_mesh_convergence_json(
        rows, delta, config_path=config_path,
        xy_sizes_m=xy_list, z_sizes_m=z_list, rtol=rtol,
        initial_temperature_K=initial_temperature, alpha=alpha,
        path=json_path)
    return {
        "case_count": len(rows),
        "rows_path": str(rows_csv),
        "json_path": str(json_path),
        "delta_Tmax": delta,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="om3dthermal")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="build horizontal geometry")
    build_parser.add_argument("config", type=Path)
    build_parser.add_argument("--out", type=Path, required=True)
    discretize_parser = subparsers.add_parser(
        "discretize", help="build the block-structured ThermalCell mesh")
    discretize_parser.add_argument("config", type=Path)
    discretize_parser.add_argument("--out", type=Path, required=True)
    conductance_parser = subparsers.add_parser(
        "conductance",
        help="compute per-edge internal face thermal conductance")
    conductance_parser.add_argument("config", type=Path)
    conductance_parser.add_argument("--out", type=Path, required=True)
    conductance_parser.add_argument(
        "--write-conductance-csv", action="store_true",
        help="also write a 790k-row conductance_edges.csv (off by default)")
    solve_parser = subparsers.add_parser(
        "solve-steady",
        help="matrix-free thermal-resistance relaxation (CPU or GPU)")
    solve_parser.add_argument("config", type=Path)
    solve_parser.add_argument("--out", type=Path, required=True)
    solve_parser.add_argument(
        "--alpha", type=float, default=0.7,
        help="relaxation factor (0 < alpha <= 1)")
    solve_parser.add_argument(
        "--backend", choices=["cpu", "gpu"], default=None,
        help="solver backend; defaults to config solver.backend or cpu")
    solve_parser.add_argument("--max-iterations", type=int, default=100_000)
    solve_parser.add_argument("--rtol", type=float, default=1e-8)
    solve_parser.add_argument(
        "--max-delta-t-K", type=float, default=1e-6)
    solve_parser.add_argument(
        "--initial-temperature", type=parse_temperature, default=293.15,
        help="uniform starting temperature in K (default 20 degC = 293.15 K)")
    sensitivity_parser = subparsers.add_parser(
        "sweep-sensitivity",
        help="single-factor sweep on DRAM lateral inset and Mold k")
    sensitivity_parser.add_argument("config", type=Path)
    sensitivity_parser.add_argument("--out", type=Path, required=True)
    sensitivity_parser.add_argument("--inset", required=True)
    sensitivity_parser.add_argument("--mold-k", dest="mold_k", required=True)
    sensitivity_parser.add_argument("--alpha", type=float, default=0.7)
    sensitivity_parser.add_argument("--rtol", type=float, default=1e-6)
    sensitivity_parser.add_argument(
        "--max-iterations", type=int, default=100_000)
    sensitivity_parser.add_argument(
        "--initial-temperature", type=parse_temperature, default=293.15)
    sensitivity_parser.add_argument("--resume", action="store_true")
    mesh_parser = subparsers.add_parser(
        "sweep-mesh",
        help="single-factor steady-state mesh convergence sweep")
    mesh_parser.add_argument("config", type=Path)
    mesh_parser.add_argument("--out", type=Path, required=True)
    mesh_parser.add_argument("--xy", required=True)
    mesh_parser.add_argument("--z", required=True)
    mesh_parser.add_argument("--alpha", type=float, default=0.7)
    mesh_parser.add_argument("--rtol", type=float, default=1e-6)
    mesh_parser.add_argument("--max-iterations", type=int, default=100_000)
    mesh_parser.add_argument(
        "--initial-temperature", type=parse_temperature, default=293.15)
    mesh_parser.add_argument("--resume", action="store_true")
    power_parser = subparsers.add_parser(
        "power", help="run a config-driven standalone memory-power model")
    power_parser.add_argument("config", type=Path)
    sweep_parser = subparsers.add_parser(
        "sweep",
        help=(
            "run a config-driven OFAT memory parameter sweep; "
            "reuses the same case -> power -> thermal pipeline"))
    sweep_parser.add_argument("config", type=Path)
    sweep_parser.add_argument(
        "--thermal-backend", choices=["cpu", "gpu", "gpu_pcg"],
        default=None,
        help="override the sweep thermal solver without changing its cases")
    sweep_parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="override the sweep output directory")
    experiment_parser = subparsers.add_parser(
        "experiment",
        help="run a formal workload-aware experiment and write a result bundle")
    experiment_parser.add_argument("config", type=Path)
    experiment_parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="override the configured formal result-bundle directory")
    args = parser.parse_args(argv)
    if args.command == "build":
        scene = build(args.config, args.out)
        print(f"Built {len(scene.boxes)} boxes in {args.out}")
    elif args.command == "discretize":
        summary = discretize(args.config, args.out)
        print(f"Discretized {summary['cell_count']} cells "
              f"({summary['adjacency_edge_count']} edges, "
              f"{summary['boundary_face_count']} boundary faces) "
              f"in {args.out}")
    elif args.command == "conductance":
        summary = conductance(args.config, args.out,
                              write_csv=args.write_conductance_csv)
        print(f"Built {summary['conductance_edge_count']} conductance edges "
              f"({summary['edges_by_axis']}) in {args.out}")
    elif args.command == "solve-steady":
        summary = solve_steady(
            args.config, args.out,
            alpha=args.alpha,
            max_iterations=args.max_iterations, rtol=args.rtol,
            max_delta_t_K=args.max_delta_t_K,
            initial_temperature=args.initial_temperature,
            backend=args.backend,
        )
        print(
            f"Solved {summary['cell_count']} cells with relaxation/"
            f"{summary['solver_backend']} (alpha={args.alpha}) "
            f"in {summary['iterations']} iterations: "
            f"T=[{summary['min_temperature_K']:.2f}, "
            f"{summary['max_temperature_K']:.2f}] K, "
            f"rel residual={summary['final_relative_residual']:.2e}, "
            f"power imbalance="
            f"{summary['relative_power_imbalance']:.2e}"
        )
    elif args.command == "sweep-sensitivity":
        result = sweep_sensitivity(
            args.config, args.out, inset_sizes=args.inset,
            k_values=args.mold_k, alpha=args.alpha, rtol=args.rtol,
            max_iterations=args.max_iterations,
            initial_temperature=args.initial_temperature,
            resume=args.resume)
        print(
            f"[sweep-sensitivity] wrote {result['case_count']} cases to "
            f"{result['rows_path']} and {result['json_path']}")
    elif args.command == "sweep-mesh":
        result = sweep_mesh(
            args.config, args.out, xy_sizes=args.xy, z_sizes=args.z,
            alpha=args.alpha, rtol=args.rtol,
            max_iterations=args.max_iterations,
            initial_temperature=args.initial_temperature,
            resume=args.resume)
        print(
            f"[sweep-mesh] wrote {result['case_count']} cases to "
            f"{result['rows_path']} and {result['json_path']}")
    elif args.command == "power":
        if args.config.parent.name == "cases":
            result = run_case_system_power(args.config)
        else:
            result = run_memory_power(args.config)
        print(json.dumps(result.as_dict(display_na=True), indent=2))
    elif args.command == "sweep":
        from .sweep import run_sweep
        from . import _git_metadata
        git_meta = _git_metadata(args.config)
        result = run_sweep(
            args.config, git_metadata=git_meta,
            thermal_backend_override=args.thermal_backend,
            output_dir_override=args.output_dir)
        print(
            f"[sweep] {result.config_name}: "
            f"{result.pass_count} PASS / {result.fail_count} FAIL "
            f"out of {len(result.point_results)} points\n"
            f"[sweep] summary.csv: {Path(result.output_dir) / 'summary.csv'}\n"
            f"[sweep] metadata.json: {Path(result.output_dir) / 'metadata.json'}\n"
            f"[sweep] output_dir:   {result.output_dir}")
        return 0 if result.fail_count == 0 else 2
    elif args.command == "experiment":
        from .experiment import run_experiment
        result = run_experiment(
            args.config, output_dir_override=args.output_dir)
        assert result.output_dir is not None
        print(
            f"[experiment] {result.experiment.experiment_id}: PASS\n"
            f"[experiment] rows:       {len(result.rows)}\n"
            f"[experiment] output_dir: {result.output_dir}\n"
            f"[experiment] manifest:   {result.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
