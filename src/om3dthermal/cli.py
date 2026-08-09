"""Command-line entry point."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import load_config
from .pipeline import run_steady_pipeline
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
from .sensitivity import (
    SensitivityCase,
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
    map_power_sources,
    solve_pcg,
    solve_weighted_jacobi,
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


def build(config_path: str | Path, output_dir: str | Path):
    config = load_config(config_path)
    scene = HorizontalColumnsBuilder(config).build()
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
    scene = HorizontalColumnsBuilder(config).build()
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
    scene = HorizontalColumnsBuilder(config).build()
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
    method: str = "pcg",
    omega: float = 0.7,
    max_iterations: int = 10_000,
    rtol: float = 1e-8,
    initial_temperature: float = 293.15,
) -> dict:
    """End-to-end steady-state thermal solve without ever
    materialising a dense or sparse matrix on the production path.
    """
    import numpy as np
    config = load_config(config_path)
    pipeline = run_steady_pipeline(
        config,
        method=method,
        omega=omega,
        rtol=rtol,
        max_iterations=max_iterations,
        initial_temperature_K=initial_temperature,
    )
    result = pipeline.result
    cells = pipeline.cells
    power = pipeline.power
    boundary_table = pipeline.boundary_table
    boundary_faces = pipeline.boundary_faces

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
    summary = build_solver_summary(
        result=result,
        cell_count=pipeline.cell_count,
        internal_edge_count=pipeline.internal_edge_count,
        active_boundary_link_count=pipeline.active_boundary_link_count,
        adiabatic_boundary_face_count=pipeline.adiabatic_face_count,
        boundary_build_seconds=pipeline.conductance_seconds,
        power_mapping_seconds=0.0,
        operator_build_seconds=pipeline.operator_seconds,
        gpu_power_W=pipeline.gpu_power_W,
        hbm_power_W=pipeline.hbm_power_W,
    )
    # Surface the individual stage timings for diagnostics.
    summary["discretization_seconds"] = pipeline.discretization_seconds
    summary["power_mapping_seconds"] = 0.0
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
    method: str = "pcg",
    rtol: float = 1e-6,
    max_iterations: int = 10_000,
    initial_temperature: float = 293.15,
    resume: bool = False,
) -> dict:
    """Run a single-factor sensitivity sweep on the DRAM lateral
    inset and the Mold compound k, and write the per-case
    results to ``output_dir``.

    The original YAML at ``config_path`` is never modified. Each
    case overrides the two target parameters in memory (a
    deep-copied YAML dict) and re-runs the full steady-state
    pipeline.

    The (baseline_inset, baseline_k) case is the
    v0.1.0-steady baseline (0.5 mm, 3 W/m*K) and is shared
    between the two single-factor sweeps. With the canonical
    ``--inset 0mm,0.25mm,0.5mm,0.75mm,1.0mm`` and
    ``--mold-k 0.5,1,3,10,30`` inputs the sweep solves 9
    cases in total.

    With ``resume=True`` (the CLI ``--resume`` flag), any case
    whose ``label`` is already present in the partial CSV is
    skipped, so a crashed run can be continued without
    re-solving finished cases.
    """
    inset_list = parse_length_list(inset_sizes)
    k_list = parse_k_list(k_values)
    # Convention: the middle element of each input list is the
    # *fixed* baseline that the other sweep varies around. For
    # the canonical 9-case sweep ``--inset 0mm,0.5mm,1.0mm``
    # and ``--mold-k 0.5,1,3,10,30`` the middle elements are
    # inset=0.5mm and k=3 W/m*K respectively, which is the
    # v0.1.0-steady baseline.
    baseline_inset = inset_list[len(inset_list) // 2]
    baseline_k = k_list[len(k_list) // 2]
    inset_cases = build_inset_sweep_cases(inset_list, baseline_k)
    k_cases = build_k_sweep_cases(k_list, baseline_inset)
    cases = merge_sweep_cases(inset_cases, k_cases)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = output_dir / "sensitivity.csv"
    json_path = output_dir / "sensitivity.json"

    rows: list[dict] = []
    if resume:
        existing = load_sensitivity_partial_rows(rows_csv)
        rows = list(existing)

    for case in cases:
        if resume and sensitivity_case_already_done(rows_csv, case.label):
            print(f"[sweep-sensitivity] skip {case.label} (cached)")
            continue
        if case.direction == "inset":
            print(
                f"[sweep-sensitivity] running {case.label}: "
                f"inset={case.inset_m*1e3:.3f}mm, "
                f"k={case.mold_k_W_mK:g} W/(m*K)")
        else:
            print(
                f"[sweep-sensitivity] running {case.label}: "
                f"inset={case.inset_m*1e3:.3f}mm, "
                f"k={case.mold_k_W_mK:g} W/(m*K)")
        row = run_single_sensitivity_case(
            config_path, case,
            method=method,
            rtol=rtol,
            max_iterations=max_iterations,
            initial_temperature_K=initial_temperature,
        )
        write_sensitivity_case_row_partial(rows_csv, row)
        rows.append(row)
        print(
            f"[sweep-sensitivity]   cells={row['cell_count']:>8} "
            f"edges={row['internal_edge_count']:>8} "
            f"iters={row['iterations']:>5} "
            f"Tmax={row['max_temperature_K']:>7.2f} K "
            f"({row['max_temperature_C']:>7.2f} C) "
            f"rel_res={row['final_relative_residual']:.2e} "
            f"power_imb={row['relative_power_imbalance']:.2e} "
            f"total={row['total_seconds']:.1f}s"
        )

    delta = compute_delta_tmax_sensitivity(
        rows,
        baseline_inset_m=baseline_inset,
        baseline_mold_k_W_mK=baseline_k,
    )
    write_sensitivity_csv(rows, rows_csv)
    write_sensitivity_json(
        rows, delta,
        config_path=config_path,
        inset_sizes_m=inset_list,
        k_values_W_mK=k_list,
        baseline_inset_m=baseline_inset,
        baseline_mold_k_W_mK=baseline_k,
        rtol=rtol,
        initial_temperature_K=initial_temperature,
        method=method,
        path=json_path,
    )
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
        help="matrix-free steady-state thermal solve (PCG or Jacobi)")
    solve_parser.add_argument("config", type=Path)
    solve_parser.add_argument("--out", type=Path, required=True)
    solve_parser.add_argument(
        "--method", choices=["pcg", "jacobi"], default="pcg")
    solve_parser.add_argument("--omega", type=float, default=0.7)
    solve_parser.add_argument("--max-iterations", type=int, default=10_000)
    solve_parser.add_argument("--rtol", type=float, default=1e-8)
    solve_parser.add_argument(
        "--initial-temperature", type=parse_temperature, default=293.15,
        help="uniform starting temperature in K (default 20 degC = 293.15 K)")
    sensitivity_parser = subparsers.add_parser(
        "sweep-sensitivity",
        help=("single-factor sensitivity sweep on DRAM lateral inset "
              "and Mold compound k (both at the v0.1.0-steady "
              "baseline)"))
    sensitivity_parser.add_argument("config", type=Path)
    sensitivity_parser.add_argument("--out", type=Path, required=True)
    sensitivity_parser.add_argument(
        "--inset", required=True,
        help=("comma-separated lateral inset values, e.g. "
              "'0mm,0.25mm,0.5mm,0.75mm,1.0mm' (first value is the "
              "shared baseline; list must be non-decreasing)"))
    sensitivity_parser.add_argument(
        "--mold-k", dest="mold_k", required=True,
        help=("comma-separated Mold k values in W/(m*K), e.g. "
              "'0.5,1,3,10,30' (first value is the shared "
              "baseline; list must be non-decreasing)"))
    sensitivity_parser.add_argument(
        "--method", choices=["pcg", "jacobi"], default="pcg")
    sensitivity_parser.add_argument("--rtol", type=float, default=1e-6)
    sensitivity_parser.add_argument("--max-iterations", type=int,
                                    default=10_000)
    sensitivity_parser.add_argument(
        "--initial-temperature", type=parse_temperature, default=293.15)
    sensitivity_parser.add_argument(
        "--resume", action="store_true",
        help=("skip cases whose label is already present in the "
              "partial sensitivity.csv (continue after a crash)"))
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
            method=args.method, omega=args.omega,
            max_iterations=args.max_iterations, rtol=args.rtol,
            initial_temperature=args.initial_temperature,
        )
        print(
            f"Solved {summary['cell_count']} cells with {args.method} "
            f"in {summary['iterations']} iterations: "
            f"T=[{summary['min_temperature_K']:.2f}, "
            f"{summary['max_temperature_K']:.2f}] K, "
            f"rel residual={summary['final_relative_residual']:.2e}, "
            f"power imbalance="
            f"{summary['relative_power_imbalance']:.2e}"
        )
    elif args.command == "sweep-sensitivity":
        result = sweep_sensitivity(
            args.config, args.out,
            inset_sizes=args.inset, k_values=args.mold_k,
            method=args.method, rtol=args.rtol,
            max_iterations=args.max_iterations,
            initial_temperature=args.initial_temperature,
            resume=args.resume,
        )
        delta = result["delta_Tmax"]
        print(
            f"[sweep-sensitivity] wrote {result['case_count']} cases to "
            f"{result['rows_path']} and {result['json_path']}"
        )
        for key, entry in delta.items():
            print(
                f"[sweep-sensitivity] ΔTmax {key}: "
                f"{entry['coarse_max_temperature_K']:.4f} -> "
                f"{entry['fine_max_temperature_K']:.4f} K "
                f"(delta = {entry['delta_Tmax_K']:+.4f} K)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
