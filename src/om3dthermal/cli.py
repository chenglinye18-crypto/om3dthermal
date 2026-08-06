"""Command-line entry point."""

from __future__ import annotations

import argparse
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
