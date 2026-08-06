"""CSV / JSON export for the discretisation artifacts."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .models import AdjacencyEdge, BoundaryFace, ThermalCell


def write_cells_csv(cells: list[ThermalCell], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "ix", "iy", "iz", "x0", "x1", "y0", "y1", "z0", "z1",
              "material", "parent_box_id", "parent_box_name", "component",
              "source_path", "rotation", "tags"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            row = {
                "id": cell.id, "ix": cell.ix, "iy": cell.iy, "iz": cell.iz,
                "x0": cell.x0, "x1": cell.x1, "y0": cell.y0, "y1": cell.y1,
                "z0": cell.z0, "z1": cell.z1,
                "material": cell.material, "parent_box_id": cell.parent_box_id,
                "parent_box_name": cell.parent_box_name,
                "component": cell.component or "",
                "source_path": cell.source_path,
                "rotation": json.dumps(cell.rotation),
                "tags": json.dumps(cell.tags, ensure_ascii=False, sort_keys=True),
            }
            writer.writerow(row)


def write_edges_csv(edges: list[AdjacencyEdge], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "cell_a", "cell_b", "axis", "interface_coordinate",
              "face_area", "center_distance", "half_distance_a",
              "half_distance_b", "material_a", "material_b",
              "is_material_interface"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for edge in edges:
            writer.writerow({
                "id": edge.id, "cell_a": edge.cell_a, "cell_b": edge.cell_b,
                "axis": edge.axis,
                "interface_coordinate": edge.interface_coordinate,
                "face_area": edge.face_area,
                "center_distance": edge.center_distance,
                "half_distance_a": edge.half_distance_a,
                "half_distance_b": edge.half_distance_b,
                "material_a": edge.material_a,
                "material_b": edge.material_b,
                "is_material_interface": edge.is_material_interface,
            })


def write_boundary_faces_csv(faces: list[BoundaryFace], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "cell_id", "axis", "side", "coordinate", "area",
              "normal_x", "normal_y", "normal_z", "component",
              "material", "classification"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for face in faces:
            writer.writerow({
                "id": face.id, "cell_id": face.cell_id,
                "axis": face.axis, "side": face.side,
                "coordinate": face.coordinate, "area": face.area,
                "normal_x": face.normal[0], "normal_y": face.normal[1],
                "normal_z": face.normal[2],
                "component": face.component or "",
                "material": face.material, "classification": face.classification,
            })


def build_mesh_summary(*, scene_boxes: int, grid, cells, edges, boundary_faces,
                       total_box_volume: float, total_cell_volume: float,
                       build_seconds: float, adjacency_seconds: float) -> dict[str, Any]:
    """Aggregate the discretisation artifacts into a single JSON-friendly
    summary dict.
    """
    cell_sizes_x = [c.size_x for c in cells]
    cell_sizes_y = [c.size_y for c in cells]
    cell_sizes_z = [c.size_z for c in cells]
    cells_by_material = dict(Counter(c.material for c in cells))
    cells_by_component: dict[str, int] = defaultdict(int)
    for c in cells:
        cells_by_component[c.component or "<unassigned>"] += 1
    edges_by_axis = dict(Counter(e.axis for e in edges))
    return {
        "scene_boxes": scene_boxes,
        "x_cut_count": grid.nx + 1,
        "y_cut_count": grid.ny + 1,
        "z_cut_count": grid.nz + 1,
        "cell_count": len(cells),
        "adjacency_edge_count": len(edges),
        "boundary_face_count": len(boundary_faces),
        "cells_by_material": dict(sorted(cells_by_material.items())),
        "cells_by_component": dict(sorted(cells_by_component.items())),
        "edges_by_axis": dict(sorted(edges_by_axis.items())),
        "material_interface_edge_count": sum(1 for e in edges
                                             if e.is_material_interface),
        "outer_boundary_face_count": sum(1 for f in boundary_faces
                                          if f.classification == "scene_outer_boundary"),
        "exposed_internal_boundary_face_count": sum(1 for f in boundary_faces
                                                    if f.classification == "exposed_internal_boundary"),
        "total_box_volume_m3": total_box_volume,
        "total_cell_volume_m3": total_cell_volume,
        "volume_error_m3": total_cell_volume - total_box_volume,
        "max_cell_size_m": (
            max(max(cell_sizes_x) if cell_sizes_x else 0.0,
                max(cell_sizes_y) if cell_sizes_y else 0.0,
                max(cell_sizes_z) if cell_sizes_z else 0.0)),
        "min_cell_size_m": (
            min(min(cell_sizes_x) if cell_sizes_x else 0.0,
                min(cell_sizes_y) if cell_sizes_y else 0.0,
                min(cell_sizes_z) if cell_sizes_z else 0.0)),
        "build_seconds": build_seconds,
        "adjacency_seconds": adjacency_seconds,
    }


def write_mesh_summary_json(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
