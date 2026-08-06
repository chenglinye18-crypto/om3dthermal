"""CSV / NPZ / JSON export for the per-edge conductance table."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..discretization.models import AdjacencyEdge, BoundaryFace, ThermalCell
from .conductance import ConductanceTable


_AXIS_INT_TO_STR = {0: "x", 1: "y", 2: "z"}


def write_conductance_npz(table: ConductanceTable, path: str | Path) -> None:
    """Write the per-edge conductance arrays to a single ``.npz``.

    The downstream solver is expected to consume the ``.npz`` directly
    rather than parsing CSV; the file always carries every column of
    :class:`ConductanceTable`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        edge_id=table.edge_id,
        cell_a=table.cell_a,
        cell_b=table.cell_b,
        axis=table.axis,
        face_area_m2=table.face_area_m2,
        half_distance_a_m=table.half_distance_a_m,
        half_distance_b_m=table.half_distance_b_m,
        k_normal_a_W_mK=table.k_normal_a_W_mK,
        k_normal_b_W_mK=table.k_normal_b_W_mK,
        interface_areal_resistance_m2K_W=table.interface_areal_resistance_m2K_W,
        resistance_K_W=table.resistance_K_W,
        conductance_W_K=table.conductance_W_K,
        material_interface=table.material_interface,
        interface_rule_index=table.interface_rule_index,
    )


def write_conductance_csv(table: ConductanceTable,
                          edges: Sequence[AdjacencyEdge],
                          path: str | Path) -> None:
    """Write the per-edge conductance as a CSV.

    Only invoked when ``--write-conductance-csv`` is passed: the
    benchmark produces ~790 k rows and CSV is wasteful when the NPZ
    already carries the full set of columns.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "edge_id", "cell_a", "cell_b", "axis",
        "face_area_m2", "half_distance_a_m", "half_distance_b_m",
        "k_normal_a_W_mK", "k_normal_b_W_mK",
        "interface_areal_resistance_m2K_W",
        "resistance_K_W", "conductance_W_K",
        "material_interface", "interface_rule_index",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, edge in enumerate(edges):
            writer.writerow({
                "edge_id": int(table.edge_id[i]),
                "cell_a": int(table.cell_a[i]),
                "cell_b": int(table.cell_b[i]),
                "axis": _AXIS_INT_TO_STR[int(table.axis[i])],
                "face_area_m2": float(table.face_area_m2[i]),
                "half_distance_a_m": float(table.half_distance_a_m[i]),
                "half_distance_b_m": float(table.half_distance_b_m[i]),
                "k_normal_a_W_mK": float(table.k_normal_a_W_mK[i]),
                "k_normal_b_W_mK": float(table.k_normal_b_W_mK[i]),
                "interface_areal_resistance_m2K_W": float(
                    table.interface_areal_resistance_m2K_W[i]),
                "resistance_K_W": float(table.resistance_K_W[i]),
                "conductance_W_K": float(table.conductance_W_K[i]),
                "material_interface": bool(table.material_interface[i]),
                "interface_rule_index": int(table.interface_rule_index[i]),
            })


def build_conductance_summary(
    *,
    table: ConductanceTable,
    scene_box_count: int,
    cells: Sequence[ThermalCell],
    edges: Sequence[AdjacencyEdge],
    boundary_faces: Sequence[BoundaryFace],
    unique_materials: Sequence[str],
    k_n_cache_entries: int,
    default_interface_areal_resistance: float,
    interface_rule_count: int,
    discretization_seconds: float,
    conductance_build_seconds: float,
) -> dict[str, Any]:
    """Aggregate the conductance artifacts into a JSON-friendly dict."""
    by_axis = Counter(_AXIS_INT_TO_STR[int(a)] for a in table.axis)
    material_iface = int(np.count_nonzero(table.material_interface))
    cell_by_id = {c.id: c for c in cells}
    material_iface_pairs: Counter[tuple[str, str]] = Counter()
    for i, edge in enumerate(edges):
        if not bool(table.material_interface[i]):
            continue
        a_mat = cell_by_id[edge.cell_a].material
        b_mat = cell_by_id[edge.cell_b].material
        material_iface_pairs[tuple(sorted((a_mat, b_mat)))] += 1
    # JSON does not accept tuple keys; encode the unordered pair as
    # "A|B" so the order in the file is unambiguous and deterministic.
    material_iface_pairs_json = {
        f"{a}|{b}": count
        for (a, b), count in sorted(
            material_iface_pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    }
    return {
        "scene_box_count": scene_box_count,
        "thermal_cell_count": len(cells),
        "adjacency_edge_count": len(edges),
        "boundary_face_count": len(boundary_faces),
        "conductance_edge_count": table.edge_count,
        "edges_by_axis": dict(sorted(by_axis.items())),
        "material_interface_edge_count": material_iface,
        "nonzero_interface_resistance_edge_count":
            table.nonzero_interface_resistance_count,
        "min_k_normal_W_mK": table.min_k_normal,
        "max_k_normal_W_mK": table.max_k_normal,
        "min_conductance_W_K": table.min_conductance,
        "max_conductance_W_K": table.max_conductance,
        "mean_conductance_W_K": table.mean_conductance,
        "min_resistance_K_W": table.min_resistance,
        "max_resistance_K_W": table.max_resistance,
        "interface_rule_count": interface_rule_count,
        "default_interface_areal_resistance_m2K_W":
            default_interface_areal_resistance,
        "unique_material_count": len(unique_materials),
        "unique_material_rotation_axis_cache_entries": k_n_cache_entries,
        "material_interface_pairs_top": material_iface_pairs_json,
        "discretization_seconds": discretization_seconds,
        "conductance_build_seconds": conductance_build_seconds,
    }


def write_conductance_summary_json(summary: dict[str, Any],
                                   path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
