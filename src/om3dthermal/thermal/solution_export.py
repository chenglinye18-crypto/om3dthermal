"""CSV / NPZ / JSON export for the steady-state solver output."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..discretization.models import BoundaryFace, ThermalCell
from .boundary import BoundaryLinkTable
from .power import PowerVector
from .steady_state import SteadyStateResult


_KIND_INT_TO_STR = {1: "convection", 2: "fixed_temperature"}


def write_temperature_npz(result: SteadyStateResult,
                          cells: list[ThermalCell],
                          path: str | Path) -> None:
    """Write the per-cell temperature vector and the cell id /
    coordinate lookup as a single ``.npz``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_ids = np.array([c.id for c in cells], dtype=np.int64)
    centers = np.array(
        [[c.center_x, c.center_y, c.center_z] for c in cells],
        dtype=np.float64,
    )
    np.savez(
        path,
        cell_id=cell_ids,
        center_x_m=centers[:, 0],
        center_y_m=centers[:, 1],
        center_z_m=centers[:, 2],
        temperature_K=result.temperature_K,
    )


def write_temperature_csv(result: SteadyStateResult,
                          cells: list[ThermalCell],
                          power: PowerVector,
                          path: str | Path) -> None:
    """Write the per-cell temperature and per-cell power as a CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cell_id", "center_x_m", "center_y_m", "center_z_m",
        "material", "component", "temperature_K", "temperature_C",
        "power_W",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, cell in enumerate(cells):
            T = float(result.temperature_K[i])
            writer.writerow({
                "cell_id": cell.id,
                "center_x_m": cell.center_x,
                "center_y_m": cell.center_y,
                "center_z_m": cell.center_z,
                "material": cell.material,
                "component": cell.component or "",
                "temperature_K": T,
                "temperature_C": T - 273.15,
                "power_W": float(power.power_W[i]),
            })


def write_boundary_heat_flows_csv(boundary: BoundaryLinkTable,
                                  result: SteadyStateResult,
                                  cells: list[ThermalCell],
                                  boundary_faces: list[BoundaryFace],
                                  path: str | Path) -> None:
    """Write the per-link outward heat flow as a CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_by_id = {c.id: c for c in cells}
    face_by_id = {f.id: f for f in boundary_faces}
    fields = [
        "boundary_face_id", "cell_id", "kind", "axis", "side",
        "face_area_m2", "k_normal_W_mK", "areal_resistance_m2K_W",
        "external_film_resistance_m2K_W", "conductance_W_K",
        "reference_temperature_K", "cell_temperature_K",
        "heat_out_W", "material", "component",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(boundary.link_count):
            face_id = int(boundary.boundary_face_id[i])
            cell_id = int(boundary.cell_id[i])
            face = face_by_id.get(face_id)
            cell = cell_by_id[cell_id]
            T_cell = float(result.temperature_K[cell_id])
            T_ref = float(boundary.reference_temperature_K[i])
            G = float(boundary.conductance_W_K[i])
            q_out = G * (T_cell - T_ref)
            axis = "xyz"[int(boundary.axis[i])]
            side = "minus" if int(boundary.side[i]) == 0 else "plus"
            writer.writerow({
                "boundary_face_id": face_id,
                "cell_id": cell_id,
                "kind": _KIND_INT_TO_STR.get(int(boundary.kind[i]),
                                              "unknown"),
                "axis": axis,
                "side": side,
                "face_area_m2": float(boundary.face_area_m2[i]),
                "k_normal_W_mK": float(boundary.k_normal_W_mK[i]),
                "areal_resistance_m2K_W":
                    float(boundary.areal_resistance_m2K_W[i]),
                "external_film_resistance_m2K_W":
                    float(boundary.external_film_resistance_m2K_W[i]),
                "conductance_W_K": G,
                "reference_temperature_K": T_ref,
                "cell_temperature_K": T_cell,
                "heat_out_W": q_out,
                "material": face.material if face is not None else "",
                "component": cell.component or "",
            })


def write_solver_history_csv(result: SteadyStateResult,
                            path: str | Path) -> None:
    """Write the residual / update history as a CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["iteration_index", "metric", "value"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, value in enumerate(result.residual_history):
            writer.writerow({
                "iteration_index": i,
                "metric": "relative_residual",
                "value": value,
            })
        for i, value in enumerate(result.update_norm_history):
            writer.writerow({
                "iteration_index": i,
                "metric": "max_temperature_update_K",
                "value": value,
            })


def build_solver_summary(
    *,
    result: SteadyStateResult,
    cell_count: int,
    internal_edge_count: int,
    active_boundary_link_count: int,
    adiabatic_boundary_face_count: int,
    boundary_build_seconds: float,
    power_mapping_seconds: float,
    operator_build_seconds: float,
    gpu_power_W: float,
    hbm_power_W: float,
) -> dict[str, Any]:
    """Aggregate the solver artifacts into a JSON-friendly dict."""
    T = result.temperature_K
    hottest_idx = int(np.argmax(T))
    summary = {
        "cell_count": cell_count,
        "internal_edge_count": internal_edge_count,
        "active_boundary_link_count": active_boundary_link_count,
        "adiabatic_boundary_face_count": adiabatic_boundary_face_count,
        "solver_method": result.method,
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "matvec_count": int(result.solver_info.get("matvec_count", 0)),
        "initial_residual": float(result.initial_residual),
        "final_absolute_residual": float(result.final_absolute_residual),
        "final_relative_residual": float(result.final_relative_residual),
        "min_temperature_K": float(result.min_temperature_K),
        "min_temperature_C": float(result.min_temperature_K - 273.15),
        "max_temperature_K": float(result.max_temperature_K),
        "max_temperature_C": float(result.max_temperature_K - 273.15),
        "mean_temperature_K": float(result.mean_temperature_K),
        "hottest_cell_id": int(hottest_idx),
        "hottest_cell_temperature_K": float(T[hottest_idx]),
        "total_input_power_W": float(result.total_input_power_W),
        "total_boundary_heat_out_W": float(result.total_boundary_heat_out_W),
        "global_power_imbalance_W":
            float(result.global_power_imbalance_W),
        "relative_power_imbalance":
            float(result.relative_power_imbalance),
        "gpu_power_W": float(gpu_power_W),
        "hbm_power_W": float(hbm_power_W),
        "boundary_build_seconds": boundary_build_seconds,
        "power_mapping_seconds": power_mapping_seconds,
        "operator_build_seconds": operator_build_seconds,
        "solve_seconds": float(result.solve_seconds),
        "total_seconds": float(
            boundary_build_seconds + power_mapping_seconds
            + operator_build_seconds + result.solve_seconds),
        "solver_info": dict(result.solver_info),
        "benchmark_label":
            "paper-parameter-aligned uniform-power baseline",
        "strict_paper_temperature_reproduction": False,
    }
    if result.max_temperature_update is not None:
        summary["max_temperature_update_K"] = float(
            result.max_temperature_update)
    return summary


def write_solver_summary_json(summary: dict[str, Any],
                              path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
