"""Single-factor steady-state mesh-convergence sweep.

The sweep runs the existing steady-state pipeline

    config -> geometry -> mesh -> conductance -> boundary/power
           -> matrix-free operator -> PCG

five times (xy 1.0 / 0.5 / 0.25 mm at fixed dz, plus z 200 / 50 um
at fixed dx=dy=0.5 mm; the 0.5 / 100 case is shared), with each
case's mesh size overridden in memory. The original YAML is never
written to, so the user can iterate by hand without touching the
on-disk file.

The module is intentionally pure: parsing, case-building, summary
aggregation and CSV / JSON writing are all plain functions. The
CLI subcommand in :mod:`om3dthermal.cli` glues them together and
handles ``--resume``.

This module never touches transient / k(T) / R''(T) / AMR / new
physics. It only changes the mesh and re-solves the same linear
system.
"""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .config import SimulationConfig, load_config
from .case_runner import PipelineResult, run_steady_pipeline
from .units import parse_length


# ---------------------------------------------------------------------------
# Mesh-size parsing
# ---------------------------------------------------------------------------

_LENGTH_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(.*)$")


def parse_mesh_sizes(spec: str) -> list[float]:
    """Parse a comma-separated list of lengths like ``"1.0mm,0.5mm,0.25mm"``.

    The list must be strictly decreasing (so adjacent ``ΔTmax`` values
    are positive-direction refinements) and contain at least one
    element. Lengths may carry any unit accepted by
    :func:`om3dthermal.units.parse_length` (e.g. ``mm``, ``um``,
    bare numbers are interpreted as metres).
    """
    if not spec or not spec.strip():
        raise ValueError("mesh-size list must be non-empty")
    out: list[float] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            raise ValueError(f"empty entry in mesh-size list {spec!r}")
        m = _LENGTH_RE.match(token)
        if not m:
            raise ValueError(
                f"could not parse mesh-size entry {token!r} in {spec!r}")
        try:
            out.append(float(parse_length(token)))
        except Exception as exc:
            raise ValueError(
                f"could not parse mesh-size entry {token!r} in {spec!r}: "
                f"{exc}") from exc
    for i in range(1, len(out)):
        # Compare with a small relative tolerance because Pint's
        # length conversion can introduce float noise (e.g.
        # ``parse_length("200um")`` = 1.999...e-4).
        prev, cur = out[i - 1], out[i]
        if cur >= prev or not math.isfinite(cur) or cur <= 0:
            raise ValueError(
                f"mesh-size list must be strictly decreasing for an "
                f"adjacent-pair convergence sweep; got {out!r}")
        if (prev - cur) / max(prev, 1e-30) < 1e-9:
            raise ValueError(
                f"adjacent mesh sizes {prev!r} and {cur!r} are "
                f"indistinguishable after unit conversion; pick a "
                f"coarser / finer pair")
    return out


# ---------------------------------------------------------------------------
# Sweep case construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaseSpec:
    """A single sweep case: a (dx, dy, dz) tuple and a label."""

    label: str
    dx_m: float
    dy_m: float
    dz_m: float

    @property
    def tuple(self) -> tuple[float, float, float]:
        return (self.dx_m, self.dy_m, self.dz_m)


def build_sweep_cases(
    xy_sizes_m: Sequence[float],
    z_sizes_m: Sequence[float],
) -> list[CaseSpec]:
    """Build the 5-case single-factor sweep.

    The middle element of each list is treated as the **fixed**
    baseline that the other sweep varies around:

    - xy sweep: each ``xy`` size paired with the middle z size.
    - z sweep: each ``z`` size paired with the middle xy size.
    - The shared baseline ``(middle_xy, middle_z)`` is included
      only once.

    For the canonical 5-case sweep
    ``xy = [1.0, 0.5, 0.25] mm`` and
    ``z = [200, 100, 50] um`` the middle elements are
    ``xy = 0.5 mm`` and ``z = 100 um`` respectively, so the xy
    sweep holds dz fixed at 100 um and the z sweep holds dx=dy
    fixed at 0.5 mm.

    The returned cases preserve input order: all xy cases first,
    then all z cases minus the duplicate baseline.
    """
    if not xy_sizes_m or not z_sizes_m:
        raise ValueError("xy_sizes and z_sizes must both be non-empty")
    base_xy = xy_sizes_m[len(xy_sizes_m) // 2]
    base_z = z_sizes_m[len(z_sizes_m) // 2]
    cases: list[CaseSpec] = []
    seen: set[tuple[float, float, float]] = set()
    for xy in xy_sizes_m:
        spec = CaseSpec(
            label=f"xy_{xy * 1e3:.4f}mm",
            dx_m=xy, dy_m=xy, dz_m=base_z,
        )
        key = spec.tuple
        if key in seen:
            continue
        seen.add(key)
        cases.append(spec)
    for z in z_sizes_m:
        spec = CaseSpec(
            label=f"z_{z * 1e6:.2f}um",
            dx_m=base_xy, dy_m=base_xy, dz_m=z,
        )
        key = spec.tuple
        if key in seen:
            continue
        seen.add(key)
        cases.append(spec)
    return cases


# ---------------------------------------------------------------------------
# Per-case result aggregation
# ---------------------------------------------------------------------------

def _case_to_row(result: PipelineResult, spec: CaseSpec) -> dict:
    """Flatten a PipelineResult into the per-case summary row used by
    both the CSV and the JSON outputs."""
    res = result.result
    T = res.temperature_K
    return {
        "label": spec.label,
        "dx_m": spec.dx_m,
        "dy_m": spec.dy_m,
        "dz_m": spec.dz_m,
        "dx_mm": spec.dx_m * 1e3,
        "dy_mm": spec.dy_m * 1e3,
        "dz_um": spec.dz_m * 1e6,
        "cell_count": result.cell_count,
        "internal_edge_count": result.internal_edge_count,
        "active_boundary_link_count": result.active_boundary_link_count,
        "adiabatic_face_count": result.adiabatic_face_count,
        "solver_method": res.method,
        "converged": bool(res.converged),
        "iterations": int(res.iterations),
        "matvec_count": int(res.solver_info.get("matvec_count", 0)),
        "initial_residual": float(res.initial_residual),
        "final_relative_residual": float(res.final_relative_residual),
        "min_temperature_K": float(res.min_temperature_K),
        "max_temperature_K": float(res.max_temperature_K),
        "mean_temperature_K": float(res.mean_temperature_K),
        "min_temperature_C": float(res.min_temperature_K - 273.15),
        "max_temperature_C": float(res.max_temperature_K - 273.15),
        "mean_temperature_C": float(res.mean_temperature_K - 273.15),
        "hottest_cell_id": result.hottest_cell_id,
        "hottest_cell_x_m": result.hottest_cell_xyz_m[0],
        "hottest_cell_y_m": result.hottest_cell_xyz_m[1],
        "hottest_cell_z_m": result.hottest_cell_xyz_m[2],
        "hottest_cell_material": result.hottest_cell_material,
        "hottest_cell_component": result.hottest_cell_component,
        "total_input_power_W": float(res.total_input_power_W),
        "total_boundary_heat_out_W": float(res.total_boundary_heat_out_W),
        "global_power_imbalance_W": float(res.global_power_imbalance_W),
        "relative_power_imbalance": float(res.relative_power_imbalance),
        "gpu_power_W": result.gpu_power_W,
        "hbm_power_W": result.hbm_power_W,
        "heat_out_by_rule_W": dict(result.heat_out_by_rule_W),
        "discretization_seconds": float(result.discretization_seconds),
        "conductance_seconds": float(result.conductance_seconds),
        "operator_seconds": float(result.operator_seconds),
        "solve_seconds": float(result.solve_seconds),
        "total_seconds": float(
            result.discretization_seconds
            + result.conductance_seconds
            + result.operator_seconds
            + result.solve_seconds
        ),
    }


# ---------------------------------------------------------------------------
# Delta-Tmax computation
# ---------------------------------------------------------------------------

def compute_delta_tmax(
    rows: Sequence[dict],
    *,
    xy_sizes_m: Sequence[float] | None = None,
    z_sizes_m: Sequence[float] | None = None,
) -> dict[str, dict]:
    """Compute ``ΔTmax`` for each adjacent refinement pair.

    The output is indexed by ``direction_sizeA_to_sizeB`` (e.g.
    ``xy_1.0000mm_to_0.5000mm`` or ``z_200.00um_to_100.00um``).
    Each value is a dict with the two mesh sizes, the two Tmax
    values and ``delta_Tmax_K``.

    The baseline case is shared between the xy and z sweeps, so it
    is solved only once. ``ΔTmax`` is therefore looked up by
    ``(dx, dy, dz)`` rather than by row order: a pair
    ``(coarse_z, fine_z)`` at the z-sweep's fixed ``dx=dy`` is
    matched against the row whose ``(dx, dy, dz)`` tuple exactly
    equals that combination, so the shared baseline contributes
    to the z-sweep ΔTmax without being solved twice.

    Pass ``xy_sizes_m`` and ``z_sizes_m`` to drive the pair search
    from the full input size lists; otherwise the function falls
    back to scanning the rows for adjacent cases (which misses the
    shared baseline when the deduped z sweep shrinks to two
    rows).
    """
    # Index every row by its (dx, dy, dz) tuple for O(1) lookup.
    by_mesh: dict[tuple[float, float, float], dict] = {}
    for row in rows:
        key = (row["dx_m"], row["dy_m"], row["dz_m"])
        by_mesh.setdefault(key, row)

    deltas: dict[str, dict] = {}

    def _label_for_direction(direction: str, size_m: float) -> str:
        if direction == "xy":
            return f"xy_{size_m * 1e3:.4f}mm"
        return f"z_{size_m * 1e6:.2f}um"

    # Determine the xy and z size lists to iterate.
    if xy_sizes_m is None:
        xy_sizes_m = sorted({
            r["dx_m"] for r in rows
            if r["label"].startswith("xy_")
        }, reverse=True)
    if z_sizes_m is None:
        z_sizes_m = sorted({
            r["dz_m"] for r in rows
            if r["label"].startswith("z_")
        }, reverse=True)

    # XY ΔTmax: pair each adjacent (coarser, finer) xy size at the
    # fixed z = the middle of the z_sizes list (i.e. the baseline
    # dz that the xy sweep held constant).
    if len(xy_sizes_m) >= 2:
        base_z = z_sizes_m[len(z_sizes_m) // 2] if z_sizes_m else None
        if base_z is not None:
            for coarse_xy, fine_xy in zip(xy_sizes_m, xy_sizes_m[1:]):
                coarse_row = by_mesh.get((coarse_xy, coarse_xy, base_z))
                fine_row = by_mesh.get((fine_xy, fine_xy, base_z))
                if coarse_row is None or fine_row is None:
                    continue
                key = (f"xy_{coarse_xy * 1e3:.4f}mm_to_"
                       f"{fine_xy * 1e3:.4f}mm")
                deltas[key] = {
                    "direction": "xy",
                    "coarse_label": _label_for_direction("xy", coarse_xy),
                    "fine_label": _label_for_direction("xy", fine_xy),
                    "coarse_max_temperature_K":
                        float(coarse_row["max_temperature_K"]),
                    "fine_max_temperature_K":
                        float(fine_row["max_temperature_K"]),
                    "delta_Tmax_K": float(
                        fine_row["max_temperature_K"]
                        - coarse_row["max_temperature_K"]),
                }

    # Z ΔTmax: pair each adjacent (coarser, finer) z size at the
    # fixed xy = the middle of the xy_sizes list. The matching row
    # may carry a ``xy_`` label (the shared baseline); the lookup
    # is purely by (dx,dy,dz) tuple so the shared baseline
    # contributes without being re-solved.
    if len(z_sizes_m) >= 2 and xy_sizes_m:
        base_xy = xy_sizes_m[len(xy_sizes_m) // 2]
        for coarse_z, fine_z in zip(z_sizes_m, z_sizes_m[1:]):
            coarse_row = by_mesh.get((base_xy, base_xy, coarse_z))
            fine_row = by_mesh.get((base_xy, base_xy, fine_z))
            if coarse_row is None or fine_row is None:
                continue
            key = (f"z_{coarse_z * 1e6:.2f}um_to_"
                   f"{fine_z * 1e6:.2f}um")
            deltas[key] = {
                "direction": "z",
                "coarse_label": _label_for_direction("z", coarse_z),
                "fine_label": _label_for_direction("z", fine_z),
                "coarse_max_temperature_K":
                    float(coarse_row["max_temperature_K"]),
                "fine_max_temperature_K":
                    float(fine_row["max_temperature_K"]),
                "delta_Tmax_K": float(
                    fine_row["max_temperature_K"]
                    - coarse_row["max_temperature_K"]),
            }
    return deltas


# ---------------------------------------------------------------------------
# CSV / JSON writing
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "label", "dx_mm", "dy_mm", "dz_um",
    "cell_count", "internal_edge_count",
    "active_boundary_link_count", "adiabatic_face_count",
    "solver_method", "converged", "iterations", "matvec_count",
    "initial_residual", "final_relative_residual",
    "min_temperature_K", "max_temperature_K", "mean_temperature_K",
    "min_temperature_C", "max_temperature_C", "mean_temperature_C",
    "hottest_cell_id", "hottest_cell_x_m", "hottest_cell_y_m",
    "hottest_cell_z_m", "hottest_cell_material", "hottest_cell_component",
    "total_input_power_W", "total_boundary_heat_out_W",
    "global_power_imbalance_W", "relative_power_imbalance",
    "gpu_power_W", "hbm_power_W",
    "discretization_seconds", "conductance_seconds",
    "operator_seconds", "solve_seconds", "total_seconds",
]


def write_mesh_convergence_csv(
    rows: Sequence[dict], path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_mesh_convergence_json(
    rows: Sequence[dict],
    delta_tmax: dict[str, dict],
    *,
    config_path: str | Path,
    xy_sizes_m: Sequence[float],
    z_sizes_m: Sequence[float],
    rtol: float,
    initial_temperature_K: float,
    method: str,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_label":
            "paper-parameter-aligned uniform-power baseline; "
            "mesh-convergence sweep, mesh overridden in memory only",
        "strict_paper_temperature_reproduction": False,
        "config_path": str(config_path),
        "solver_method": method,
        "solver_rtol": rtol,
        "initial_temperature_K": initial_temperature_K,
        "xy_sizes_m": list(xy_sizes_m),
        "z_sizes_m": list(z_sizes_m),
        "cases": [
            # Keep the heat_out_by_rule dict as a nested object for
            # machine consumption; drop it from the CSV row but keep
            # the full case dict in the JSON.
            {k: v for k, v in row.items()}
            for row in rows
        ],
        "delta_Tmax": delta_tmax,
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Per-case IO (for --resume)
# ---------------------------------------------------------------------------

def case_already_done(rows_path: Path, label: str) -> bool:
    """Return True iff a previous sweep run already wrote a row for
    ``label`` into ``rows_path`` (which may be a partial CSV from a
    prior interrupted run)."""
    if not rows_path.exists():
        return False
    with rows_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return any(r.get("label") == label for r in reader)


def write_case_row_partial(
    rows_path: Path, row: dict, fields: Sequence[str] = _CSV_FIELDS,
) -> None:
    """Append a single row to ``rows_path``, creating the file (with
    header) if it does not exist yet. Used by ``--resume`` so a
    crashed sweep can be continued without re-running completed
    cases."""
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = rows_path.exists()
    with rows_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields,
                                extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_partial_rows(rows_path: Path) -> list[dict]:
    """Read any rows already written to ``rows_path`` and convert
    the CSV-friendly units back to the in-memory row schema.

    The CSV stores ``dx_mm`` / ``dy_mm`` / ``dz_um`` /
    ``max_temperature_C`` (and the like); the in-memory rows
    consumed by :func:`compute_delta_tmax` need ``dx_m`` /
    ``dy_m`` / ``dz_m`` / ``max_temperature_K`` etc. We rebuild
    the missing fields so a resumed run can pass the cached rows
    straight to the summary writer.
    """
    if not rows_path.exists():
        return []
    with rows_path.open("r", encoding="utf-8", newline="") as stream:
        raw = list(csv.DictReader(stream))
    out: list[dict] = []
    for r in raw:
        row = dict(r)
        # CSV stores scalars as strings; cast the ones the
        # downstream consumers read.
        for key in (
            "dx_m", "dy_m", "dz_m",
            "cell_count", "internal_edge_count",
            "active_boundary_link_count", "adiabatic_face_count",
            "iterations", "matvec_count",
            "hottest_cell_id",
            "min_temperature_K", "max_temperature_K", "mean_temperature_K",
            "min_temperature_C", "max_temperature_C", "mean_temperature_C",
            "hottest_cell_x_m", "hottest_cell_y_m", "hottest_cell_z_m",
            "total_input_power_W", "total_boundary_heat_out_W",
            "global_power_imbalance_W", "relative_power_imbalance",
            "gpu_power_W", "hbm_power_W",
            "initial_residual", "final_relative_residual",
            "discretization_seconds", "conductance_seconds",
            "operator_seconds", "solve_seconds", "total_seconds",
        ):
            if key in row and row[key] != "":
                try:
                    row[key] = float(row[key])
                except (TypeError, ValueError):
                    pass
        if "cell_count" in row:
            try:
                row["cell_count"] = int(float(row["cell_count"]))
            except (TypeError, ValueError):
                pass
        if "internal_edge_count" in row:
            try:
                row["internal_edge_count"] = int(
                    float(row["internal_edge_count"]))
            except (TypeError, ValueError):
                pass
        if "iterations" in row:
            try:
                row["iterations"] = int(float(row["iterations"]))
            except (TypeError, ValueError):
                pass
        if "converged" in row:
            row["converged"] = str(row["converged"]).lower() in (
                "true", "1", "yes")
        # Provide dx_m / dy_m / dz_m aliases if the CSV has the mm
        # / um columns.
        if "dx_m" not in row and "dx_mm" in row:
            try:
                row["dx_m"] = float(row["dx_mm"]) * 1e-3
            except (TypeError, ValueError):
                pass
        if "dy_m" not in row and "dy_mm" in row:
            try:
                row["dy_m"] = float(row["dy_mm"]) * 1e-3
            except (TypeError, ValueError):
                pass
        if "dz_m" not in row and "dz_um" in row:
            try:
                row["dz_m"] = float(row["dz_um"]) * 1e-6
            except (TypeError, ValueError):
                pass
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Single-case runner (used by the CLI and the unit tests)
# ---------------------------------------------------------------------------

def run_single_case(
    config: SimulationConfig,
    spec: CaseSpec,
    *,
    method: str = "pcg",
    rtol: float = 1e-6,
    max_iterations: int = 10_000,
    initial_temperature_K: float = 293.15,
) -> dict:
    """Run one sweep case and return the per-case summary row.

    The result is a flat dict matching the CSV columns. Heavy
    arrays (temperature, cells, edges) are not retained; only the
    per-case aggregates that the sweep summary needs.
    """
    pipeline = run_steady_pipeline(
        config,
        max_cell_size_m=spec.tuple,
        method=method,
        rtol=rtol,
        max_iterations=max_iterations,
        initial_temperature_K=initial_temperature_K,
    )
    return _case_to_row(pipeline, spec)
