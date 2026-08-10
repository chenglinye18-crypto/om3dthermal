"""Single-factor steady-state sensitivity sweep for the two
``MODELING_CHOICE`` / ``DERIVED_FROM_PAPER_FIGURE`` parameters
that the v0.1.0-steady benchmark does not get from the paper:

1. **DRAM lateral inset** (0.5 mm per side in the v0.1.0-steady
   YAML; the paper does not publish a per-edge DRAM footprint).
2. **Mold compound thermal conductivity** (3 W/(m*K); a typical
   EMC value, not paper-reported).

The sweep runs the existing steady-state pipeline
(``config -> geometry -> mesh -> conductance -> boundary/power
-> matrix-free PCG``) with each swept parameter overridden
**in memory**; the YAML on disk is never modified.

Two single-factor sweeps are produced (mirroring the mesh
convergence sweep's shape):

- **Inset sweep**: 5 inset values at fixed Mold k
  (default 0 / 0.25 / 0.5 / 0.75 / 1.0 mm).
- **Mold-k sweep**: 5 k values at fixed inset
  (default 0.5 / 1 / 3 / 10 / 30 W/(m*K)).

The (inset = 0.5 mm, k = 3 W/(m*K)) case is the v0.1.0-steady
baseline and is shared between the two sweeps, so the canonical
9-case sweep solves 9 cases in total.

This module never introduces transient / k(T) / R''(T) / AMR /
new physics. It only changes the DRAM lateral inset and the
Mold compound thermal conductivity and re-solves the same
linear system.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import SimulationConfig
from .pipeline import PipelineResult, run_steady_pipeline
from .units import parse_length


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

_LENGTH_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(.*)$")


def parse_length_list(spec: str) -> list[float]:
    """Parse a comma-separated list of lengths like ``"0mm,0.25mm,0.5mm"``.

    Lengths may carry any unit accepted by
    :func:`om3dthermal.units.parse_length`. The list must be
    strictly non-decreasing (the user is sweeping from no-inset
    to largest-inset, so the second direction is correct here).
    """
    if not spec or not spec.strip():
        raise ValueError("length list must be non-empty")
    out: list[float] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            raise ValueError(f"empty entry in length list {spec!r}")
        try:
            out.append(float(parse_length(token)))
        except Exception as exc:
            raise ValueError(
                f"could not parse length entry {token!r} in {spec!r}: "
                f"{exc}") from exc
    # Every entry must be finite and non-negative.
    for value in out:
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"length list entries must be finite and non-negative; "
                f"got {out!r}")
    for i in range(1, len(out)):
        prev, cur = out[i - 1], out[i]
        if cur < prev:
            raise ValueError(
                f"length list must be non-decreasing for the "
                f"sensitivity sweep; got {out!r}")
        if (cur - prev) / max(prev, 1e-30) < 1e-9 and cur != prev:
            raise ValueError(
                f"adjacent lengths {prev!r} and {cur!r} are "
                f"indistinguishable after unit conversion")
    return out


def parse_k_list(spec: str) -> list[float]:
    """Parse a comma-separated list of positive W/(m*K) values."""
    if not spec or not spec.strip():
        raise ValueError("k list must be non-empty")
    out: list[float] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            raise ValueError(f"empty entry in k list {spec!r}")
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(
                f"could not parse k entry {token!r} in {spec!r}: "
                f"{exc}") from exc
        if value <= 0 or not math.isfinite(value):
            raise ValueError(
                f"k value must be strictly positive, got {token!r}")
        out.append(value)
    return out


# ---------------------------------------------------------------------------
# Sweep case construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SensitivityCase:
    """A single sweep case: which knob is varied and the fixed
    baseline values for the other knobs."""

    label: str
    direction: str           # "inset" or "mold_k"
    inset_m: float           # DRAM lateral inset (per side, m)
    mold_k_W_mK: float       # Mold compound thermal conductivity (W/m*K)


def build_inset_sweep_cases(
    inset_sizes_m: Sequence[float],
    fixed_mold_k_W_mK: float,
) -> list[SensitivityCase]:
    """Build the inset-sweep cases.

    Each case is labelled by its inset value (``inset_<mm>mm``)
    except for the (inset == baseline_inset, k == fixed_k)
    case, which is labelled ``baseline`` to make it easy to
    spot the shared v0.1.0-steady case in the per-case output.
    The "baseline inset" is the middle of the input list, so
    the sweep varies both coarser and finer around it.
    """
    if not inset_sizes_m:
        raise ValueError("inset_sizes_m must be non-empty")
    base_inset = inset_sizes_m[len(inset_sizes_m) // 2]
    cases: list[SensitivityCase] = []
    for ins in inset_sizes_m:
        if ins == base_inset and fixed_mold_k_W_mK is not None:
            label = "baseline"
        else:
            label = f"inset_{ins*1e3:.3f}mm"
        cases.append(SensitivityCase(
            label=label,
            direction="inset",
            inset_m=ins,
            mold_k_W_mK=fixed_mold_k_W_mK,
        ))
    return cases


def build_k_sweep_cases(
    k_values_W_mK: Sequence[float],
    fixed_inset_m: float,
) -> list[SensitivityCase]:
    """Build the k-sweep cases.

    Each case is labelled by its k value (``mold_k_<value>``)
    except for the (inset == fixed_inset, k == baseline_k) case,
    which is labelled ``baseline``. The "baseline k" is the
    middle of the input list.
    """
    if not k_values_W_mK:
        raise ValueError("k_values_W_mK must be non-empty")
    base_k = k_values_W_mK[len(k_values_W_mK) // 2]
    cases: list[SensitivityCase] = []
    for k in k_values_W_mK:
        if k == base_k and fixed_inset_m is not None:
            label = "baseline"
        else:
            label = f"mold_k_{k:g}"
        cases.append(SensitivityCase(
            label=label,
            direction="mold_k",
            inset_m=fixed_inset_m,
            mold_k_W_mK=k,
        ))
    return cases


def merge_sweep_cases(
    inset_cases: Sequence[SensitivityCase],
    k_cases: Sequence[SensitivityCase],
) -> list[SensitivityCase]:
    """Combine the two single-factor sweeps into one list, with
    the (baseline, baseline) case deduplicated. The order
    preserved is: inset cases first, then k cases minus the
    shared baseline.
    """
    seen: set[tuple[float, float]] = set()
    out: list[SensitivityCase] = []
    for case in inset_cases:
        key = (case.inset_m, case.mold_k_W_mK)
        if key in seen:
            continue
        seen.add(key)
        out.append(case)
    for case in k_cases:
        key = (case.inset_m, case.mold_k_W_mK)
        if key in seen:
            continue
        seen.add(key)
        out.append(case)
    return out


# ---------------------------------------------------------------------------
# YAML override + per-case runner
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict:
    """Load a YAML file as a plain dict, without compiling the
    compact form. The dict is a *deep copy* of the on-disk
    structure so subsequent mutations never touch the file."""
    import yaml
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}")
    return copy.deepcopy(data)


def _apply_overrides(
    data: dict,
    *,
    inset_m: float | None = None,
    mold_k_W_mK: float | None = None,
) -> dict:
    """Return a deep-copied ``data`` with the requested
    parameter values overridden. The caller is expected to pass
    the result to :func:`compile_user_config`; this function
    does **not** compile the YAML itself.

    Both overrides are applied to the compact user form so the
    compile path that already handles ``geometry.hbm.inset`` and
    ``materials.Mold`` is reused; the per-edge DRAM
    ``lateral_inset`` is therefore automatically propagated to
    every DRAM layer and the top die by the existing HBM
    sub-template builder.
    """
    out = copy.deepcopy(data)
    if inset_m is not None:
        if "geometry" not in out or not isinstance(out["geometry"], dict):
            raise ValueError(
                "YAML has no 'geometry' block; cannot override "
                "DRAM lateral inset")
        hbm = out["geometry"].get("hbm")
        if not isinstance(hbm, dict):
            raise ValueError(
                "YAML 'geometry' has no 'hbm' block; cannot override "
                "DRAM lateral inset")
        # Store as a length in metres. The compact compiler / Length
        # validator accepts a raw float (metres) or a string like
        # "0.5 mm"; we use the raw float for compactness.
        hbm["inset"] = float(inset_m)
    if mold_k_W_mK is not None:
        if "materials" not in out or not isinstance(out["materials"], dict):
            raise ValueError(
                "YAML has no 'materials' block; cannot override "
                "Mold k")
        if "Mold" not in out["materials"]:
            raise ValueError(
                "YAML 'materials' has no 'Mold' entry; cannot "
                "override Mold k")
        out["materials"]["Mold"] = float(mold_k_W_mK)
    return out


def _row_from_pipeline(pipeline: PipelineResult,
                        case: SensitivityCase) -> dict:
    """Flatten a PipelineResult into the per-case summary row
    used by both the CSV and the JSON outputs."""
    res = pipeline.result
    return {
        "label": case.label,
        "direction": case.direction,
        "inset_m": case.inset_m,
        "inset_mm": case.inset_m * 1e3,
        "mold_k_W_mK": case.mold_k_W_mK,
        "cell_count": pipeline.cell_count,
        "internal_edge_count": pipeline.internal_edge_count,
        "active_boundary_link_count": pipeline.active_boundary_link_count,
        "adiabatic_face_count": pipeline.adiabatic_face_count,
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
        "hottest_cell_id": pipeline.hottest_cell_id,
        "hottest_cell_x_m": pipeline.hottest_cell_xyz_m[0],
        "hottest_cell_y_m": pipeline.hottest_cell_xyz_m[1],
        "hottest_cell_z_m": pipeline.hottest_cell_xyz_m[2],
        "hottest_cell_material": pipeline.hottest_cell_material,
        "hottest_cell_component": pipeline.hottest_cell_component,
        "total_input_power_W": float(res.total_input_power_W),
        "total_boundary_heat_out_W": float(res.total_boundary_heat_out_W),
        "global_power_imbalance_W":
            float(res.global_power_imbalance_W),
        "relative_power_imbalance":
            float(res.relative_power_imbalance),
        "gpu_power_W": pipeline.gpu_power_W,
        "hbm_power_W": pipeline.hbm_power_W,
        "heat_out_by_rule_W": dict(pipeline.heat_out_by_rule_W),
        "discretization_seconds": float(pipeline.discretization_seconds),
        "conductance_seconds": float(pipeline.conductance_seconds),
        "operator_seconds": float(pipeline.operator_seconds),
        "solve_seconds": float(pipeline.solve_seconds),
        "total_seconds": float(
            pipeline.discretization_seconds
            + pipeline.conductance_seconds
            + pipeline.operator_seconds
            + pipeline.solve_seconds
        ),
    }


# ---------------------------------------------------------------------------
# Delta-Tmax computation
# ---------------------------------------------------------------------------

def compute_delta_tmax_sensitivity(
    rows: Sequence[dict],
    *,
    baseline_inset_m: float,
    baseline_mold_k_W_mK: float,
) -> dict[str, dict]:
    """Compute adjacent-pair ``ΔTmax`` for both sweep directions.

    The returned dict has keys ``inset_<from>_to_<to>`` (e.g.
    ``inset_0.250mm_to_0.500mm``) and
    ``mold_k_<from>_to_<to>`` (e.g. ``mold_k_3_to_10``).
    """
    deltas: dict[str, dict] = {}
    inset_rows = [r for r in rows if r["direction"] == "inset"]
    k_rows = [r for r in rows if r["direction"] == "mold_k"]

    for row in inset_rows:
        next_bigger = [r for r in inset_rows
                       if r["inset_m"] > row["inset_m"]]
        if not next_bigger:
            continue
        fine = min(next_bigger, key=lambda r: r["inset_m"])
        key = (f"inset_{row['inset_m']*1e3:.3f}mm_to_"
               f"{fine['inset_m']*1e3:.3f}mm")
        deltas[key] = {
            "direction": "inset",
            "coarse_label": row["label"],
            "fine_label": fine["label"],
            "coarse_inset_m": row["inset_m"],
            "fine_inset_m": fine["inset_m"],
            "coarse_max_temperature_K":
                float(row["max_temperature_K"]),
            "fine_max_temperature_K":
                float(fine["max_temperature_K"]),
            "delta_Tmax_K": float(
                fine["max_temperature_K"]
                - row["max_temperature_K"]),
        }

    for row in k_rows:
        next_bigger = [r for r in k_rows
                       if r["mold_k_W_mK"] > row["mold_k_W_mK"]]
        if not next_bigger:
            continue
        fine = min(next_bigger, key=lambda r: r["mold_k_W_mK"])
        key = (f"mold_k_{row['mold_k_W_mK']:g}_to_"
               f"{fine['mold_k_W_mK']:g}")
        deltas[key] = {
            "direction": "mold_k",
            "coarse_label": row["label"],
            "fine_label": fine["label"],
            "coarse_mold_k_W_mK": row["mold_k_W_mK"],
            "fine_mold_k_W_mK": fine["mold_k_W_mK"],
            "coarse_max_temperature_K":
                float(row["max_temperature_K"]),
            "fine_max_temperature_K":
                float(fine["max_temperature_K"]),
            "delta_Tmax_K": float(
                fine["max_temperature_K"]
                - row["max_temperature_K"]),
        }
    return deltas


# ---------------------------------------------------------------------------
# CSV / JSON writers
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "label", "direction", "inset_mm", "mold_k_W_mK",
    "cell_count", "internal_edge_count",
    "active_boundary_link_count", "adiabatic_face_count",
    "solver_method", "converged", "iterations", "matvec_count",
    "initial_residual", "final_relative_residual",
    "min_temperature_K", "max_temperature_K", "mean_temperature_K",
    "min_temperature_C", "max_temperature_C", "mean_temperature_C",
    "hottest_cell_id", "hottest_cell_x_m", "hottest_cell_y_m",
    "hottest_cell_z_m", "hottest_cell_material",
    "hottest_cell_component",
    "total_input_power_W", "total_boundary_heat_out_W",
    "global_power_imbalance_W", "relative_power_imbalance",
    "gpu_power_W", "hbm_power_W",
    "discretization_seconds", "conductance_seconds",
    "operator_seconds", "solve_seconds", "total_seconds",
]


def write_sensitivity_csv(rows: Sequence[dict],
                          path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_sensitivity_json(
    rows: Sequence[dict],
    delta_tmax: dict[str, dict],
    *,
    config_path: str | Path,
    inset_sizes_m: Sequence[float],
    k_values_W_mK: Sequence[float],
    baseline_inset_m: float,
    baseline_mold_k_W_mK: float,
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
            "sensitivity sweep on DRAM lateral inset and Mold k; "
            "parameters overridden in memory only",
        "strict_paper_temperature_reproduction": False,
        "caveat":
            ("The DRAM lateral inset (0.5 mm) and Mold k (3 W/m*K) "
             "are MODELING_CHOICE / DERIVED_FROM_PAPER_FIGURE values "
             "(see docs/benchmarks/hbm_on_gpu_12hi.md). This sweep "
             "does NOT introduce new physics; it only varies those "
             "two parameters and re-solves the same steady-state "
             "linear system."),
        "config_path": str(config_path),
        "solver_method": method,
        "solver_rtol": rtol,
        "initial_temperature_K": initial_temperature_K,
        "inset_sizes_m": list(inset_sizes_m),
        "k_values_W_mK": list(k_values_W_mK),
        "baseline_inset_m": baseline_inset_m,
        "baseline_mold_k_W_mK": baseline_mold_k_W_mK,
        "cases": [dict(r) for r in rows],
        "delta_Tmax": delta_tmax,
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# --resume / partial-row helpers
# ---------------------------------------------------------------------------

def case_already_done(rows_path: Path, label: str) -> bool:
    if not rows_path.exists():
        return False
    with rows_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return any(r.get("label") == label for r in reader)


def write_case_row_partial(
    rows_path: Path, row: dict,
    fields: Sequence[str] = _CSV_FIELDS,
) -> None:
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
    """
    if not rows_path.exists():
        return []
    with rows_path.open("r", encoding="utf-8", newline="") as stream:
        raw = list(csv.DictReader(stream))
    out: list[dict] = []
    for r in raw:
        row = dict(r)
        for key in (
            "cell_count", "internal_edge_count",
            "active_boundary_link_count", "adiabatic_face_count",
            "iterations", "matvec_count",
            "hottest_cell_id",
            "min_temperature_K", "max_temperature_K",
            "mean_temperature_K",
            "min_temperature_C", "max_temperature_C",
            "mean_temperature_C",
            "hottest_cell_x_m", "hottest_cell_y_m",
            "hottest_cell_z_m",
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
        for key in (
            "cell_count", "internal_edge_count",
            "iterations",
        ):
            if key in row:
                try:
                    row[key] = int(float(row[key]))
                except (TypeError, ValueError):
                    pass
        # The Mold k survives the CSV round-trip as a string
        # by default; cast it so ``compute_delta_tmax_sensitivity``
        # can format it with the ``:g`` spec.
        if "mold_k_W_mK" in row and row["mold_k_W_mK"] != "":
            try:
                row["mold_k_W_mK"] = float(row["mold_k_W_mK"])
            except (TypeError, ValueError):
                pass
        if "converged" in row:
            row["converged"] = str(row["converged"]).lower() in (
                "true", "1", "yes")
        if "inset_m" not in row and "inset_mm" in row:
            try:
                row["inset_m"] = float(row["inset_mm"]) * 1e-3
            except (TypeError, ValueError):
                pass
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Single-case runner (used by the CLI and the unit tests)
# ---------------------------------------------------------------------------

def run_single_sensitivity_case(
    yaml_path: str | Path,
    case: SensitivityCase,
    *,
    method: str = "pcg",
    rtol: float = 1e-6,
    max_iterations: int = 10_000,
    initial_temperature_K: float = 293.15,
) -> dict:
    """Run one sensitivity-sweep case and return the per-case
    summary row. The result is a flat dict matching the CSV
    columns; the underlying temperature array is discarded to
    keep the sweep memory-bounded."""
    from .config import compile_user_config
    raw = _load_yaml(yaml_path)
    overridden = _apply_overrides(
        raw,
        inset_m=case.inset_m,
        mold_k_W_mK=case.mold_k_W_mK,
    )
    config = SimulationConfig.model_validate(compile_user_config(overridden))
    pipeline = run_steady_pipeline(
        config,
        method=method,
        rtol=rtol,
        max_iterations=max_iterations,
        initial_temperature_K=initial_temperature_K,
    )
    return _row_from_pipeline(pipeline, case)
