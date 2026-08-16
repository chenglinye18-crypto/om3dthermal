"""Config-driven OFAT memory parameter sweep benchmark.

The framework is intentionally thin: it never re-implements physics. For each
``(case, sweep_name, parameter_value)`` point it

  1. deep-copies the canonical ``CanonicalCaseConfig`` (no rewrite of YAML,
     no edit of any file in ``third_party/DreamRAM/``);
  2. applies one transient override, routing it through the same pydantic
     fields the production pipeline consumes (or, for MAT rows/cols, through
     a per-point JSON written under the sweep's own ``point_configs/``
     directory, leaving the pinned DreamRAM config untouched);
  3. calls the existing ``architecture_comparison.run_architecture_comparison``
     pipeline (case -> power -> thermal mapping -> compile_case_thermal ->
     run_steady_pipeline);
  4. writes a ``metrics.json`` and a ``resolved_case.yaml`` so the run is
     self-describing and re-runnable.

The CLI entry point is ``python -m om3dthermal sweep <sweep_yaml>`` and
mirrors the legacy ``sweep-sensitivity`` / ``sweep-mesh`` argument style.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .architecture_comparison import (
    _resolved_capacity,
    _temperature_maxima,
    compile_case_thermal,
    resolve_system_power,
)
from .case_runner import run_steady_pipeline
from .power.config import (
    CanonicalCaseConfig,
    find_project_root,
    load_case_config,
)
from .power.geometry import resolve_case_geometry


# ---------------------------------------------------------------------------
# Sweep configuration schema
# ---------------------------------------------------------------------------

# User-facing alias -> applicability (which canonical cases accept it).
# Keep this table small and explicit: every alias must have a clearly
# understood physical meaning and a known place in the model.
#
# The alias keys match the user-facing parameter names declared in the
# sweep YAML, so the YAML "parameter: <alias>" is a direct reference into
# this table.

ALLOWED_ALIASES: dict[str, dict[str, Any]] = {
    "activated_row_data_utilization": {
        "applies_to": {"dreamram_hbm", "orthogonal_si"},
        "physical_field": "workload.row_policy.activated_row_data_utilization",
        "type": "float",
        "description": (
            "Fraction of an activated Si DRAM row that is consumed by RD on "
            "average. effective_RD_per_ACT = utilization * atoms_per_page."
        ),
    },
    "mat_rows": {
        "applies_to": {"dreamram_hbm", "orthogonal_si"},
        "physical_field": "memory.dreamram.memory_config[mat.wordlines]",
        "type": "int",
        "description": (
            "Wordlines per MAT. Transient override written to a per-point "
            "JSON under point_configs/; the pinned DreamRAM config is "
            "untouched."
        ),
    },
    "mat_cols": {
        "applies_to": {"dreamram_hbm", "orthogonal_si"},
        "physical_field": "memory.dreamram.memory_config[mat.bitlines]",
        "type": "int",
        "description": (
            "Bitlines per MAT. Transient override written to a per-point "
            "JSON under point_configs/; the pinned DreamRAM config is "
            "untouched. Also shifts atoms_per_page and the minimum "
            "utilization floor."
        ),
    },
    "m3d_subarray_rows": {
        "applies_to": {"orthogonal_m3d"},
        "physical_field": "architecture.m3d_subarray.subarray.n_rows",
        "type": "int",
        "description": (
            "Subarray rows in the M3D Tang-style topology. Scales the Zhu "
            "operation energy as n_rows / reference_n_rows and the local "
            "RBL route length."
        ),
    },
    "m3d_subarray_cols": {
        "applies_to": {"orthogonal_m3d"},
        "physical_field": "architecture.m3d_subarray.subarray.n_cols",
        "type": "int",
        "description": (
            "Subarray columns in the M3D Tang-style topology. Recorded for "
            "provenance; only resizes the subarray geometry."
        ),
    },
}


class CaseRef(BaseModel):
    """Reference to one canonical case in the sweep config."""

    model_config = ConfigDict(extra="forbid")
    alias: str = Field(min_length=1)
    path: str = Field(min_length=1)


class SweepAxis(BaseModel):
    """One OFAT sweep axis: which parameter, on which cases, at which values."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    cases: list[str] = Field(min_length=1)
    parameter: str = Field(min_length=1)
    values: list[float] = Field(min_length=1)

    @field_validator("parameter")
    @classmethod
    def known_alias(cls, value: str) -> str:
        if value not in ALLOWED_ALIASES:
            raise ValueError(
                f"unknown sweep parameter {value!r}; allowed: "
                f"{sorted(ALLOWED_ALIASES)}"
            )
        return value


class SweepConfig(BaseModel):
    """Top-level sweep configuration."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    mode: Literal["ofat"] = "ofat"
    thermal: bool = True
    # Selects the matrix-free PCG backend for the steady-state solve.
    # ``"cpu"`` -> :func:`solve_pcg`; ``"gpu"`` -> :func:`solve_pcg_gpu`
    # (CuPy/NVRTC). This is a thin pass-through to the same routing
    # used by ``om3dthermal.cli solve-steady --backend <cpu|gpu>``;
    # the framework does not re-implement either solver.
    thermal_backend: Literal["cpu", "gpu"] = "cpu"
    cases: dict[str, CaseRef] = Field(min_length=1)
    sweeps: list[SweepAxis] = Field(min_length=1)
    output_dir: str = Field(min_length=1)

    def case_aliases(self) -> list[str]:
        return list(self.cases.keys())

    def resolve_case_path(self, alias: str, project_root: Path) -> Path:
        ref = self.cases[alias]
        p = Path(ref.path)
        if not p.is_absolute():
            p = project_root / p
        return p.resolve()


def load_sweep_config(path: str | Path) -> SweepConfig:
    p = Path(path).resolve()
    with p.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("sweep YAML root must be a mapping")
    return SweepConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Override application
# ---------------------------------------------------------------------------

def _enum_axes(sweep: SweepConfig) -> Iterable[tuple[str, SweepAxis, float]]:
    """Enumerate (case_alias, axis, value) triples."""
    for axis in sweep.sweeps:
        for case_alias in axis.cases:
            for value in axis.values:
                yield case_alias, axis, float(value)


def _hash_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _write_mat_override_json(
        baseline_json_path: Path, override_path: Path,
        *, mat_rows: int | None, mat_cols: int | None) -> None:
    """Copy the pinned DreamRAM baseline JSON, mutate the MAT dimensions,
    and write a new JSON under the sweep's point_configs/ directory.

    The pinned JSON under ``third_party/DreamRAM/`` is never modified.
    """
    data = json.loads(baseline_json_path.read_text())
    if mat_rows is not None:
        data["memconfig"]["mat"]["wordlines"] = int(mat_rows)
    if mat_cols is not None:
        data["memconfig"]["mat"]["bitlines"] = int(mat_cols)
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.write_text(json.dumps(data, indent=4))


def apply_override(
        case: CanonicalCaseConfig, project_root: Path, *,
        alias: str, value: float,
        override_dir: Path) -> tuple[CanonicalCaseConfig, dict[str, Any]]:
    """Return a deep-copied case with the OFAT override applied, plus a
    provenance record describing exactly which physical fields were touched.

    All overrides are routed through the same pydantic fields the production
    pipeline consumes. For MAT rows/cols the JSON path is replaced with a
    per-point JSON under ``override_dir``; the pinned DreamRAM config is
    never modified.
    """
    if alias not in ALLOWED_ALIASES:
        raise ValueError(f"unknown alias {alias!r}")
    arch_type = case.geometry.type
    if arch_type not in ALLOWED_ALIASES[alias]["applies_to"]:
        raise ValueError(
            f"alias {alias!r} does not apply to architecture {arch_type!r} "
            f"(applicable: {sorted(ALLOWED_ALIASES[alias]['applies_to'])})"
        )

    provenance: dict[str, Any] = {
        "alias": alias,
        "value": value,
        "physical_field": ALLOWED_ALIASES[alias]["physical_field"],
        "architecture_type": arch_type,
    }

    if alias == "activated_row_data_utilization":
        if case.workload.row_policy is None:
            raise ValueError("Si DRAM case must declare workload.row_policy")
        if not (0.0 < value <= 1.0):
            raise ValueError(
                f"activated_row_data_utilization must lie in (0, 1]; "
                f"got {value}")
        new_workload = case.workload.model_copy(update={
            "row_policy": case.workload.row_policy.model_copy(
                update={"activated_row_data_utilization": float(value)}),
        })
        provenance["old_value"] = (
            case.workload.row_policy.activated_row_data_utilization)
        new_case = case.model_copy(update={"workload": new_workload})
        provenance["new_value"] = float(value)

    elif alias in {"mat_rows", "mat_cols"}:
        if case.memory.dreamram is None:
            raise ValueError(
                f"alias {alias!r} requires memory.dreamram; got None")
        baseline_path = Path(case.memory.dreamram.memory_config)
        if not baseline_path.is_absolute():
            baseline_path = project_root / baseline_path
        baseline_path = baseline_path.resolve()
        baseline_sha1 = _hash_file(baseline_path)
        provenance["baseline_json_path"] = str(baseline_path)
        provenance["baseline_json_sha1"] = baseline_sha1

        # Read baseline to discover the current value and the other axis.
        baseline = json.loads(baseline_path.read_text())
        old_rows = int(baseline["memconfig"]["mat"]["wordlines"])
        old_cols = int(baseline["memconfig"]["mat"]["bitlines"])
        if alias == "mat_rows":
            new_rows = int(value)
            new_cols = old_cols
            provenance["old_value"] = old_rows
        else:
            new_rows = old_rows
            new_cols = int(value)
            provenance["old_value"] = old_cols

        # Build a stable per-point filename so reruns are idempotent.
        override_path = (
            override_dir / "point_configs" / f"mat_{new_rows}x{new_cols}.json"
        )
        _write_mat_override_json(
            baseline_path, override_path,
            mat_rows=new_rows, mat_cols=new_cols,
        )
        provenance["override_json_path"] = str(override_path)
        provenance["override_json_sha1"] = _hash_file(override_path)
        provenance["new_value"] = (
            new_rows if alias == "mat_rows" else new_cols)

        new_memory = case.memory.model_copy(update={
            "dreamram": case.memory.dreamram.model_copy(
                update={"memory_config": override_path}),
        })
        new_case = case.model_copy(update={"memory": new_memory})

    elif alias == "m3d_subarray_rows":
        if case.architecture.m3d_subarray is None:
            raise ValueError("M3D case must declare architecture.m3d_subarray")
        if int(value) <= 0:
            raise ValueError(f"m3d_subarray_rows must be positive; got {value}")
        new_sub = case.architecture.m3d_subarray.subarray.model_copy(
            update={"n_rows": int(value)})
        new_top = case.architecture.m3d_subarray.model_copy(
            update={"subarray": new_sub})
        new_arch = case.architecture.model_copy(
            update={"m3d_subarray": new_top})
        new_case = case.model_copy(update={"architecture": new_arch})
        provenance["old_value"] = case.architecture.m3d_subarray.subarray.n_rows
        provenance["new_value"] = int(value)

    elif alias == "m3d_subarray_cols":
        if case.architecture.m3d_subarray is None:
            raise ValueError("M3D case must declare architecture.m3d_subarray")
        if int(value) <= 0:
            raise ValueError(f"m3d_subarray_cols must be positive; got {value}")
        new_sub = case.architecture.m3d_subarray.subarray.model_copy(
            update={"n_cols": int(value)})
        new_top = case.architecture.m3d_subarray.model_copy(
            update={"subarray": new_sub})
        new_arch = case.architecture.model_copy(
            update={"m3d_subarray": new_top})
        new_case = case.model_copy(update={"architecture": new_arch})
        provenance["old_value"] = case.architecture.m3d_subarray.subarray.n_cols
        provenance["new_value"] = int(value)

    else:
        raise ValueError(f"alias {alias!r} not implemented in apply_override")

    return new_case, provenance


# ---------------------------------------------------------------------------
# One-point pipeline
# ---------------------------------------------------------------------------

@dataclass
class PointResult:
    case: str
    case_path: str
    sweep_name: str
    parameter: str
    parameter_value: float
    is_nominal_point: bool
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    override_provenance: dict[str, Any] = field(default_factory=dict)
    failure: dict[str, Any] = field(default_factory=dict)


def _system_metrics(
        case: CanonicalCaseConfig, project_root: Path,
        ) -> dict[str, Any]:
    geom = resolve_case_geometry(case)
    sys_pow = resolve_system_power(
        case, project_root=project_root, geometry=geom)
    capacity = _resolved_capacity(case, geom, sys_pow)
    diagnostics = sys_pow.diagnostics or {}
    memory = sys_pow.memory_result
    if memory is None:
        # All three canonical cases have analytical ``MemoryPowerResult``;
        # if the value is missing here, the power backend has silently
        # degraded to a reference-fixed mode and the per-component
        # decomposition would be fabricated. Fail loudly.
        raise RuntimeError(
            "sweep requires analytical MemoryPowerResult; "
            f"got memory_result=None for {case.name!r} "
            f"(model={sys_pow.memory_power_model}, "
            f"status={sys_pow.memory_power_status})")
    pj = lambda v: None if v is None else float(v)
    return {
        "system_capacity_GiB": float(capacity["system_capacity_GiB"]),
        "capacity_per_instance_GiB": float(capacity["capacity_per_instance_GiB"]),
        "instance_count": int(capacity["instance_count"]),
        "memory_plane_area_mm2": float(capacity["memory_plane_area_mm2"]),
        "memory_plane_density_Mb_mm2": float(
            capacity["memory_plane_density_Mb_mm2"]),
        "architecture_footprint_area_mm2": float(
            capacity["architecture_footprint_area_mm2"]),
        "architecture_footprint_density_Gb_mm2": float(
            capacity["architecture_footprint_density_Gb_mm2"]),
        # Read energy decomposition directly from MemoryPowerResult;
        # do not read energy fields from ``sys_pow.diagnostics`` (that
        # was the previous bug: ``E_memory_internal`` was aliased to the
        # access total).
        "E_memory_internal_pJ_per_bit": float(memory.E_memory_internal_pj_bit),
        "E_vertical_pJ_per_bit": float(memory.E_vertical_pj_bit),
        "E_feol_route_pJ_per_bit": float(memory.E_feol_route_pj_bit),
        "E_base_route_pJ_per_bit": float(memory.E_base_route_pj_bit),
        "E_interface_pJ_per_bit": float(memory.E_interface_pj_bit),
        "E_access_total_pJ_per_bit": float(memory.E_access_total_pj_bit),
        # Read per-component power directly from MemoryPowerResult.
        "P_access_W": float(memory.P_access_W),
        "P_refresh_W": pj(memory.P_refresh_W),
        "P_memory_background_W": pj(memory.P_memory_background_W),
        "P_base_logic_background_W": pj(memory.P_logic_background_W),
        # System-level totals come from ResolvedSystemPower; the
        # analytical ``memory.P_total_W`` is the same number when the
        # backend is component-aware, but the system path is the
        # canonical closure.
        "P_total_memory_W": float(sys_pow.resolved_total_memory_power_W),
        "P_memory_total_W": pj(memory.P_total_W),
        "P_gpu_W": float(sys_pow.gpu_power_W),
        "P_package_W": (
            float(sys_pow.gpu_power_W)
            + float(sys_pow.resolved_total_memory_power_W or 0.0)),
        # Geometrical bank-level quantities are not defined as a
        # single resolved value in the current ResolvedGeometry /
        # MemoryPowerResult schema; record N/A rather than fabricate.
        "packing_utilization": "N/A",
        "bank_count": "N/A",
        "bank_tile_dimensions": "N/A",
        "activated_row_data_utilization": (
            None if case.workload.row_policy is None
            else case.workload.row_policy.activated_row_data_utilization),
        "mat_rows": diagnostics.get("rows_per_mat"),
        "mat_cols": diagnostics.get("columns_per_mat"),
        "atoms_per_page": diagnostics.get("atoms_per_page"),
        "effective_RD_per_ACT": diagnostics.get("effective_rd_per_act"),
        "minimum_activated_row_data_utilization": (
            diagnostics.get(
                "minimum_activated_row_data_utilization_percent")),
        "atom_size_bits": diagnostics.get("atom_size_bits"),
        "m3d_subarray_rows": (
            None if case.architecture.m3d_subarray is None
            else case.architecture.m3d_subarray.subarray.n_rows),
        "m3d_subarray_cols": (
            None if case.architecture.m3d_subarray is None
            else case.architecture.m3d_subarray.subarray.n_cols),
    }


def _thermal_metrics(
        case: CanonicalCaseConfig, project_root: Path,
        system_metrics: dict[str, Any], *,
        backend: str = "cpu") -> dict[str, Any]:
    geom = resolve_case_geometry(case)
    sys_pow = resolve_system_power(
        case, project_root=project_root, geometry=geom)
    sim = compile_case_thermal(case, sys_pow)
    pipeline = run_steady_pipeline(
        sim, method="pcg", rtol=1e-6, max_iterations=10_000,
        initial_temperature_K=293.15, backend=backend)
    mem_t, gpu_t, pkg_t = _temperature_maxima(pipeline)
    mapped_actual = float(sum(pipeline.power.power_W))
    resolved = system_metrics["P_package_W"]
    return {
        "memory_Tmax_degC": mem_t,
        "gpu_Tmax_degC": gpu_t,
        "package_Tmax_degC": pkg_t,
        "delta_Tmax_K": pkg_t - 20.0,
        "thermal_backend": backend,
        "converged": bool(pipeline.result.converged),
        "iterations": int(pipeline.result.iterations),
        "final_relative_residual": float(
            pipeline.result.final_relative_residual),
        "relative_power_imbalance": float(
            pipeline.result.relative_power_imbalance),
        "cell_count": int(pipeline.cell_count),
        "internal_edge_count": int(pipeline.internal_edge_count),
        "hottest_cell_id": int(pipeline.hottest_cell_id),
        "hottest_cell_material": str(pipeline.hottest_cell_material),
        "hottest_cell_component": str(pipeline.hottest_cell_component),
        "resolved_package_power_W": float(resolved),
        "mapped_package_power_W": float(mapped_actual),
        "power_closure_absolute_error_W": float(
            abs(mapped_actual - resolved)),
        "power_closure_relative_error": float(
            abs(mapped_actual - resolved) / resolved) if resolved else None,
    }


def _run_one_point(
        case_alias: str, case_path: Path, axis: SweepAxis, value: float,
        is_nominal: bool, output_dir: Path, project_root: Path, *,
        backend: str = "cpu") -> PointResult:
    canonical = load_case_config(case_path)
    if axis.parameter not in ALLOWED_ALIASES:
        return PointResult(
            case=case_alias, case_path=str(case_path),
            sweep_name=axis.name, parameter=axis.parameter,
            parameter_value=value, is_nominal_point=is_nominal,
            status="FAIL",
            failure={"failure_stage": "config_validate",
                     "failure_type": "unknown_alias",
                     "failure_message": f"unknown alias {axis.parameter!r}"})
    try:
        new_case, provenance = apply_override(
            canonical, project_root, alias=axis.parameter, value=value,
            override_dir=output_dir)
    except Exception as exc:
        return PointResult(
            case=case_alias, case_path=str(case_path),
            sweep_name=axis.name, parameter=axis.parameter,
            parameter_value=value, is_nominal_point=is_nominal,
            status="FAIL",
            failure={"failure_stage": "override",
                     "failure_type": type(exc).__name__,
                     "failure_message": str(exc)})

    point_dir = (
        output_dir / "points" / case_alias / axis.name / axis.parameter
        / _safe_value_token(value))
    point_dir.mkdir(parents=True, exist_ok=True)

    # Resolved case YAML: record what the deep-copied case actually
    # looked like, plus the override provenance.
    resolved_path = point_dir / "resolved_case.yaml"
    try:
        resolved_dump = new_case.model_dump(mode="json", round_trip=True)
    except TypeError:
        resolved_dump = new_case.model_dump(mode="json")
    resolved_dump["sweep_override"] = {
        "source": "dreamram_runtime_override" if axis.parameter in {
            "mat_rows", "mat_cols"} else "config_deep_copy",
        "parameter": axis.parameter,
        "nominal_value": provenance.get("old_value"),
        "resolved_value": provenance.get("new_value"),
        "alias": axis.parameter,
        "value": value,
    }
    if axis.parameter in {"mat_rows", "mat_cols"}:
        resolved_dump["sweep_override"][
            "override_json_path"
        ] = provenance.get("override_json_path")
        resolved_dump["sweep_override"][
            "baseline_json_sha1"
        ] = provenance.get("baseline_json_sha1")
    with resolved_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved_dump, stream, sort_keys=False, allow_unicode=True)

    # Pipeline
    try:
        sys_metrics = _system_metrics(new_case, project_root)
    except Exception as exc:
        return PointResult(
            case=case_alias, case_path=str(case_path),
            sweep_name=axis.name, parameter=axis.parameter,
            parameter_value=value, is_nominal_point=is_nominal,
            status="FAIL",
            failure={"failure_stage": "system_power",
                     "failure_type": type(exc).__name__,
                     "failure_message": str(exc)},
            override_provenance=provenance)

    thermal_metrics: dict[str, Any] = {}
    if True:  # thermal: true is the only supported mode for v0
        try:
            thermal_metrics = _thermal_metrics(
                new_case, project_root, sys_metrics, backend=backend)
        except Exception as exc:
            return PointResult(
                case=case_alias, case_path=str(case_path),
                sweep_name=axis.name, parameter=axis.parameter,
                parameter_value=value, is_nominal_point=is_nominal,
                status="FAIL",
                failure={"failure_stage": "thermal",
                         "failure_type": type(exc).__name__,
                         "failure_message": str(exc),
                         "traceback": traceback.format_exc(limit=6)},
                metrics=sys_metrics, override_provenance=provenance)

        # Force a GC pass between points so the steady-state
        # per-cell power and temperature arrays from the previous
        # point are released before the next heavy solve.
        gc.collect()

        metrics = {**sys_metrics, **thermal_metrics}

    # Final closure / finiteness check
    nan_inf_fields = [
        k for k, v in metrics.items()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v))
    ]
    if nan_inf_fields:
        return PointResult(
            case=case_alias, case_path=str(case_path),
            sweep_name=axis.name, parameter=axis.parameter,
            parameter_value=value, is_nominal_point=is_nominal,
            status="FAIL",
            failure={"failure_stage": "postprocess",
                     "failure_type": "non_finite_metric",
                     "failure_message": "NaN/Inf in metrics",
                     "non_finite_fields": nan_inf_fields},
            metrics=metrics, override_provenance=provenance)
    if not thermal_metrics.get("converged", True):
        return PointResult(
            case=case_alias, case_path=str(case_path),
            sweep_name=axis.name, parameter=axis.parameter,
            parameter_value=value, is_nominal_point=is_nominal,
            status="FAIL",
            failure={"failure_stage": "thermal",
                     "failure_type": "non_convergence",
                     "failure_message": "PCG did not converge"},
            metrics=metrics, override_provenance=provenance)
    if thermal_metrics.get("power_closure_absolute_error_W", 0.0) > 1e-6:
        return PointResult(
            case=case_alias, case_path=str(case_path),
            sweep_name=axis.name, parameter=axis.parameter,
            parameter_value=value, is_nominal_point=is_nominal,
            status="FAIL",
            failure={"failure_stage": "thermal",
                     "failure_type": "power_closure",
                     "failure_message": "mapped thermal power does not "
                                        "close to resolved package power"},
            metrics=metrics, override_provenance=provenance)

    return PointResult(
        case=case_alias, case_path=str(case_path),
        sweep_name=axis.name, parameter=axis.parameter,
        parameter_value=value, is_nominal_point=is_nominal,
        status="PASS",
        metrics=metrics, override_provenance=provenance)


def _safe_value_token(value: float) -> str:
    return f"{value:.6g}".replace(".", "p").replace("-", "n")


# ---------------------------------------------------------------------------
# Top-level sweep driver
# ---------------------------------------------------------------------------

@dataclass
class SweepRunResult:
    config_path: str
    config_name: str
    output_dir: str
    metadata: dict[str, Any]
    point_results: list[PointResult]
    pass_count: int
    fail_count: int

    def as_summary_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for pt in self.point_results:
            row: dict[str, Any] = {
                "case": pt.case,
                "case_path": pt.case_path,
                "sweep_name": pt.sweep_name,
                "parameter": pt.parameter,
                "parameter_value": pt.parameter_value,
                "is_nominal_point": pt.is_nominal_point,
                "status": pt.status,
            }
            for key, value in pt.metrics.items():
                row[key] = value
            if pt.failure:
                row["failure_stage"] = pt.failure.get("failure_stage")
                row["failure_type"] = pt.failure.get("failure_type")
                row["failure_message"] = pt.failure.get("failure_message")
            rows.append(row)
        return rows


def _nominal_value_for_alias(alias: str, canonical: CanonicalCaseConfig,
                              project_root: Path) -> float:
    """Return the canonical-case nominal value for the given alias.

    Used to flag the sweep point whose value equals the nominal.
    """
    if alias == "activated_row_data_utilization":
        if canonical.workload.row_policy is None:
            return float("nan")
        return float(canonical.workload.row_policy.activated_row_data_utilization)
    if alias in {"mat_rows", "mat_cols"}:
        if canonical.memory.dreamram is None:
            return float("nan")
        bp = Path(canonical.memory.dreamram.memory_config)
        if not bp.is_absolute():
            bp = project_root / bp
        baseline = json.loads(bp.read_text())
        if alias == "mat_rows":
            return int(baseline["memconfig"]["mat"]["wordlines"])
        return int(baseline["memconfig"]["mat"]["bitlines"])
    if alias in {"m3d_subarray_rows", "m3d_subarray_cols"}:
        if canonical.architecture.m3d_subarray is None:
            return float("nan")
        sub = canonical.architecture.m3d_subarray.subarray
        return (
            int(sub.n_rows) if alias == "m3d_subarray_rows"
            else int(sub.n_cols))
    return float("nan")


def run_sweep(
        config_path: str | Path,
        *,
        repo_root: Path | None = None,
        git_metadata: dict[str, Any] | None = None) -> SweepRunResult:
    cfg = load_sweep_config(config_path)
    if repo_root is None:
        repo_root = Path(config_path).resolve().parent
        # walk up to find pyproject.toml
        for parent in (repo_root, *repo_root.parents):
            if (parent / "pyproject.toml").is_file():
                repo_root = parent
                break
    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist the resolved sweep config (so reruns and audits see what
    # the framework actually executed).
    resolved_cfg_path = output_dir / "sweep_config.resolved.yaml"
    with resolved_cfg_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg.model_dump(mode="json"), stream,
                       sort_keys=False, allow_unicode=True)

    metadata: dict[str, Any] = {
        "benchmark_name": cfg.name,
        "execution_mode": cfg.mode,
        "main_repo_commit": (git_metadata or {}).get("main_repo_commit"),
        "main_repo_branch": (git_metadata or {}).get("main_repo_branch"),
        "dreamram_commit": (git_metadata or {}).get("dreamram_commit"),
        "dreamram_branch": (git_metadata or {}).get("dreamram_branch"),
        "thermal_backend": cfg.thermal_backend,
        "python_version": _python_version(),
        "case_paths": {
            alias: str(cfg.resolve_case_path(alias, repo_root))
            for alias in cfg.case_aliases()
        },
        "sweep_config_path": str(Path(config_path).resolve()),
        "execution_started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    results: list[PointResult] = []
    for case_alias, axis, value in _enum_axes(cfg):
        if case_alias not in cfg.cases:
            results.append(PointResult(
                case=case_alias, case_path="<unknown>",
                sweep_name=axis.name, parameter=axis.parameter,
                parameter_value=value, is_nominal_point=False,
                status="FAIL",
                failure={"failure_stage": "config_validate",
                         "failure_type": "unknown_case_alias",
                         "failure_message": (
                             f"case alias {case_alias!r} is not declared in "
                             f"sweep config 'cases' block")}))
            continue
        case_path = cfg.resolve_case_path(case_alias, repo_root)
        try:
            canonical = load_case_config(case_path)
        except Exception as exc:
            results.append(PointResult(
                case=case_alias, case_path=str(case_path),
                sweep_name=axis.name, parameter=axis.parameter,
                parameter_value=value, is_nominal_point=False,
                status="FAIL",
                failure={"failure_stage": "load_case",
                         "failure_type": type(exc).__name__,
                         "failure_message": str(exc)}))
            continue
        nominal = _nominal_value_for_alias(
            axis.parameter, canonical, repo_root)
        is_nominal = (float(nominal) == float(value))
        t0 = time.perf_counter()
        pt = _run_one_point(
            case_alias, case_path, axis, value, is_nominal,
            output_dir, repo_root, backend=cfg.thermal_backend)
        dt = time.perf_counter() - t0
        # persist per-point metrics.json
        point_dir = (
            output_dir / "points" / case_alias / axis.name / axis.parameter
            / _safe_value_token(value))
        (point_dir / "metrics.json").write_text(
            json.dumps({
                "case": pt.case,
                "case_path": pt.case_path,
                "sweep_name": pt.sweep_name,
                "parameter": pt.parameter,
                "parameter_value": pt.parameter_value,
                "is_nominal_point": pt.is_nominal_point,
                "status": pt.status,
                "metrics": pt.metrics,
                "override_provenance": pt.override_provenance,
                "failure": pt.failure,
                "wall_seconds": dt,
            }, indent=2, sort_keys=True, default=str),
            encoding="utf-8")
        print(
            f"[sweep] {pt.status:>4} {case_alias}/{axis.name}/{axis.parameter}"
            f"={value} ({dt:.1f}s)")
        results.append(pt)

    # Write summary.csv
    rows = SweepRunResult(
        config_path=str(Path(config_path).resolve()),
        config_name=cfg.name,
        output_dir=str(output_dir),
        metadata=metadata,
        point_results=results,
        pass_count=sum(1 for r in results if r.status == "PASS"),
        fail_count=sum(1 for r in results if r.status == "FAIL"),
    )
    _write_summary_csv(output_dir / "summary.csv", rows)
    _write_failures_csv(output_dir / "failures.csv", rows)
    # Update metadata with run-end summary
    metadata.update({
        "execution_finished_utc": datetime.now(timezone.utc).isoformat(),
        "total_points": len(results),
        "pass_count": rows.pass_count,
        "fail_count": rows.fail_count,
    })
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return rows


def _write_summary_csv(path: Path, run: SweepRunResult) -> None:
    rows = run.as_summary_rows()
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_failures_csv(path: Path, run: SweepRunResult) -> None:
    failures = [r for r in run.point_results if r.status == "FAIL"]
    if not failures:
        # Write an empty file with header so downstream readers don't trip
        path.write_text(
            "case,sweep_name,parameter,parameter_value,"
            "failure_stage,failure_type,failure_message\n",
            encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "case", "sweep_name", "parameter", "parameter_value",
            "failure_stage", "failure_type", "failure_message"])
        for r in failures:
            f = r.failure
            writer.writerow([
                r.case, r.sweep_name, r.parameter, r.parameter_value,
                f.get("failure_stage"), f.get("failure_type"),
                f.get("failure_message")])


def _python_version() -> str:
    import sys
    return ".".join(str(x) for x in sys.version_info[:3])
