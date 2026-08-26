"""Strict three-layer configuration for formal workload-aware experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from om3dthermal.architecture import ArchitectureSpec
from om3dthermal.platform import HostOffloadSpec, PlatformSpec
from om3dthermal.workload import WorkloadSpec
from om3dthermal.provenance import ProvenanceRecord
from om3dthermal.serving import MeasuredBatchCurvePoint


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class M3DParameterSensitivitySpec(_StrictFrozenModel):
    architecture_id: Literal["orthogonal_m3d_igzo"]
    interface_energy_pj_per_bit: tuple[float, ...]
    logic_background_w: tuple[float, ...]
    status: Literal["PARAMETRIC_SENSITIVITY"]

    @field_validator("interface_energy_pj_per_bit", "logic_background_w")
    @classmethod
    def _finite_unique_nonnegative(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        result = tuple(float(value) for value in values)
        if not result or any(not math.isfinite(value) or value < 0.0
                             for value in result):
            raise ValueError("sensitivity values must be finite and non-negative")
        if len(set(result)) != len(result):
            raise ValueError("sensitivity values must be unique")
        return result


class ExperimentScenarioSpec(_StrictFrozenModel):
    matched_payload_bandwidth_bits_per_second: float = Field(gt=0.0)
    bandwidth_status: Literal[
        "MATCHED_REFERENCE_NOT_CAPABILITY_VALIDATED"
    ]
    effective_compute_flops_per_second: float = Field(gt=0.0)
    compute_status: Literal["NUMERICAL_CHOICE_NOT_HARDWARE_VALIDATED"]
    reserved_capacity_bytes: int | float = Field(ge=0)
    rho_values: tuple[float, ...]
    unresolved_logic_background_policy: dict[
        str, Literal["REQUIRE_RESOLVED", "EXISTING_PLACEHOLDER_ZERO"]
    ]
    m3d_parameter_sensitivity: M3DParameterSensitivitySpec | None = None

    @field_validator("rho_values")
    @classmethod
    def _rho_values(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values:
            raise ValueError("rho_values must not be empty")
        result = tuple(float(value) for value in values)
        if any(not math.isfinite(value) or value < 0.0 for value in result):
            raise ValueError("rho_values must be finite and non-negative")
        if len(set(result)) != len(result):
            raise ValueError("rho_values must be unique")
        return result


class ExperimentSpec(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    experiment_id: str = Field(min_length=1)
    mode: Literal["conditional_llm_decode_e2e"]
    architecture_configs: tuple[Path, ...]
    platform_config: Path
    workload_config: Path
    scenario: ExperimentScenarioSpec
    output_dir: Path
    output_policy: Literal["ERROR_IF_EXISTS"] = "ERROR_IF_EXISTS"

    @field_validator("architecture_configs")
    @classmethod
    def _architectures(cls, values: tuple[Path, ...]) -> tuple[Path, ...]:
        if not values:
            raise ValueError("architecture_configs must not be empty")
        return values


class CapacityReferenceSpec(_StrictFrozenModel):
    reference_id: str = Field(min_length=1)
    status: Literal["RESOLVED", "UNRESOLVED"]
    usable_capacity_bytes: int | float | None = Field(default=None, ge=0)
    provenance: tuple[ProvenanceRecord, ...]

    @model_validator(mode="after")
    def _status_closure(self) -> "CapacityReferenceSpec":
        if not self.provenance:
            raise ValueError("capacity reference requires provenance")
        if self.status == "RESOLVED" and self.usable_capacity_bytes is None:
            raise ValueError("RESOLVED capacity reference requires usable capacity")
        if self.status == "UNRESOLVED" and self.usable_capacity_bytes is not None:
            raise ValueError("UNRESOLVED capacity reference must not guess capacity")
        return self


class ServingGPUPerformanceSpec(_StrictFrozenModel):
    backend: Literal["ANALYTICAL_CONDITIONAL", "MEASURED_BATCH_CURVE"]
    matched_payload_bandwidth_bits_per_second: float | None = Field(
        default=None, gt=0.0)
    effective_compute_flops_per_second: float | None = Field(
        default=None, gt=0.0)
    measured_points: tuple[MeasuredBatchCurvePoint, ...] = ()

    @model_validator(mode="after")
    def _backend_inputs(self) -> "ServingGPUPerformanceSpec":
        if self.backend == "ANALYTICAL_CONDITIONAL":
            if (
                self.matched_payload_bandwidth_bits_per_second is None
                or self.effective_compute_flops_per_second is None
                or self.measured_points
            ):
                raise ValueError(
                    "analytical GPU backend requires bandwidth/compute and no curve"
                )
        elif (
            not self.measured_points
            or self.matched_payload_bandwidth_bits_per_second is not None
            or self.effective_compute_flops_per_second is not None
        ):
            raise ValueError(
                "measured GPU backend requires points and no analytical inputs"
            )
        return self


class ServingSensitivityPointSpec(_StrictFrozenModel):
    sensitivity_id: str = Field(min_length=1)
    host_offload: HostOffloadSpec
    host_overlap_policy: Literal[
        "NO_OVERLAP", "FULL_OVERLAP", "PARTIAL_OVERLAP"]
    host_overlap_fraction: float = Field(ge=0.0, le=1.0)
    status: Literal[
        "MEASURED_CROSS_SYSTEM_SENSITIVITY",
        "PARAMETRIC_SENSITIVITY",
        "ANALYTICAL_UPPER_BOUND",
    ]

    @model_validator(mode="after")
    def _resolved_host_and_overlap(self) -> "ServingSensitivityPointSpec":
        if self.host_offload.status != "RESOLVED":
            raise ValueError("serving sensitivity host offload must be RESOLVED")
        if self.host_overlap_policy == "NO_OVERLAP" and self.host_overlap_fraction != 0:
            raise ValueError("NO_OVERLAP requires zero overlap fraction")
        if self.host_overlap_policy == "FULL_OVERLAP" and self.host_overlap_fraction != 1:
            raise ValueError("FULL_OVERLAP requires overlap fraction one")
        return self


class ServingScenarioSpec(_StrictFrozenModel):
    requested_requests: tuple[int, ...]
    reserved_capacity_bytes: int | float = Field(ge=0)
    host_overlap_policy: Literal[
        "NO_OVERLAP", "FULL_OVERLAP", "PARTIAL_OVERLAP"]
    host_overlap_fraction: float = Field(ge=0.0, le=1.0)
    gpu_performance: ServingGPUPerformanceSpec
    capacity_references: tuple[CapacityReferenceSpec, ...] = ()
    sensitivity_points: tuple[ServingSensitivityPointSpec, ...] = ()

    @field_validator("requested_requests")
    @classmethod
    def _request_points(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if (
            not values
            or len(set(values)) != len(values)
            or any(isinstance(value, bool) or value <= 0 for value in values)
        ):
            raise ValueError("requested_requests must be unique positive integers")
        return values

    @model_validator(mode="after")
    def _overlap_policy(self) -> "ServingScenarioSpec":
        if self.host_overlap_policy == "NO_OVERLAP" and self.host_overlap_fraction != 0:
            raise ValueError("NO_OVERLAP requires zero overlap fraction")
        if self.host_overlap_policy == "FULL_OVERLAP" and self.host_overlap_fraction != 1:
            raise ValueError("FULL_OVERLAP requires overlap fraction one")
        ids = [reference.reference_id for reference in self.capacity_references]
        if len(set(ids)) != len(ids):
            raise ValueError("capacity reference IDs must be unique")
        sensitivity_ids = [point.sensitivity_id for point in self.sensitivity_points]
        if len(set(sensitivity_ids)) != len(sensitivity_ids):
            raise ValueError("serving sensitivity IDs must be unique")
        return self


class ServingExperimentSpec(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    experiment_id: str = Field(min_length=1)
    mode: Literal["capacity_aware_serving"]
    architecture_configs: tuple[Path, ...]
    platform_config: Path
    workload_config: Path
    serving: ServingScenarioSpec
    output_dir: Path
    output_policy: Literal["ERROR_IF_EXISTS", "OVERWRITE"] = "ERROR_IF_EXISTS"

    @field_validator("architecture_configs")
    @classmethod
    def _architectures(cls, values: tuple[Path, ...]) -> tuple[Path, ...]:
        if not values:
            raise ValueError("architecture_configs must not be empty")
        return values


def _load_mapping(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _resolve_path(project_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def load_architecture_spec(path: str | Path, *, project_root: Path) -> ArchitectureSpec:
    config_path = _resolve_path(project_root, Path(path))
    spec = ArchitectureSpec.model_validate(_load_mapping(config_path))
    return spec.model_copy(update={
        "canonical_case": _resolve_path(project_root, spec.canonical_case)
    })


def load_platform_spec(path: str | Path, *, project_root: Path) -> PlatformSpec:
    return PlatformSpec.model_validate(
        _load_mapping(_resolve_path(project_root, Path(path))))


def load_workload_spec(path: str | Path, *, project_root: Path) -> WorkloadSpec:
    return WorkloadSpec.model_validate(
        _load_mapping(_resolve_path(project_root, Path(path))))


def load_experiment_spec(path: str | Path, *, project_root: Path) -> ExperimentSpec:
    config_path = _resolve_path(project_root, Path(path))
    spec = ExperimentSpec.model_validate(_load_mapping(config_path))
    return spec.model_copy(update={
        "architecture_configs": tuple(
            _resolve_path(project_root, value) for value in spec.architecture_configs),
        "platform_config": _resolve_path(project_root, spec.platform_config),
        "workload_config": _resolve_path(project_root, spec.workload_config),
        "output_dir": _resolve_path(project_root, spec.output_dir),
    })


def load_serving_experiment_spec(
    path: str | Path, *, project_root: Path
) -> ServingExperimentSpec:
    config_path = _resolve_path(project_root, Path(path))
    spec = ServingExperimentSpec.model_validate(_load_mapping(config_path))
    return spec.model_copy(update={
        "architecture_configs": tuple(
            _resolve_path(project_root, value) for value in spec.architecture_configs),
        "platform_config": _resolve_path(project_root, spec.platform_config),
        "workload_config": _resolve_path(project_root, spec.workload_config),
        "output_dir": _resolve_path(project_root, spec.output_dir),
    })
