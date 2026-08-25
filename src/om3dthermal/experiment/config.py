"""Strict three-layer configuration for formal workload-aware experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from om3dthermal.architecture import ArchitectureSpec
from om3dthermal.platform import PlatformSpec
from om3dthermal.workload import WorkloadSpec


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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
