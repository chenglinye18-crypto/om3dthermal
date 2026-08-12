"""Configuration schema for standalone memory-power experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BinaryProbability(StrictModel):
    p0: float = Field(ge=0.0, le=1.0)
    p1: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def sum_to_one(self) -> "BinaryProbability":
        if abs(self.p0 + self.p1 - 1.0) > 1e-12:
            raise ValueError("p0 and p1 must sum to 1")
        return self


class WriteProbability(StrictModel):
    p00: float = Field(ge=0.0, le=1.0)
    p01: float = Field(ge=0.0, le=1.0)
    p10: float = Field(ge=0.0, le=1.0)
    p11: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def sum_to_one(self) -> "WriteProbability":
        if abs(self.p00 + self.p01 + self.p10 + self.p11 - 1.0) > 1e-12:
            raise ValueError("write-transition probabilities must sum to 1")
        return self


class DreamRAMInput(StrictModel):
    memory_config: Path
    technology_config: Path


class OperationTable(StrictModel):
    read_0_pj_per_bit: float = Field(ge=0.0)
    read_1_pj_per_bit: float = Field(ge=0.0)
    write_00_pj_per_bit: float = Field(ge=0.0)
    write_01_pj_per_bit: float = Field(ge=0.0)
    write_10_pj_per_bit: float = Field(ge=0.0)
    write_11_pj_per_bit: float = Field(ge=0.0)
    refresh_0_pj_per_bit: float = Field(ge=0.0)
    refresh_1_pj_per_bit: float = Field(ge=0.0)


class OperationEnergyProvenance(StrictModel):
    source: Literal["IEDM2026_HaotongZhu_V5"]
    accounting_level: Literal["SPICE_EXTRACTED_MAT_LOCAL_OPERATION_ENERGY"]
    sensing_included: Literal[True]
    distributed_rc_included: Literal[True]
    accounting_note: str


class BackgroundInput(StrictModel):
    type: Literal["per_row", "per_bit", "per_die", "total"]
    value_w: float = Field(ge=0.0)


class CellGeometryProvenance(StrictModel):
    cell_area_um2: Literal["PAPER_REPORTED"]
    pitch_x_um: Literal["DERIVED_FROM_REFERENCE"]
    pitch_y_um: Literal["DERIVED_FROM_REFERENCE"]
    aspect_ratio: Literal["MODELING_CHOICE"]


class CellGeometryInput(StrictModel):
    """Physical pitches mapped by the adapter as X->BL and Y->WL."""

    cell_area_um2: float = Field(gt=0.0)
    pitch_x_um: float = Field(gt=0.0)
    pitch_y_um: float = Field(gt=0.0)
    aspect_ratio: float = Field(gt=0.0)
    provenance: CellGeometryProvenance

    @model_validator(mode="after")
    def geometry_closes(self) -> "CellGeometryInput":
        area = self.pitch_x_um * self.pitch_y_um
        if not math.isclose(
                area, self.cell_area_um2, rel_tol=1e-4, abs_tol=1e-12):
            raise ValueError(
                "cell geometry area does not close: "
                "pitch_x_um * pitch_y_um must equal cell_area_um2")
        ratio = self.pitch_x_um / self.pitch_y_um
        if not math.isclose(
                ratio, self.aspect_ratio, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                "cell geometry aspect ratio does not close: "
                "pitch_x_um / pitch_y_um must equal aspect_ratio")
        return self


class CellReplacementInput(StrictModel):
    mapping_status: Literal["validated", "unresolved"]
    components: tuple[str, ...]
    energy_source: Literal["component_values", "operation_table"] = (
        "component_values")
    component_energy_pj_per_bit: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_component_values(self) -> "CellReplacementInput":
        if len(set(self.components)) != len(self.components):
            raise ValueError("replacement components must be unique")
        if any(value < 0.0 for value in self.component_energy_pj_per_bit.values()):
            raise ValueError("replacement component energy must be non-negative")
        if (self.energy_source == "operation_table"
                and self.component_energy_pj_per_bit):
            raise ValueError(
                "operation_table replacement is one combined primitive; "
                "component_energy_pj_per_bit must be empty")
        return self


class CellModelInput(StrictModel):
    type: Literal[
        "dreamram_native", "component_replacement", "operation_table"
    ] = "dreamram_native"
    source: str | None = None
    geometry: CellGeometryInput | None = None
    replacement: CellReplacementInput | None = None
    operations: OperationTable | None = None
    operation_energy_provenance: OperationEnergyProvenance | None = None
    background: BackgroundInput | None = None
    retention_s: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def model_inputs(self) -> "CellModelInput":
        if self.type in {"component_replacement", "operation_table"}:
            if self.replacement is None:
                raise ValueError(f"{self.type} requires cell_model.replacement")
        if self.type == "operation_table" and self.operations is None:
            raise ValueError("operation_table requires cell_model.operations")
        if self.type == "operation_table":
            if self.operation_energy_provenance is None:
                raise ValueError(
                    "operation_table requires operation_energy_provenance")
            if (self.replacement is not None
                    and self.replacement.mapping_status == "validated"
                    and self.replacement.energy_source != "operation_table"):
                raise ValueError(
                    "validated operation_table replacement must use "
                    "energy_source: operation_table")
        return self


class MemoryInput(StrictModel):
    technology: str
    backend: Literal["dreamram"]
    dreamram: DreamRAMInput | None = None
    cell_model: CellModelInput = Field(default_factory=CellModelInput)

    @model_validator(mode="after")
    def backend_inputs(self) -> "MemoryInput":
        if self.backend == "dreamram" and self.dreamram is None:
            raise ValueError("dreamram backend requires memory.dreamram")
        return self


class TransportInput(StrictModel):
    type: Literal["tsv", "miv", "none"]
    source: Literal["dreamram", "constant", "none"]
    energy_pj_per_bit: float | None = Field(default=None, ge=0.0)


class BaseRouteInput(StrictModel):
    enabled: bool
    source: Literal["dreamram", "constant", "none"]
    energy_pj_per_bit: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def disabled_is_none(self) -> "BaseRouteInput":
        if not self.enabled and self.source != "none":
            raise ValueError("disabled base_route must use source: none")
        return self


class InterfaceInput(StrictModel):
    type: Literal["hbm_dq", "contactless", "direct"]
    source: Literal["dreamram", "constant", "none"]
    energy_pj_per_bit: float | None = Field(default=None, ge=0.0)


class GeometrySourceInput(StrictModel):
    """Existing thermal-geometry source for memory footprint constraints."""

    config: Path
    memory_region: Literal["hbm_dram_die", "orthogonal_memory_slab"]


class ArchitectureInput(StrictModel):
    name: str
    layers: int = Field(gt=0)
    dies: int | None = Field(default=None, gt=0)
    geometry_source: GeometrySourceInput
    vertical: TransportInput
    base_route: BaseRouteInput
    interface: InterfaceInput
    logic_background_w: float | None = Field(default=None, ge=0.0)


class RowPolicy(StrictModel):
    rd_per_act: int = Field(gt=0)


class WorkloadInput(StrictModel):
    read_bandwidth_gbps: float = Field(ge=0.0)
    write_bandwidth_gbps: float = Field(ge=0.0)
    read_data: BinaryProbability | None = None
    write_transition: WriteProbability | None = None
    refresh_data: BinaryProbability | None = None
    row_policy: RowPolicy | None = None
    stored_bits: float | None = Field(default=None, gt=0.0)
    active_rows: int | None = Field(default=None, ge=0)


class EnableInput(StrictModel):
    enabled: bool


class PowerInput(StrictModel):
    refresh: EnableInput
    background: EnableInput


class MemoryPowerConfig(StrictModel):
    memory: MemoryInput
    architecture: ArchitectureInput
    workload: WorkloadInput
    power: PowerInput


def load_power_config(path: str | Path) -> MemoryPowerConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("memory-power YAML root must be a mapping")
    return MemoryPowerConfig.model_validate(raw)


def find_project_root(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError(f"cannot locate project root from {path}")


def resolve_project_path(project_root: Path, configured: Path) -> Path:
    return configured if configured.is_absolute() else project_root / configured
