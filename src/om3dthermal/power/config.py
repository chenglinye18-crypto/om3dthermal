"""Configuration schema for standalone memory-power experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class RoutingElectricalInput(StrictModel):
    metal_width_nm: float = Field(gt=0.0)
    metal_thickness_nm: float = Field(gt=0.0)
    capacitance_fF_per_um: float = Field(gt=0.0)
    resistance_ohm_per_um: float = Field(gt=0.0)
    voltage_V: float = Field(gt=0.0)
    activity_factor: float = Field(ge=0.0, le=1.0)
    active_line_count: int = Field(gt=0)
    provenance: Literal["MODELING_CHOICE"]


class LocalMuxInput(StrictModel):
    footprint_height_f: float = Field(ge=0.0)
    energy_pj_per_selected_bit: float = Field(ge=0.0)
    provenance: Literal["MODELING_CHOICE"]


class SubarrayGridInput(StrictModel):
    nx: int | Literal["auto"] = "auto"
    ny: int | Literal["auto"] = "auto"

    @field_validator("nx", "ny")
    @classmethod
    def positive_override(cls, value: int | str) -> int | str:
        if value != "auto" and value <= 0:
            raise ValueError("explicit subarray count must be positive")
        return value


class SubarrayCoreInput(StrictModel):
    n_rows: int = Field(gt=0)
    n_cols: int = Field(gt=0)


class SubarrayClusterInput(StrictModel):
    subarrays_x: int = Field(gt=0)
    subarrays_y: int = Field(gt=0)
    grid: SubarrayGridInput = Field(default_factory=SubarrayGridInput)
    provenance: Literal["MODELING_CHOICE"]


class M3DSpacingInput(StrictModel):
    subarray_gap_x_f: float = Field(ge=0.0)
    subarray_gap_y_f: float = Field(ge=0.0)
    cluster_gap_x_f: float = Field(ge=0.0)
    cluster_gap_y_f: float = Field(ge=0.0)
    provenance: Literal["MODELING_CHOICE"]


class GlobalPeripheralInput(StrictModel):
    row_selection_band_f: float = Field(ge=0.0)
    column_write_selection_band_f: float = Field(ge=0.0)
    row_selection_band_axis: Literal["x", "y"]
    column_write_selection_band_axis: Literal["x", "y"]
    provenance: Literal["MODELING_CHOICE"]

    @model_validator(mode="after")
    def distinct_axes(self) -> "GlobalPeripheralInput":
        if (self.row_selection_band_axis
                == self.column_write_selection_band_axis):
            raise ValueError("shared global peripheral bands require distinct axes")
        return self


class M3DInterconnectInput(StrictModel):
    global_rwl: RoutingElectricalInput
    global_wwl: RoutingElectricalInput
    global_wbl: RoutingElectricalInput
    local_rbl: RoutingElectricalInput


class M3DAccessInput(StrictModel):
    accessed_subarrays_per_access: int = Field(gt=0)
    accessed_clusters_per_access: int = Field(gt=0)
    selected_bits_per_subarray: int = Field(gt=0)
    provenance: Literal["MODELING_CHOICE"]


class M3DSubarrayInput(StrictModel):
    type: Literal["tang_embedded_subarray"]
    subarray: SubarrayCoreInput
    subarray_cluster: SubarrayClusterInput
    spacing: M3DSpacingInput
    global_peripheral: GlobalPeripheralInput
    local_mux: LocalMuxInput
    interconnect: M3DInterconnectInput
    access: M3DAccessInput
    topology_provenance: Literal["MODELING_CHOICE"]


class FEOLWireInput(StrictModel):
    capacitance_fF_per_um: float = Field(gt=0.0)
    voltage_V: float = Field(gt=0.0)
    activity_factor: float = Field(ge=0.0, le=1.0)
    provenance: Literal["MODELING_CHOICE"]


class FEOLRouteInput(StrictModel):
    type: Literal["nearest_edge_io"]
    edge: Literal["x_min", "x_max", "y_min", "y_max"]
    io_channels: int = Field(gt=0)
    io_channel_distribution: Literal["uniform_centered_bins"]
    wire: FEOLWireInput
    access_assumption: Literal["UNIFORM_CLUSTER_ACCESS"]
    topology_provenance: Literal["MODELING_CHOICE"]
    io_channel_count_source: str


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
    source: Literal["dreamram", "constant", "miv_topology", "none"]
    energy_pj_per_bit: float | None = Field(default=None, ge=0.0)
    electrical_model: Literal["dreamram_length_scaled_reference"] | None = None
    vertical_serialization_factor: int | Literal["unresolved"] | None = None
    capacitance_fF: float | Literal["unresolved"] | None = None
    fixed_load_pF: float | None = Field(default=None, gt=0.0)
    fixed_load_provenance: Literal["MODELING_CHOICE"] | None = None

    @model_validator(mode="after")
    def miv_inputs(self) -> "TransportInput":
        if self.source == "miv_topology":
            if self.type != "miv":
                raise ValueError("miv_topology source requires type: miv")
            if self.electrical_model == "dreamram_length_scaled_reference":
                if (self.vertical_serialization_factor is not None
                        or self.capacitance_fF is not None):
                    raise ValueError(
                        "length-scaled MIV slope/serialization are resolved from "
                        "DreamRAM and must not be duplicated in power config")
                if self.fixed_load_pF is None:
                    raise ValueError(
                        "length-scaled MIV requires fixed_load_pF")
                if self.fixed_load_provenance != "MODELING_CHOICE":
                    raise ValueError(
                        "MIV fixed load must be marked MODELING_CHOICE")
                return self
            if self.vertical_serialization_factor is None:
                raise ValueError(
                    "miv_topology requires vertical_serialization_factor")
            if self.capacitance_fF is None:
                raise ValueError("miv_topology requires capacitance_fF")
            if (self.vertical_serialization_factor != "unresolved"
                    and self.vertical_serialization_factor <= 0):
                raise ValueError(
                    "resolved vertical serialization factor must be positive")
            if (self.capacitance_fF != "unresolved"
                    and self.capacitance_fF <= 0.0):
                raise ValueError("resolved MIV capacitance must be positive")
        return self


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
    memory_region: Literal[
        "hbm_dram_die", "orthogonal_memory_slab", "orthogonal_m3d_slab"]


class ArchitectureInput(StrictModel):
    name: str
    layers: int | None = Field(default=None, gt=0)
    dies: int | None = Field(default=None, gt=0)
    geometry_source: GeometrySourceInput
    m3d_subarray: M3DSubarrayInput | None = None
    feol_route: FEOLRouteInput | None = None
    vertical: TransportInput
    base_route: BaseRouteInput
    interface: InterfaceInput
    logic_background_w: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def m3d_topology_required(self) -> "ArchitectureInput":
        if self.vertical.type == "miv" and self.m3d_subarray is None:
            raise ValueError(
                "MIV architecture requires architecture.m3d_subarray")
        if self.feol_route is not None and self.m3d_subarray is None:
            raise ValueError("FEOL route requires architecture.m3d_subarray")
        return self


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
    layer_access_probability: tuple[float, ...] | None = None

    @field_validator("layer_access_probability")
    @classmethod
    def valid_layer_probability(
            cls, value: tuple[float, ...] | None) -> tuple[float, ...] | None:
        if value is None:
            return None
        if not value or any(probability < 0.0 for probability in value):
            raise ValueError("layer access probabilities must be non-negative")
        if not math.isclose(sum(value), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("layer access probabilities must sum to 1")
        return value


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
