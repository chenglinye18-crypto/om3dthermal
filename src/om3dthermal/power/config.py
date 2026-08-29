"""Configuration schema for standalone memory-power experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

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
    classification: Literal["PAPER_REPORTED"]
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


class FEOLParameterProvenance(StrictModel):
    classification: Literal[
        "MODELING_CHOICE", "MODELING_CHOICE_PLACEHOLDER"]
    status: Literal["CONDITIONAL_MODELING_CHOICE"]
    source_note: str


class FEOLWireInput(StrictModel):
    capacitance_fF_per_um: float = Field(gt=0.0)
    voltage_V: float = Field(gt=0.0)
    activity_factor: float = Field(ge=0.0, le=1.0)
    provenance: Literal["MODELING_CHOICE"]
    resistance_ohm_per_um: float | None = Field(default=None, gt=0.0)
    fixed_driver_resistance_ohm: float | None = Field(default=None, gt=0.0)
    fixed_load_pF: float | None = Field(default=None, gt=0.0)
    resistance_provenance: FEOLParameterProvenance | None = None
    driver_resistance_provenance: FEOLParameterProvenance | None = None
    load_capacitance_provenance: FEOLParameterProvenance | None = None

    @model_validator(mode="after")
    def latency_inputs_close(self) -> "FEOLWireInput":
        values = (
            self.resistance_ohm_per_um,
            self.fixed_driver_resistance_ohm,
            self.fixed_load_pF,
            self.resistance_provenance,
            self.driver_resistance_provenance,
            self.load_capacitance_provenance,
        )
        if any(value is not None for value in values) and any(
                value is None for value in values):
            raise ValueError(
                "FEOL latency requires complete resistance, driver, load, "
                "and provenance inputs")
        return self


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


class OperationSizeScalingInput(StrictModel):
    model: Literal["common_rc_linear_nrow"]
    reference_n_rows: int = Field(gt=0)
    reference_n_cols: int = Field(gt=0)
    provenance: Literal["MODELING_CHOICE"]


class CellModelInput(StrictModel):
    type: Literal[
        "dreamram_native", "component_replacement", "operation_table"
    ] = "dreamram_native"
    source: str | None = None
    geometry: CellGeometryInput | None = None
    replacement: CellReplacementInput | None = None
    operations: OperationTable | None = None
    size_scaling: OperationSizeScalingInput | None = None
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


class MIVResistanceProvenance(StrictModel):
    classification: Literal["MODELING_CHOICE"]
    status: Literal["CONDITIONAL_MODELING_CHOICE"]
    note: str


class MemoryInput(StrictModel):
    technology: str
    backend: Literal["dreamram", "unresolved"]
    dreamram: DreamRAMInput | None = None
    cell_model: CellModelInput = Field(default_factory=CellModelInput)

    @model_validator(mode="after")
    def backend_inputs(self) -> "MemoryInput":
        if self.backend == "dreamram" and self.dreamram is None:
            raise ValueError("dreamram backend requires memory.dreamram")
        if self.backend == "unresolved" and self.dreamram is not None:
            raise ValueError("unresolved memory must not select DreamRAM")
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
    miv_load_resistance_ohm: float | None = Field(default=None, gt=0.0)
    miv_resistance_ohm_per_um: float | Literal["unresolved"] | None = None
    miv_resistance_provenance: MIVResistanceProvenance | None = None

    @model_validator(mode="after")
    def miv_inputs(self) -> "TransportInput":
        if (self.type != "miv"
                and (self.miv_load_resistance_ohm is not None
                     or self.miv_resistance_ohm_per_um is not None
                     or self.miv_resistance_provenance is not None)):
            raise ValueError(
                "MIV resistance inputs are only valid for MIV transport")
        if isinstance(self.miv_resistance_ohm_per_um, (int, float)):
            if (not math.isfinite(self.miv_resistance_ohm_per_um)
                    or self.miv_resistance_ohm_per_um < 0.0):
                raise ValueError(
                    "miv_resistance_ohm_per_um must be finite, non-negative, "
                    "or unresolved")
            if self.miv_resistance_provenance is None:
                raise ValueError(
                    "resolved MIV resistance per length requires provenance")
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
    energy_status: Literal["CONDITIONAL_ASSUMPTION"] | None = None
    included_components: tuple[str, ...] = ()
    excluded_components: tuple[str, ...] = ()
    unconfirmed_components: tuple[str, ...] = ()
    source_boundary: str | None = None


class PhysicalAccessLatencyInput(StrictModel):
    mat_latency_ns: float = Field(gt=0.0)
    mat_classification: Literal["MODELING_CHOICE_PLACEHOLDER"]
    mat_status: Literal["NOT_CAPABILITY_VALIDATED"]
    mat_note: str
    interface_latency_ns: float = Field(ge=0.0)
    interface_classification: Literal["MODELING_PLACEHOLDER"]
    interface_status: Literal["NOT_YET_CALIBRATED"]
    interface_included_in_total: Literal[True]
    interface_note: str


class CoilBandwidthInput(StrictModel):
    links_per_die: int = Field(gt=0)
    data_rate_gbps_per_link: float = Field(gt=0.0)
    classification: Literal["DERIVED_FROM_ARCHITECTURE"]
    parameter_classification: Literal["MODELING_CHOICE"]
    link_count_source: Literal["ARCHITECTURE_CONTACTLESS_LINK_COUNT"]


class InternalBandwidthInput(StrictModel):
    model: Literal["FIRST_ORDER_PARALLEL_MEMORY_SERVICE_MODEL"]
    service_unit: Literal[
        "FEOL_IO_CHANNEL_ALIGNED_CLUSTER_ACCESS_GROUP"]
    parallel_units_per_slab_source: Literal["FEOL_IO_CHANNEL_COUNT"]
    parallel_slabs_source: Literal["GEOMETRY_MEMORY_REGION_COUNT"]
    clusters_per_service_source: Literal[
        "M3D_ACCESSED_CLUSTERS_PER_ACCESS"]
    read_payload_source: Literal["M3D_DELIVERED_BITS_PER_ACCESS"]
    service_cycle_source: Literal[
        "FIRST_ORDER_SERVICE_CYCLE_APPROXIMATION"]
    service_cycle_scale: float = Field(gt=0.0)
    classification: Literal["MODELING_CHOICE"]


class GPUInternalBandwidthInput(StrictModel):
    bandwidth_bytes_per_s: float | None = Field(default=None, gt=0.0)
    status: Literal[
        "GPU_INTERNAL_BW_NOT_MODELED_AS_BINDING",
        "NON_BINDING_NUMERICAL_CHOICE_NOT_HARDWARE_CAPABILITY",
    ]

    @model_validator(mode="after")
    def status_closure(self) -> "GPUInternalBandwidthInput":
        if (
            self.bandwidth_bytes_per_s is None
            and self.status != "GPU_INTERNAL_BW_NOT_MODELED_AS_BINDING"
        ):
            raise ValueError("unbounded GPU internal bandwidth needs None status")
        if (
            self.bandwidth_bytes_per_s is not None
            and self.status != (
                "NON_BINDING_NUMERICAL_CHOICE_NOT_HARDWARE_CAPABILITY")
        ):
            raise ValueError("finite GPU internal bandwidth needs numerical status")
        return self


class HierarchicalMemoryServiceInput(StrictModel):
    model: Literal["HIERARCHICAL_BANDWIDTH_MODEL"]
    die_count_source: Literal["GEOMETRY_MEMORY_REGION_COUNT"]
    coil: CoilBandwidthInput
    internal: InternalBandwidthInput
    gpu_internal: GPUInternalBandwidthInput


class GeometrySourceInput(StrictModel):
    """Existing thermal-geometry source for memory footprint constraints."""

    config: Path
    memory_region: Literal[
        "hbm_dram_die", "orthogonal_memory_slab", "orthogonal_m3d_slab"]


class ArchitectureInput(StrictModel):
    name: str | None = None
    layers: int | None = Field(default=None, gt=0)
    dies: int | None = Field(default=None, gt=0)
    geometry_source: GeometrySourceInput | None = None
    m3d_subarray: M3DSubarrayInput | None = None
    feol_route: FEOLRouteInput | None = None
    vertical: TransportInput
    base_route: BaseRouteInput
    interface: InterfaceInput
    physical_access_latency: PhysicalAccessLatencyInput | None = None
    memory_service: HierarchicalMemoryServiceInput | None = None
    logic_background_w: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def m3d_topology_required(self) -> "ArchitectureInput":
        if self.vertical.type == "miv" and self.m3d_subarray is None:
            raise ValueError(
                "MIV architecture requires architecture.m3d_subarray")
        if self.feol_route is not None and self.m3d_subarray is None:
            raise ValueError("FEOL route requires architecture.m3d_subarray")
        if self.physical_access_latency is not None:
            if self.vertical.type != "miv" or self.feol_route is None:
                raise ValueError(
                    "physical access latency requires MIV and FEOL routes")
        if self.memory_service is not None:
            if (
                self.physical_access_latency is None
                or self.m3d_subarray is None
                or self.feol_route is None
                or self.interface.type != "contactless"
            ):
                raise ValueError(
                    "hierarchical memory service requires contactless M3D "
                    "topology and physical latency")
        return self


class RowPolicy(StrictModel):
    activated_row_data_utilization: float = Field(gt=0.0, le=1.0)


class WorkloadInput(StrictModel):
    read_bandwidth_gbps: float = Field(ge=0.0)
    write_bandwidth_gbps: float = Field(ge=0.0)
    read_data: BinaryProbability | None = None
    write_transition: WriteProbability | None = None
    refresh_data: BinaryProbability | None = None
    row_policy: RowPolicy | None = None
    control_address_reuse: int | None = Field(default=None, gt=0)
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


class RefreshInput(StrictModel):
    enabled: bool
    model: Literal[
        "dreamram_internal_refresh", "operation_table_retention"
    ] | None = None
    refresh_window_s: float | None = Field(default=None, gt=0.0)
    refresh_window_provenance: str | None = None
    retention_reference_s: float | None = Field(default=None, gt=0.0)
    retention_reference_source: Literal["TANG_IEDM2023_IGZO_2T0C"] | None = None
    retention_reference_provenance: Literal["PAPER_REPORTED"] | None = None
    refresh_safety_factor: float | None = Field(default=None, gt=0.0)
    refresh_interval_provenance: Literal["MODELING_CHOICE"] | None = None

    @model_validator(mode="after")
    def model_inputs(self) -> "RefreshInput":
        if not self.enabled:
            if self.model is not None:
                raise ValueError("disabled refresh must not select a model")
            return self
        if self.model == "dreamram_internal_refresh":
            if self.refresh_window_s is None or self.refresh_window_provenance is None:
                raise ValueError(
                    "DreamRAM refresh requires window and provenance")
        elif self.model == "operation_table_retention":
            if (self.retention_reference_s is None
                    or self.retention_reference_source is None
                    or self.retention_reference_provenance != "PAPER_REPORTED"
                    or self.refresh_safety_factor is None
                    or self.refresh_interval_provenance != "MODELING_CHOICE"):
                raise ValueError(
                    "operation-table refresh requires retention reference, "
                    "safety factor, and explicit provenance")
        else:
            raise ValueError("enabled refresh requires a refresh model")
        return self


class EnableInput(StrictModel):
    enabled: bool


class GPUCasePowerInput(StrictModel):
    model: Literal["fixed"]
    power_W: float = Field(ge=0.0)


class MemoryCasePowerInput(StrictModel):
    model: Literal["analytical", "reference_fixed", "unresolved"]
    status: Literal[
        "VALIDATED", "CONDITIONAL_ASSUMPTION",
        "NOT_ANALYTICAL", "NOT_VALIDATED"]
    total_power_W: float | None = Field(default=None, ge=0.0)
    source: str | None = None
    provenance: str | None = None
    accounting_level: str | None = None

    @model_validator(mode="after")
    def mode_fields(self) -> "MemoryCasePowerInput":
        if self.model == "analytical":
            if (self.status not in {"VALIDATED", "CONDITIONAL_ASSUMPTION"}
                    or self.total_power_W is not None):
                raise ValueError(
                    "analytical memory power must be derived and explicitly "
                    "VALIDATED or CONDITIONAL_ASSUMPTION")
        elif self.model == "reference_fixed":
            if (self.status != "NOT_ANALYTICAL"
                    or self.total_power_W is None
                    or self.source is None
                    or self.provenance is None
                    or self.accounting_level is None):
                raise ValueError(
                    "reference_fixed requires total, source, provenance, "
                    "accounting level, and NOT_ANALYTICAL status")
        elif self.status != "NOT_VALIDATED" or self.total_power_W is not None:
            raise ValueError(
                "unresolved memory power must be NOT_VALIDATED with no total")
        return self


class PowerInput(StrictModel):
    refresh: RefreshInput
    background: EnableInput
    gpu: GPUCasePowerInput | None = None
    memory: MemoryCasePowerInput | None = None


class MemoryPowerConfig(StrictModel):
    memory: MemoryInput
    architecture: ArchitectureInput
    workload: WorkloadInput
    power: PowerInput

    @model_validator(mode="after")
    def workload_semantics(self) -> "MemoryPowerConfig":
        is_m3d = self.architecture.m3d_subarray is not None
        if is_m3d:
            if self.workload.row_policy is not None:
                raise ValueError(
                    "Orthogonal-M3D must not use conventional HBM row_policy")
            if self.workload.control_address_reuse is None:
                raise ValueError(
                    "Orthogonal-M3D requires workload.control_address_reuse")
        else:
            if self.workload.row_policy is None:
                raise ValueError(
                    "1T1C DreamRAM paths require workload.row_policy")
            if self.workload.control_address_reuse is not None:
                raise ValueError(
                    "control_address_reuse is specific to Orthogonal-M3D")
        return self


class CaseMemoryRegionInput(StrictModel):
    width_mm: float = Field(gt=0.0)
    height_mm: float = Field(gt=0.0)


class CaseOrthogonalGeometryInput(StrictModel):
    slab_count: int = Field(gt=0)
    cube_length_x_mm: float = Field(gt=0.0)
    slab_plane_y_mm: float = Field(gt=0.0)
    slab_height_z_mm: float = Field(gt=0.0)
    slab_pitch_x_um: float = Field(gt=0.0)
    slab_plane: Literal["y-z"]
    thickness_direction: Literal["global_x"]


class CaseOrthogonalSiStackGeometryInput(StrictModel):
    si_substrate_um: float = Field(gt=0.0)
    beol_um: float = Field(gt=0.0)
    daa_um: float = Field(gt=0.0)


class CaseM3DStackGeometryInput(StrictModel):
    si_substrate_um: float = Field(gt=0.0)
    feol_um: float = Field(gt=0.0)
    bitcell_layers: int = Field(gt=0)
    bitcell_layer_pitch_nm: float = Field(gt=0.0)
    beol_interconnect_um: float = Field(gt=0.0)
    daa_um: float = Field(gt=0.0)
    cell_area_um2: float = Field(gt=0.0)


class CaseGeometryInput(StrictModel):
    type: Literal["dreamram_hbm", "orthogonal_si", "orthogonal_m3d"]
    memory_region: CaseMemoryRegionInput | None = None
    capacity_instance_region: CaseMemoryRegionInput | None = None
    orthogonal: CaseOrthogonalGeometryInput | None = None
    m3d_stack: CaseM3DStackGeometryInput | None = None
    orthogonal_si_stack: CaseOrthogonalSiStackGeometryInput | None = None
    layout: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def type_fields(self) -> "CaseGeometryInput":
        if self.type in {"orthogonal_si", "orthogonal_m3d"}:
            if self.orthogonal is None:
                raise ValueError("orthogonal geometry requires orthogonal slab")
            if self.memory_region is not None:
                raise ValueError(
                    "orthogonal memory region is derived from slab geometry")
            if self.capacity_instance_region is not None:
                raise ValueError(
                    "orthogonal capacity is derived from slab geometry")
        if self.type == "orthogonal_m3d":
            if self.orthogonal is None or self.m3d_stack is None:
                raise ValueError(
                    "orthogonal_m3d geometry requires orthogonal and m3d_stack")
            bitcell_um = (
                self.m3d_stack.bitcell_layers
                * self.m3d_stack.bitcell_layer_pitch_nm * 1e-3)
            resolved = (
                self.m3d_stack.si_substrate_um + self.m3d_stack.feol_um
                + bitcell_um + self.m3d_stack.beol_interconnect_um
                + self.m3d_stack.daa_um)
            if not math.isclose(
                    resolved, self.orthogonal.slab_pitch_x_um,
                    rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"M3D slab thickness does not close: {resolved} um != "
                    f"slab pitch {self.orthogonal.slab_pitch_x_um} um")
            if self.orthogonal_si_stack is not None:
                raise ValueError("M3D geometry must not contain Si slab stack")
        elif self.type == "orthogonal_si":
            if self.orthogonal_si_stack is None or self.m3d_stack is not None:
                raise ValueError(
                    "orthogonal_si requires only orthogonal_si_stack")
            resolved = (
                self.orthogonal_si_stack.si_substrate_um
                + self.orthogonal_si_stack.beol_um
                + self.orthogonal_si_stack.daa_um)
            if not math.isclose(
                    resolved, self.orthogonal.slab_pitch_x_um,
                    rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("orthogonal Si slab thickness does not close")
        else:
            if self.memory_region is None:
                raise ValueError("dreamram_hbm requires a memory region")
            if (self.orthogonal is not None or self.m3d_stack is not None
                    or self.orthogonal_si_stack is not None):
                raise ValueError("dreamram_hbm must not duplicate an M3D stack")
        return self


class CanonicalCaseConfig(MemoryPowerConfig):
    name: str
    geometry: CaseGeometryInput
    thermal: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def canonical_sources(self) -> "CanonicalCaseConfig":
        if self.architecture.geometry_source is not None:
            raise ValueError(
                "canonical case must use its inline geometry, not geometry_source")
        is_m3d = self.architecture.m3d_subarray is not None
        if is_m3d != (self.geometry.type == "orthogonal_m3d"):
            raise ValueError("architecture and canonical geometry type disagree")
        if self.power.gpu is None or self.power.memory is None:
            raise ValueError("canonical case requires GPU and memory power modes")
        if self.power.gpu.power_W != 300.0:
            raise ValueError("active canonical research cases require GPU=300 W")
        if (self.power.memory.model == "analytical"
                and self.memory.backend != "dreamram"):
            raise ValueError("analytical memory power requires DreamRAM backend")
        if (self.power.memory.model == "unresolved"
                and self.memory.backend != "unresolved"):
            raise ValueError("unresolved power requires unresolved memory backend")
        return self


def load_power_config(path: str | Path) -> MemoryPowerConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("memory-power YAML root must be a mapping")
    return MemoryPowerConfig.model_validate(raw)


def load_case_config(path: str | Path) -> CanonicalCaseConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("canonical case YAML root must be a mapping")
    return CanonicalCaseConfig.model_validate(raw)


def find_project_root(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError(f"cannot locate project root from {path}")


def resolve_project_path(project_root: Path, configured: Path) -> Path:
    return configured if configured.is_absolute() else project_root / configured
