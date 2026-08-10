"""Pydantic v2 configuration schema and YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from .geometry.primitives import Footprint
from .materials import Material
from .units import (
    parse_areal_thermal_resistance,
    parse_heat_transfer_coefficient,
    parse_length,
    parse_power,
    parse_temperature,
)

Length = Annotated[float, BeforeValidator(parse_length)]
ArealThermalResistance = Annotated[
    float, BeforeValidator(parse_areal_thermal_resistance)]
Power = Annotated[float, BeforeValidator(parse_power)]
HeatTransferCoefficient = Annotated[
    float, BeforeValidator(parse_heat_transfer_coefficient)]
Temperature = Annotated[float, BeforeValidator(parse_temperature)]


class UnresolvedPhysicalParametersError(ValueError):
    """A research template is structurally valid but not solver-ready."""

    def __init__(self, architecture: str, parameters: list[str]):
        self.architecture = architecture
        self.parameters = tuple(parameters)
        joined = ", ".join(parameters)
        super().__init__(
            f"{architecture} geometry bookkeeping is valid but the config "
            f"cannot enter thermal material/operator/solve stages; "
            f"unresolved physical parameters: {joined}")


UnresolvedFloat = float | Literal["unresolved"]


class OrthogonalM3DArchitectureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["orthogonal_m3d_edram"]


class OrthogonalM3DArrayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slab_count: Annotated[int, Field(strict=True, gt=0)]
    cube_length_x_mm: Annotated[float, Field(gt=0)]
    slab_plane_y_mm: Annotated[float, Field(gt=0)]
    slab_height_z_mm: Annotated[float, Field(gt=0)]
    slab_pitch_x_um: Annotated[float, Field(gt=0)]
    daa_um: Annotated[float, Field(gt=0)]
    slab_plane: Literal["y-z"] = "y-z"
    thickness_direction: Literal["global_x"] = "global_x"
    placement: Literal["reuse_orthogonal_mosaic_array"] = (
        "reuse_orthogonal_mosaic_array")


class OrthogonalM3DSlabConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_pitch_um: Annotated[float, Field(gt=0)]
    si_substrate_um: Annotated[float, Field(gt=0)]
    feol_um: Annotated[float, Field(gt=0)]
    region_order: tuple[str, ...] = (
        "si_substrate", "feol", "m3d_bitcell_stack",
        "beol_interconnect", "daa")

    @field_validator("region_order")
    @classmethod
    def fixed_region_order(cls, value: tuple[str, ...]):
        expected = (
            "si_substrate", "feol", "m3d_bitcell_stack",
            "beol_interconnect", "daa")
        if value != expected:
            raise ValueError(
                "M3D slab region_order must be Si substrate -> FEOL -> "
                "M3D bit-cell stack -> BEOL interconnect -> DAA")
        return value


class M3DBEOLThermalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["effective_isotropic", "effective_anisotropic"]
    k_in_plane_W_mK: UnresolvedFloat
    k_cross_plane_W_mK: UnresolvedFloat

    @field_validator("k_in_plane_W_mK", "k_cross_plane_W_mK")
    @classmethod
    def positive_if_resolved(cls, value: UnresolvedFloat):
        if value != "unresolved" and value <= 0:
            raise ValueError("resolved effective M3D-BEOL k must be positive")
        return value


class M3DBEOLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bitcell_layers: Annotated[int, Field(strict=True, gt=0)]
    bitcell_layer_pitch_nm: Annotated[float, Field(gt=0)]
    bitcell_stack_um: Annotated[float, Field(gt=0)]
    interconnect_um: Annotated[float, Field(gt=0)]
    total_um: Annotated[float, Field(gt=0)]
    region_order: tuple[str, ...] = ("bitcell_stack", "interconnect")
    thermal: M3DBEOLThermalConfig

    @model_validator(mode="after")
    def derived_thicknesses(self):
        expected_stack_um = (
            self.bitcell_layers * self.bitcell_layer_pitch_nm / 1000.0)
        if abs(self.bitcell_stack_um - expected_stack_um) > 1e-12:
            raise ValueError(
                "m3d_beol.bitcell_stack_um must equal bitcell_layers * "
                "bitcell_layer_pitch_nm")
        if abs(self.total_um - (
                self.bitcell_stack_um + self.interconnect_um)) > 1e-12:
            raise ValueError(
                "m3d_beol.total_um must equal bitcell_stack_um + "
                "interconnect_um")
        if self.region_order != ("bitcell_stack", "interconnect"):
            raise ValueError(
                "m3d_beol.region_order must be bitcell_stack -> interconnect")
        return self


class M3DMemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    technology: Literal["CAA_IGZO_2T0C"]
    layers: Annotated[int, Field(strict=True, gt=0)]
    density_Mb_mm2_per_layer: Annotated[float, Field(gt=0)]
    cell_area_um2: Annotated[float, Field(gt=0)]
    slab_array_fill_factor: Annotated[float, Field(gt=0, le=1)]
    placement: Literal["within_beol_above_feol"]
    daa_between_m3d_layers: Literal[False]


class OrthogonalM3DPowerDistributionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["uniform_m3d_layers"]
    target_region: Literal["m3d_bitcell_stack"]
    direct_power_regions: tuple[str, ...] = ("m3d_bitcell_stack",)

    @field_validator("direct_power_regions")
    @classmethod
    def bitcell_stack_only(cls, value: tuple[str, ...]):
        if value != ("m3d_bitcell_stack",):
            raise ValueError(
                "M3D memory direct power must target only "
                "m3d_bitcell_stack")
        return value


class M3DIsoTotalPowerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memory_total_W: Annotated[float, Field(gt=0)]
    distribution: OrthogonalM3DPowerDistributionConfig
    cim_metrics_used_as_memory_power: Literal[False] = False


class M3DOperationEnergyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    read_0: Annotated[float, Field(ge=0)]
    read_1: Annotated[float, Field(ge=0)]
    write_0_to_0: Annotated[float, Field(ge=0)]
    write_0_to_1: Annotated[float, Field(ge=0)]
    write_1_to_0: Annotated[float, Field(ge=0)]
    write_1_to_1: Annotated[float, Field(ge=0)]
    refresh_0: Annotated[float, Field(ge=0)]
    refresh_1: Annotated[float, Field(ge=0)]


ProbabilityValue = float | Literal["unresolved"]
ActivityRate = float | Literal["unresolved"]
ActiveRows = int | Literal["unresolved"]


def _validate_probability(value: ProbabilityValue) -> ProbabilityValue:
    if value != "unresolved" and not 0.0 <= value <= 1.0:
        raise ValueError("resolved probability must be within [0, 1]")
    return value


class M3DStateProbabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    p0: ProbabilityValue
    p1: ProbabilityValue

    _probability_range = field_validator("p0", "p1")(_validate_probability)

    @model_validator(mode="after")
    def sum_if_resolved(self):
        if self.p0 != "unresolved" and self.p1 != "unresolved":
            if abs(float(self.p0) + float(self.p1) - 1.0) > 1e-12:
                raise ValueError("state probabilities p0 + p1 must equal 1")
        return self


class M3DWriteTransitionProbabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    p00: ProbabilityValue
    p01: ProbabilityValue
    p10: ProbabilityValue
    p11: ProbabilityValue

    _probability_range = field_validator(
        "p00", "p01", "p10", "p11")(_validate_probability)

    @model_validator(mode="after")
    def sum_if_resolved(self):
        values = (self.p00, self.p01, self.p10, self.p11)
        if all(value != "unresolved" for value in values):
            if abs(sum(float(value) for value in values) - 1.0) > 1e-12:
                raise ValueError(
                    "write transition probabilities must sum to 1")
        return self


class M3DOperationActivityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    read_bit_rate_per_s: ActivityRate
    write_bit_rate_per_s: ActivityRate
    read_state_probability: M3DStateProbabilityConfig
    write_transition_probability: M3DWriteTransitionProbabilityConfig
    refresh_period_s: ActivityRate
    refresh_state_probability: M3DStateProbabilityConfig
    active_rows: ActiveRows

    @field_validator(
        "read_bit_rate_per_s", "write_bit_rate_per_s", "refresh_period_s")
    @classmethod
    def nonnegative_rate_positive_period(cls, value, info):
        if value == "unresolved":
            return value
        if info.field_name == "refresh_period_s":
            if value <= 0:
                raise ValueError("refresh_period_s must be positive")
        elif value < 0:
            raise ValueError(f"{info.field_name} must be nonnegative")
        return value

    @field_validator("active_rows")
    @classmethod
    def nonnegative_active_rows(cls, value: ActiveRows):
        if value != "unresolved" and value < 0:
            raise ValueError("active_rows must be nonnegative")
        return value

    def unresolved_parameters(self) -> list[str]:
        candidates = {
            "read_bit_rate_per_s": self.read_bit_rate_per_s,
            "write_bit_rate_per_s": self.write_bit_rate_per_s,
            "read_state_probability.p0": self.read_state_probability.p0,
            "read_state_probability.p1": self.read_state_probability.p1,
            "write_transition_probability.p00": (
                self.write_transition_probability.p00),
            "write_transition_probability.p01": (
                self.write_transition_probability.p01),
            "write_transition_probability.p10": (
                self.write_transition_probability.p10),
            "write_transition_probability.p11": (
                self.write_transition_probability.p11),
            "refresh_period_s": self.refresh_period_s,
            "refresh_state_probability.p0": (
                self.refresh_state_probability.p0),
            "refresh_state_probability.p1": (
                self.refresh_state_probability.p1),
            "active_rows": self.active_rows,
        }
        return [name for name, value in candidates.items()
                if value == "unresolved"]


class M3DOperationEnergyPowerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_energy_fJ_per_bit: M3DOperationEnergyConfig
    hold_power_W_per_row: Annotated[float, Field(ge=0)]
    activity: M3DOperationActivityConfig
    nominal_workload: "M3DNominalArrayReadWorkloadConfig"
    distribution: OrthogonalM3DPowerDistributionConfig
    energy_provenance: Literal["PAPER_REPORTED"]
    cim_metrics_used_as_memory_power: Literal[False] = False


class OrthogonalM3DPowerModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    iso_total: M3DIsoTotalPowerConfig
    operation_energy: M3DOperationEnergyPowerConfig


class M3DNominalArrayReadWorkloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    delivered_bandwidth_bit_per_s: Annotated[float, Field(gt=0)]
    read_fraction: Annotated[float, Field(ge=0, le=1)]
    write_fraction: Annotated[float, Field(ge=0, le=1)]
    read_state_probability: M3DStateProbabilityConfig
    included_power_terms: tuple[str, ...] = ("array_read",)
    power_scope: Literal["array_core_power_only"]
    bandwidth_provenance: Literal["MATCHED_DELIVERED_BANDWIDTH_REFERENCE"]

    @model_validator(mode="after")
    def nominal_read_contract(self):
        if abs(self.read_fraction + self.write_fraction - 1.0) > 1e-12:
            raise ValueError("read_fraction + write_fraction must equal 1")
        if self.read_fraction != 1.0 or self.write_fraction != 0.0:
            raise ValueError(
                "M3D-v1 nominal workload must be read_fraction=1 and "
                "write_fraction=0")
        probabilities = self.read_state_probability
        if probabilities.p0 == "unresolved" or probabilities.p1 == "unresolved":
            raise ValueError(
                "nominal array-read state probabilities must be resolved")
        if self.included_power_terms != ("array_read",):
            raise ValueError(
                "M3D-v1 nominal workload includes only array_read power")
        return self


class OrthogonalM3DPowerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_mode: Literal["operation_energy"]


class OrthogonalM3DPaperMetricsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    energy_efficiency_TOPS_W: Annotated[float, Field(gt=0)]
    compute_density_TOPS_mm2: Annotated[float, Field(gt=0)]


class OrthogonalM3DTemplateConfig(BaseModel):
    """Paper-parameter template that is intentionally not solver-ready."""

    model_config = ConfigDict(extra="forbid")
    name: str
    architecture: OrthogonalM3DArchitectureConfig
    orthogonal: OrthogonalM3DArrayConfig
    slab: OrthogonalM3DSlabConfig
    m3d_beol: M3DBEOLConfig
    m3d_memory: M3DMemoryConfig
    power: OrthogonalM3DPowerConfig
    power_models: OrthogonalM3DPowerModelsConfig
    paper_metrics: OrthogonalM3DPaperMetricsConfig
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def pitch_and_layer_contract(self):
        if abs(self.slab.total_pitch_um
               - self.orthogonal.slab_pitch_x_um) > 1e-12:
            raise ValueError(
                "slab.total_pitch_um must equal orthogonal.slab_pitch_x_um")
        array_length_mm = (
            self.orthogonal.slab_count
            * self.orthogonal.slab_pitch_x_um / 1000.0)
        if array_length_mm > self.orthogonal.cube_length_x_mm + 1e-12:
            raise ValueError(
                "orthogonal slab array length exceeds cube_length_x_mm")
        if self.m3d_beol.bitcell_layers != self.m3d_memory.layers:
            raise ValueError(
                "m3d_beol.bitcell_layers must equal m3d_memory.layers")
        expected_si_um = (
            self.slab.total_pitch_um - self.slab.feol_um
            - self.m3d_beol.total_um - self.orthogonal.daa_um)
        if abs(self.slab.si_substrate_um - expected_si_um) > 1e-9:
            raise ValueError(
                "slab.si_substrate_um must be derived by pitch closure")
        closure_um = (
            self.slab.si_substrate_um + self.slab.feol_um
            + self.m3d_beol.bitcell_stack_um
            + self.m3d_beol.interconnect_um + self.orthogonal.daa_um)
        if abs(closure_um - self.slab.total_pitch_um) > 1e-9:
            raise ValueError(
                "Si + FEOL + bit-cell stack + interconnect + DAA must "
                "equal slab.total_pitch_um")
        return self

    def unresolved_physical_parameters(self) -> list[str]:
        candidates = {
            "m3d_beol.thermal.k_in_plane_W_mK": (
                self.m3d_beol.thermal.k_in_plane_W_mK),
            "m3d_beol.thermal.k_cross_plane_W_mK": (
                self.m3d_beol.thermal.k_cross_plane_W_mK),
        }
        return [name for name, value in candidates.items()
                if value == "unresolved"]

    def capacity_bookkeeping(self) -> dict[str, float]:
        slab_area_mm2 = (
            self.orthogonal.slab_plane_y_mm
            * self.orthogonal.slab_height_z_mm)
        capacity_per_layer_Mb = (
            self.m3d_memory.density_Mb_mm2_per_layer
            * slab_area_mm2 * self.m3d_memory.slab_array_fill_factor)
        capacity_per_slab_Mb = (
            capacity_per_layer_Mb * self.m3d_memory.layers)
        capacity_cube_Mb = (
            capacity_per_slab_Mb * self.orthogonal.slab_count)
        return {
            "slab_area_mm2": slab_area_mm2,
            "capacity_per_layer_Mb": capacity_per_layer_Mb,
            "capacity_per_slab_Mb": capacity_per_slab_Mb,
            "capacity_cube_Mb": capacity_cube_Mb,
            "capacity_cube_Gb_decimal": capacity_cube_Mb / 1000.0,
            "capacity_cube_GB_decimal": capacity_cube_Mb / 8000.0,
        }


class LateralInset(BaseModel):
    """Per-edge lateral inset applied to a ``Layer``'s parent footprint.

    The shorthand form ``{"x": v, "y": v}`` is normalised to the explicit
    four-edge form ``{"x_minus": v, "x_plus": v, "y_minus": v, "y_plus": v}``.
    The four values are all non-negative; their pairwise sums on each axis
    must leave a strictly positive remainder for the central entity.
    """

    model_config = ConfigDict(extra="forbid")
    x_minus: Length = 0.0
    x_plus: Length = 0.0
    y_minus: Length = 0.0
    y_plus: Length = 0.0

    @model_validator(mode="before")
    @classmethod
    def expand_shorthand(cls, data):
        if not isinstance(data, dict):
            raise TypeError("lateral_inset must be a mapping")
        # Copy the input so we never mutate the caller's dict (e.g. the
        # raw YAML mapping or any downstream user-side reference).
        data = dict(data)
        if "x" in data:
            x = data["x"]
            data.setdefault("x_minus", x)
            data.setdefault("x_plus", x)
            del data["x"]
        if "y" in data:
            y = data["y"]
            data.setdefault("y_minus", y)
            data.setdefault("y_plus", y)
            del data["y"]
        return data

    @field_validator("x_minus", "x_plus", "y_minus", "y_plus")
    @classmethod
    def non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("lateral_inset edge values must be >= 0")
        return value

    def is_zero(self) -> bool:
        return (self.x_minus == 0 and self.x_plus == 0
                and self.y_minus == 0 and self.y_plus == 0)


class Layer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["layer"] = "layer"
    name: str
    material: str
    thickness: Length
    tags: dict[str, Any] = Field(default_factory=dict)
    lateral_inset: LateralInset | None = None

    @field_validator("thickness")
    @classmethod
    def positive_thickness(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("layer thickness must be strictly positive")
        return value


class RepeatBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["repeat"]
    count: Annotated[int, Field(strict=True, gt=0)]
    layers: list[Layer]

    @field_validator("layers")
    @classmethod
    def nonempty(cls, value: list[Layer]) -> list[Layer]:
        if not value:
            raise ValueError("repeat block must contain at least one layer")
        return value


StackItem = Annotated[Layer | RepeatBlock, Field(discriminator="kind")]


class ExpandedLayer(BaseModel):
    name: str
    material: str
    thickness: float
    tags: dict[str, Any]
    source_suffix: str
    lateral_inset: LateralInset | None = None


class StackTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[StackItem]

    @field_validator("items")
    @classmethod
    def nonempty(cls, value: list[StackItem]) -> list[StackItem]:
        if not value:
            raise ValueError("stack template must contain at least one item")
        return value

    def expand(self) -> list[ExpandedLayer]:
        expanded: list[ExpandedLayer] = []
        for item_index, item in enumerate(self.items):
            if isinstance(item, Layer):
                expanded.append(ExpandedLayer(
                    name=item.name, material=item.material, thickness=item.thickness,
                    tags=dict(item.tags), source_suffix=f"items[{item_index}]",
                    lateral_inset=item.lateral_inset))
            else:
                for repeat_index in range(1, item.count + 1):
                    for layer_index, layer in enumerate(item.layers):
                        expanded.append(ExpandedLayer(
                            name=f"{layer.name}_{repeat_index:02d}", material=layer.material,
                            thickness=layer.thickness,
                            tags={**layer.tags, "repeat_index": repeat_index},
                            source_suffix=(f"items[{item_index}].layers[{layer_index}]"
                                           f"#repeat={repeat_index}"),
                            lateral_inset=layer.lateral_inset))
        names = [layer.name for layer in expanded]
        if len(names) != len(set(names)):
            raise ValueError("expanded layer names are not unique")
        return expanded

    @property
    def total_thickness(self) -> float:
        return sum(layer.thickness for layer in self.expand())


class StackPlacement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    footprint: str
    stack: str


class ColumnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    footprint: str
    stack: str | None = None
    material: str | None = None
    match_height_of: str | None = None
    fill_above: str | None = None
    priority: int = 10
    tags: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_mode(self):
        if (self.stack is None) == (self.material is None):
            raise ValueError("column must specify exactly one of stack or material")
        if self.material is not None and self.match_height_of is None:
            raise ValueError("single-material column requires match_height_of")
        return self


class MemoryZoneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    footprint: str
    reference_stack: str
    background_material: str
    background_priority: int = 0
    columns: list[ColumnConfig]


class HorizontalStructureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    foundation: StackPlacement
    gpu: StackPlacement
    memory_zone: MemoryZoneConfig
    top: StackPlacement


class OrthogonalDieLayerConfig(BaseModel):
    """One material layer through an orthogonal memory die thickness."""

    model_config = ConfigDict(extra="forbid")
    name: str
    material: str
    thickness: Length
    role: str

    @field_validator("thickness")
    @classmethod
    def positive_thickness(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("orthogonal die layer thickness must be positive")
        return value


class OrthogonalMemoryDieConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: Annotated[int, Field(strict=True, gt=0)]
    width: Length
    height: Length
    layers: list[OrthogonalDieLayerConfig]
    power_per_die: Power = 0.0

    @field_validator("width", "height")
    @classmethod
    def positive_dimension(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("orthogonal die dimensions must be positive")
        return value

    @field_validator("layers")
    @classmethod
    def nonempty_layers(cls, value: list[OrthogonalDieLayerConfig]):
        if not value:
            raise ValueError("orthogonal memory die requires at least one layer")
        return value

    @property
    def thickness(self) -> float:
        return sum(layer.thickness for layer in self.layers)


class OrthogonalAdhesiveConfig(BaseModel):
    """Package-level bond layer between the GPU and the MOSAIC cube."""

    model_config = ConfigDict(extra="forbid")
    material: str
    thickness: Length

    @field_validator("thickness")
    @classmethod
    def positive_thickness(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("orthogonal HBM adhesive thickness must be positive")
        return value


class OrthogonalHBMStructureConfig(BaseModel):
    """Parametric vertical-die MOSAIC cube placed above the GPU."""

    model_config = ConfigDict(extra="forbid")
    cube_footprint: str
    cube_height: Length
    background_material: str = "Mold"
    foundation: StackPlacement
    gpu: StackPlacement
    top: StackPlacement
    adhesive: OrthogonalAdhesiveConfig
    memory_die: OrthogonalMemoryDieConfig

    @field_validator("cube_height")
    @classmethod
    def positive_height(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("orthogonal HBM cube height must be positive")
        return value


class CellSizeConfig(BaseModel):
    """Per-axis uniform maximum cell size. All three must be strictly
    positive; equality is rejected so the discretisation always
    introduces at least one subdivision plane per axis.
    """

    model_config = ConfigDict(extra="forbid")
    x: Length
    y: Length
    z: Length

    @field_validator("x", "y", "z")
    @classmethod
    def strictly_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("max_cell_size.{x,y,z} must be strictly positive")
        return value


class DiscretizationConfig(BaseModel):
    """Discretisation settings. Only the boundary-preserving mode is
    supported in this stage; setting ``preserve_box_boundaries=False``
    raises ``NotImplementedError`` at discretisation time so the
    behaviour is explicit rather than silent.
    """

    model_config = ConfigDict(extra="forbid")
    max_cell_size: CellSizeConfig
    preserve_box_boundaries: bool = True

    @field_validator("preserve_box_boundaries")
    @classmethod
    def only_boundary_preserving_supported(cls, value: bool) -> bool:
        if not value:
            raise ValueError("preserve_box_boundaries = false is not "
                             "implemented yet; this stage only supports the "
                             "boundary-preserving mode")
        return value


class InterfaceResistanceConfig(BaseModel):
    """Optional per-material-pair interface areal resistance.

    ``materials`` is treated as an **unordered** pair; ``[A, B]`` and
    ``[B, A]`` are equivalent and the registry will reject duplicates.
    Same-material pairs are accepted but the default
    ``R'' = 0 m^2*K/W`` is the only sane choice for internal faces.
    """

    model_config = ConfigDict(extra="forbid")
    materials: tuple[str, str]
    areal_resistance: ArealThermalResistance
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThermalConductanceConfig(BaseModel):
    """Settings for the per-edge internal face conductance.

    Only the ``axis_aligned_only`` rotation policy is implemented in
    this stage. The default interface ``R''`` is applied when no
    explicit unordered pair rule matches. Setting a non-axis-aligned
    rotation on a ``ThermalCell`` will raise
    :class:`UnsupportedMaterialRotationError` at conductance build
    time.
    """

    model_config = ConfigDict(extra="forbid")
    rotation_policy: Literal["axis_aligned_only"] = "axis_aligned_only"
    default_interface_areal_resistance: ArealThermalResistance = 0.0
    interfaces: list[InterfaceResistanceConfig] = Field(default_factory=list)


class BoundarySelector(BaseModel):
    """Selector for matching a ``BoundaryFace`` to a rule.

    All fields are optional; a face is matched if every non-``None``
    field agrees. ``priority`` decides which of the matching rules
    wins (highest first); ties on the same selector fields are a
    config error.
    """

    model_config = ConfigDict(extra="forbid")
    component: str | None = None
    material: str | None = None
    layer: str | None = None
    axis: Literal["x", "y", "z"] | None = None
    side: Literal["minus", "plus"] | None = None
    classification: Literal[
        "scene_outer_boundary", "exposed_internal_boundary"] | None = None
    tags: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0


class BoundaryConditionConfig(BaseModel):
    """A single boundary condition rule.

    The three kinds have mutually exclusive parameter sets:

    - ``adiabatic``: no ``h``, ``ambient_temperature`` or
      ``surface_temperature`` may be set.
    - ``convection``: requires ``heat_transfer_coefficient`` and
      ``ambient_temperature``; no ``surface_temperature``.
    - ``fixed_temperature``: requires ``surface_temperature``; no
      ``h`` or ``ambient_temperature``.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    kind: Literal["adiabatic", "convection", "fixed_temperature"]
    selector: BoundarySelector
    heat_transfer_coefficient: HeatTransferCoefficient | None = None
    ambient_temperature: Temperature | None = None
    surface_temperature: Temperature | None = None
    areal_resistance: ArealThermalResistance = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def kind_specific_fields(self):
        if self.kind == "adiabatic":
            forbidden = {
                "heat_transfer_coefficient": self.heat_transfer_coefficient,
                "ambient_temperature": self.ambient_temperature,
                "surface_temperature": self.surface_temperature,
            }
            offenders = [k for k, v in forbidden.items() if v is not None]
            if offenders:
                raise ValueError(
                    f"adiabatic rule {self.name!r} must not set "
                    f"{offenders}")
        elif self.kind == "convection":
            if self.heat_transfer_coefficient is None:
                raise ValueError(
                    f"convection rule {self.name!r} requires "
                    "heat_transfer_coefficient")
            if self.ambient_temperature is None:
                raise ValueError(
                    f"convection rule {self.name!r} requires "
                    "ambient_temperature")
            if self.surface_temperature is not None:
                raise ValueError(
                    f"convection rule {self.name!r} must not set "
                    "surface_temperature")
        elif self.kind == "fixed_temperature":
            if self.surface_temperature is None:
                raise ValueError(
                    f"fixed_temperature rule {self.name!r} requires "
                    "surface_temperature")
            if self.heat_transfer_coefficient is not None:
                raise ValueError(
                    f"fixed_temperature rule {self.name!r} must not set "
                    "heat_transfer_coefficient")
            if self.ambient_temperature is not None:
                raise ValueError(
                    f"fixed_temperature rule {self.name!r} must not set "
                    "ambient_temperature")
        return self


class ThermalBoundaryConditionsConfig(BaseModel):
    """The boundary conditions block.

    ``default`` must be ``adiabatic`` in this stage (the
    anchored-component check refuses to solve a network that has no
    non-adiabatic boundary link in any connected component).
    """

    model_config = ConfigDict(extra="forbid")
    default: Literal["adiabatic"] = "adiabatic"
    rules: list[BoundaryConditionConfig] = Field(default_factory=list)


class PowerSelector(BaseModel):
    """Selector for matching a ``ThermalCell`` to a power source.

    The ``tags`` field matches when every key / value in the rule is
    present in the cell's tags dict. ``layer`` is matched against the
    parent box's name (which is the unique expanded layer name for
    cells emitted from stack templates).
    """

    model_config = ConfigDict(extra="forbid")
    component: str | None = None
    material: str | None = None
    layer: str | None = None
    tags: dict[str, Any] = Field(default_factory=dict)


class PowerSourceConfig(BaseModel):
    """A single power source.

    Only ``uniform_volume`` is supported in this stage. The total
    power is split across the selected cells in proportion to their
    volumes; multiple sources covering the same cell are additive.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    total_power: Power
    selector: PowerSelector
    distribution: Literal["uniform_volume"] = "uniform_volume"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThermalPowerSourcesConfig(BaseModel):
    """Container for the list of power sources."""

    model_config = ConfigDict(extra="forbid")
    sources: list[PowerSourceConfig] = Field(default_factory=list)


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    package_footprint: str
    materials: dict[str, Material]
    footprints: dict[str, Footprint]
    stack_templates: dict[str, StackTemplate]
    horizontal: HorizontalStructureConfig | None = None
    orthogonal_hbm: OrthogonalHBMStructureConfig | None = None
    discretization: DiscretizationConfig | None = None
    thermal_conductance: ThermalConductanceConfig | None = None
    thermal_boundary_conditions: ThermalBoundaryConditionsConfig | None = None
    thermal_power_sources: ThermalPowerSourcesConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def references_and_bounds(self):
        if (self.horizontal is None) == (self.orthogonal_hbm is None):
            raise ValueError(
                "config must specify exactly one of horizontal or orthogonal_hbm")
        for key, material in self.materials.items():
            if material.name != key:
                raise ValueError(f"material key {key!r} must match its name")
        for key, footprint in self.footprints.items():
            if footprint.name != key:
                raise ValueError(f"footprint key {key!r} must match its name")
        if self.package_footprint not in self.footprints:
            raise ValueError("package_footprint does not exist")
        package = self.footprints[self.package_footprint]
        tolerance = 1e-15
        for name, footprint in self.footprints.items():
            if (footprint.x0 < package.x0 - tolerance or footprint.x1 > package.x1 + tolerance
                    or footprint.y0 < package.y0 - tolerance or footprint.y1 > package.y1 + tolerance):
                raise ValueError(f"footprint {name!r} exceeds package bounds")
        if self.horizontal is not None:
            used_stacks = [self.horizontal.foundation.stack, self.horizontal.gpu.stack,
                           self.horizontal.top.stack,
                           self.horizontal.memory_zone.reference_stack]
            used_footprints = [self.horizontal.foundation.footprint,
                               self.horizontal.gpu.footprint,
                               self.horizontal.top.footprint,
                               self.horizontal.memory_zone.footprint]
            for column in self.horizontal.memory_zone.columns:
                used_footprints.append(column.footprint)
                if column.stack:
                    used_stacks.append(column.stack)
                if column.match_height_of:
                    used_stacks.append(column.match_height_of)
        else:
            orthogonal = self.orthogonal_hbm
            used_stacks = [orthogonal.foundation.stack, orthogonal.gpu.stack,
                           orthogonal.top.stack]
            used_footprints = [orthogonal.foundation.footprint,
                               orthogonal.gpu.footprint,
                               orthogonal.top.footprint,
                               orthogonal.cube_footprint]
        missing_stacks = sorted(set(used_stacks) - self.stack_templates.keys())
        missing_footprints = sorted(set(used_footprints) - self.footprints.keys())
        if missing_stacks:
            raise ValueError(f"unknown stack reference(s): {missing_stacks}")
        if missing_footprints:
            raise ValueError(f"unknown footprint reference(s): {missing_footprints}")
        referenced_materials = {
            layer.material for stack in self.stack_templates.values() for layer in stack.expand()
        }
        if self.horizontal is not None:
            referenced_materials.add(self.horizontal.memory_zone.background_material)
            for column in self.horizontal.memory_zone.columns:
                if column.material:
                    referenced_materials.add(column.material)
                if column.fill_above:
                    referenced_materials.add(column.fill_above)
        else:
            orthogonal = self.orthogonal_hbm
            referenced_materials.add(orthogonal.background_material)
            referenced_materials.add(orthogonal.adhesive.material)
            referenced_materials.update(
                layer.material for layer in orthogonal.memory_die.layers)
        missing_materials = sorted(referenced_materials - self.materials.keys())
        if missing_materials:
            raise ValueError(f"unknown material reference(s): {missing_materials}")
        if self.horizontal is not None:
            reference_height = self.stack_templates[
                self.horizontal.memory_zone.reference_stack].total_thickness
            for column in self.horizontal.memory_zone.columns:
                if column.stack:
                    height = self.stack_templates[column.stack].total_thickness
                else:
                    height = self.stack_templates[column.match_height_of].total_thickness
                if height > reference_height + 1e-15:
                    raise ValueError(f"column {column.name!r} match height exceeds memory zone")
                if height < reference_height - 1e-15 and not column.fill_above:
                    raise ValueError(f"short column {column.name!r} requires fill_above")
        else:
            orthogonal = self.orthogonal_hbm
            cube = self.footprints[orthogonal.cube_footprint]
            die = orthogonal.memory_die
            tolerance = 1e-12
            if die.width > cube.size_y + tolerance:
                raise ValueError("orthogonal die width exceeds cube width")
            if die.height > orthogonal.cube_height + tolerance:
                raise ValueError("orthogonal die height exceeds cube height")
            if die.count * die.thickness > cube.size_x + tolerance:
                raise ValueError("orthogonal die array exceeds cube arrangement length")
        return self


def is_orthogonal_m3d_template(data: Any) -> bool:
    """Return whether raw YAML declares the M3D-eDRAM research template."""
    return (
        isinstance(data, dict)
        and isinstance(data.get("architecture"), dict)
        and data["architecture"].get("type") == "orthogonal_m3d_edram")


def load_orthogonal_m3d_template(
        path: str | Path) -> OrthogonalM3DTemplateConfig:
    """Parse an M3D research template without claiming solver readiness."""
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not is_orthogonal_m3d_template(data):
        raise ValueError(
            "config does not declare architecture.type="
            "'orthogonal_m3d_edram'")
    return OrthogonalM3DTemplateConfig.model_validate(data)


def load_config(path: str | Path) -> SimulationConfig:
    """Load and validate a benchmark YAML.

    The file may be either the *legacy* form (top-level keys
    ``materials``/``footprints``/``stack_templates``/``horizontal``/...)
    or the *compact* form (``materials`` as a flat mapping of name to
    scalar-or-3-list, top-level ``geometry``/``stacks``/``mesh``/
    ``boundary``/``power``/``solver``). The compact form is expanded
    into the legacy :class:`SimulationConfig` dict by
    :func:`compile_user_config`; the legacy form passes through
    unchanged. Both produce the same validated
    :class:`SimulationConfig` instance.
    """
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if is_orthogonal_m3d_template(data):
        template = OrthogonalM3DTemplateConfig.model_validate(data)
        unresolved = template.unresolved_physical_parameters()
        if unresolved:
            raise UnresolvedPhysicalParametersError(
                template.architecture.type, unresolved)
        raise NotImplementedError(
            "orthogonal_m3d_edram template parameters are resolved, but "
            "v0 intentionally has no thermal geometry compilation path")
    expanded = compile_user_config(data)
    return SimulationConfig.model_validate(expanded)


# ---------------------------------------------------------------------------
# Compact-format compiler
# ---------------------------------------------------------------------------

_COMPACT_MARKERS = ("geometry", "orthogonal_hbm", "stacks", "mesh", "solver")


def is_compact_user_config(data: dict) -> bool:
    """Return ``True`` iff ``data`` looks like a compact user config
    (i.e. any of :data:`_COMPACT_MARKERS` is a top-level key)."""
    if not isinstance(data, dict):
        return False
    return any(marker in data for marker in _COMPACT_MARKERS)


def _snake_case(name: str) -> str:
    """``"GPU_HBM_uBump"`` -> ``"gpu_hbm_ubump"``; idempotent for an
    already-snake-case name. The compact user config declares
    materials in PascalCase / SCREAMING_SNAKE_CASE, but the
    expanded layer / box names follow the same convention as the
    legacy YAML (lower-snake-case)."""
    return name.lower()


def _k_local(value) -> list[float]:
    """Coerce a compact material k entry to the 3-list form the
    legacy ``Material`` model expects. A scalar is replicated; a
    3-list is accepted as-is; anything else is a config error."""
    if isinstance(value, bool):
        raise ValueError("material k must be a number or 3-list, not bool")
    if isinstance(value, (int, float)):
        return [float(value), float(value), float(value)]
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(
                f"material k must be a scalar or 3-list, got {value!r}")
        return [float(v) for v in value]
    raise ValueError(
        f"material k must be a number or 3-list, got {type(value).__name__}")


def _format_temperature(value) -> str:
    """Coerce a compact temperature value to a Pint string that the
    legacy :class:`Temperature` Annotated validator can parse.

    Pint's plain unit registry only ships ``K`` as a temperature
    unit; it does not recognise ``degC`` / ``celsius`` as a parser
    input. We therefore pre-convert a small set of common
    human-friendly forms to ``"<value> K"`` here so the user can
    write ``"20 degC"`` / ``"20 celsius"`` in the compact YAML.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value)} K"
    if isinstance(value, str):
        s = value.strip()
        for unit in ("degC", "deg_C", "celsius", "°C", "Celsius"):
            if s.endswith(unit):
                head = s[: -len(unit)].strip()
                try:
                    celsius_value = float(head)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid temperature {value!r}") from exc
                return f"{celsius_value + 273.15} K"
        return s
    raise ValueError(
        f"temperature must be a number or unit string, got {value!r}")


def _format_size_pair(pair):
    """Pass through a compact ``size: [w, h]`` as a list.

    The legacy ``Footprint`` model uses ``Length`` (a Pydantic
    ``Annotated`` type wrapping ``parse_length``) so a Pint string
    like ``"65 mm"`` is converted to SI metres by the model
    validator; we must NOT pre-parse to float here.
    """
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError(
            f"size must be a [width, height] pair, got {pair!r}")
    return [pair[0], pair[1]]


def _format_center(pair):
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError(
            f"center must be a [x, y] pair, got {pair!r}")
    return [pair[0], pair[1]]


def _build_layer_entry(material: str, thickness, *, name: str | None = None,
                       tags: dict | None = None,
                       lateral_inset: dict | None = None) -> dict:
    """One legacy ``{kind: layer, ...}`` entry from a compact
    ``[material, thickness]`` pair.
    """
    entry: dict = {
        "kind": "layer",
        "name": name if name is not None else _snake_case(material),
        "material": material,
        "thickness": thickness,
    }
    if tags:
        entry["tags"] = dict(tags)
    if lateral_inset is not None:
        entry["lateral_inset"] = dict(lateral_inset)
    return entry


def _build_legacy_materials(compact: dict) -> dict:
    """Expand ``materials: {Name: 400 or [kx, ky, kz]}`` into the
    legacy ``{Name: {name, k_local, metadata: {}}}`` form."""
    out: dict = {}
    for name, k in compact.items():
        out[name] = {
            "name": name,
            "k_local": _k_local(k),
            "metadata": {},
        }
    return out


def _build_legacy_footprints(geometry: dict) -> dict:
    """Expand the ``geometry`` block into the legacy ``footprints``
    mapping. The five simple footprints are placed at the origin;
    the parametric ``hbm`` block emits one footprint per centre in
    ``geometry.hbm.centers``.
    """
    out: dict = {}
    for name in ("package", "gpu", "memory_zone", "thermal_silicon"):
        block = geometry.get(name)
        if block is None:
            continue
        size_x, size_y = _format_size_pair(block["size"])
        out[name] = {
            "name": name,
            "center_x": 0.0,
            "center_y": 0.0,
            "size_x": size_x,
            "size_y": size_y,
        }
    hbm = geometry.get("hbm")
    if hbm is not None:
        size_x, size_y = _format_size_pair(hbm["size"])
        for col_name, center in hbm.get("centers", {}).items():
            cx, cy = _format_center(center)
            out[col_name] = {
                "name": col_name,
                "center_x": cx,
                "center_y": cy,
                "size_x": size_x,
                "size_y": size_y,
            }
    orthogonal = geometry.get("orthogonal_hbm")
    if orthogonal is not None:
        cube_size = orthogonal["cube_size"]
        if not isinstance(cube_size, (list, tuple)) or len(cube_size) != 3:
            raise ValueError(
                f"orthogonal_hbm.cube_size must contain three lengths, got {cube_size!r}")
        out["mosaic_cube"] = {
            "name": "mosaic_cube",
            "center_x": 0.0,
            "center_y": 0.0,
            # Paper ordering is [die width, array length, cube height].
            # Fig. 2 aligns the 30 x 22 mm top view with the canonical
            # GPU's global x=30 mm, y=22 mm footprint.
            "size_x": cube_size[1],
            "size_y": cube_size[0],
        }
    return out


def _build_legacy_stack_templates(stacks: dict,
                                geometry: dict | None = None) -> dict:
    """Expand the ``stacks`` block into the legacy
    ``stack_templates`` mapping.

    The flat ``{name: {layers: [...]}}`` form is the most common
    case. The HBM-style ``{name: {base, dram, top}}`` form
    is recognised and compiled with the legacy HBM-12hi layer
    naming convention (``hbm_base_si`` / ``dram_si_NN`` /
    ``top_dram_si`` etc.) plus the role tags the legacy fixtures
    expect (``dram_si`` / ``dram_beol`` / ``hybrid_bonding`` /
    ``hbm_base`` / ``gpu_hbm_interface``). The resulting stack is
    emitted under the key ``hbm_12hi`` so the column placement
    code and the legacy tests can find it.

    ``geometry`` is needed to source the DRAM lateral inset
    which is shared by every DRAM-layer entry (the 11 repeats
    and the top die). The compact user form expresses the DRAM
    geometry in terms of the **DRAM die size**
    (``geometry.hbm.dram_size``) and the **HBM column size**
    (``geometry.hbm.size``); the per-side lateral inset is
    computed as ``(base - dram) / 2`` and applied to every
    DRAM-layer entry. The older ``geometry.hbm.inset`` form is
    still accepted as a backwards-compat shortcut: when given,
    it is used directly as the per-side inset on both axes.
    """
    dram_inset = None
    if geometry is not None:
        hbm_block = geometry.get("hbm", {})
        if "dram_size" in hbm_block and "size" in hbm_block:
            # DRAM die size -> per-side lateral inset = (base - dram) / 2.
            from .units import parse_length
            dram_x, dram_y = (
                float(parse_length(hbm_block["dram_size"][0])),
                float(parse_length(hbm_block["dram_size"][1])),
            )
            base_x, base_y = (
                float(parse_length(hbm_block["size"][0])),
                float(parse_length(hbm_block["size"][1])),
            )
            inset_x = 0.5 * (base_x - dram_x)
            inset_y = 0.5 * (base_y - dram_y)
            if inset_x < 0 or inset_y < 0:
                raise ValueError(
                    f"geometry.hbm.dram_size {hbm_block['dram_size']!r} "
                    f"is larger than the HBM column size "
                    f"{hbm_block['size']!r}; the per-side lateral "
                    f"inset would be negative")
            dram_inset = {"x": inset_x, "y": inset_y}
        elif "inset" in hbm_block:
            # Backwards-compat: explicit per-side inset.
            inset = hbm_block["inset"]
            dram_inset = {"x": inset, "y": inset}
    out: dict = {}
    for name, body in stacks.items():
        if "layers" in body:
            items = _stack_items_from_layers(
                body["layers"], tags_factory=None,
                lateral_inset=None, name_suffix="")
            out[name] = {"items": items}
        elif "base" in body or "dram" in body or "top" in body:
            # HBM-style sub-template: base + repeat(dram) + top.
            # The legacy HBM fixture expects specific layer names and
            # role tags (``hbm_base_si`` / ``dram_si_NN`` etc.) which
            # the generic layer compiler cannot produce from a flat
            # ``[material, thickness]`` list. Apply the HBM-specific
            # naming here and emit under the legacy ``hbm_12hi`` key.
            out["hbm_12hi"] = _build_legacy_hbm_12hi_template(
                body, dram_inset=dram_inset)
        else:
            raise ValueError(
                f"stack '{name}' must have 'layers' or one of "
                f"'base'/'dram'/'top'; got keys {list(body.keys())}")
    # Rename compact user-facing stack keys to the legacy key names the
    # rest of the codebase (and the test fixtures) expect. The compact
    # YAML keeps the names short; the legacy form embeds the
    # physical-semantic suffix.
    renamed: dict = {}
    for key, body in out.items():
        legacy_key = _LEGACY_STACK_KEY_ALIASES.get(key, key)
        renamed[legacy_key] = body
    return renamed


# Compact user-facing stack key -> legacy stack key. The HBM sub-template
# is already mapped to ``hbm_12hi`` by the dedicated builder above; the
# only flat-form rename is ``thermal_silicon`` -> ``thermal_silicon_stack``
# so the scene summary keeps the canonical height names.
_LEGACY_STACK_KEY_ALIASES = {
    "thermal_silicon": "thermal_silicon_stack",
}


# Legacy HBM-12hi layer name + role mapping. The order matters: the
# base layers are read positionally (1st = uBump, 2nd = base BEOL,
# 3rd = base Si), and the DRAM repeat / top layers are matched by
# material name. The compact YAML keeps the same material names as
# the legacy fixture, so a small lookup table is enough.
_HBM_BASE_LEGACY = (
    # (compact material, legacy layer name, legacy role)
    ("GPU_HBM_uBump", "gpu_hbm_ubump", "gpu_hbm_interface"),
    ("HBM_Base_BEOL", "hbm_base_beol", "hbm_base"),
    ("Silicon",       "hbm_base_si",   "hbm_base"),
)

_HBM_DRAM_LEGACY = {
    "Hybrid_Bonding": ("hybrid_bonding", "hybrid_bonding"),
    "DRAM_BEOL":      ("dram_beol",      "dram_beol"),
    "Silicon":        ("dram_si",        "dram_si"),
}

_HBM_TOP_LEGACY = {
    "Hybrid_Bonding": ("top_hybrid_bonding", "hybrid_bonding"),
    "DRAM_BEOL":      ("top_dram_beol",      "dram_beol"),
    "Silicon":        ("top_dram_si",        "dram_si"),
}


def _build_legacy_hbm_12hi_template(hbm_block: dict,
                                    *, dram_inset: dict | None) -> dict:
    """Compile the compact HBM sub-template into the legacy HBM-12hi
    stack template with the layer names and role tags the legacy
    fixtures expect."""
    items: list = []
    base_layers = hbm_block.get("base", {}).get("layers", [])
    for idx, entry in enumerate(base_layers):
        if idx >= len(_HBM_BASE_LEGACY):
            break
        expected_material, layer_name, role = _HBM_BASE_LEGACY[idx]
        material, thickness = _unpack_layer_entry(entry)
        if material != expected_material:
            raise ValueError(
                f"HBM base layer {idx + 1} must be {expected_material!r} "
                f"(got {material!r}); the legacy naming convention "
                f"requires the uBump, base BEOL and base Si in that order")
        items.append(_build_layer_entry(
            material, thickness, name=layer_name,
            tags={"role": role}, lateral_inset=None))
    dram = hbm_block.get("dram", {})
    dram_layers = dram.get("layers", [])
    repeat_count = int(dram.get("repeat", 1))
    if dram_layers:
        repeat_layers: list = []
        for entry in dram_layers:
            material, thickness = _unpack_layer_entry(entry)
            if material in _HBM_DRAM_LEGACY:
                layer_name, role = _HBM_DRAM_LEGACY[material]
            else:
                layer_name = _snake_case(material)
                role = None
            tags = {"role": role} if role else {}
            repeat_layers.append(_build_layer_entry(
                material, thickness, name=layer_name,
                tags=tags, lateral_inset=dram_inset))
        items.append({
            "kind": "repeat",
            "count": repeat_count,
            "layers": repeat_layers,
        })
    top_layers = hbm_block.get("top", {}).get("layers", [])
    for entry in top_layers:
        material, thickness = _unpack_layer_entry(entry)
        if material in _HBM_TOP_LEGACY:
            layer_name, role = _HBM_TOP_LEGACY[material]
        else:
            layer_name = "top_" + _snake_case(material)
            role = None
        tags = {"role": role} if role else {}
        if material == "Silicon":
            tags["top_die"] = True
        items.append(_build_layer_entry(
            material, thickness, name=layer_name,
            tags=tags, lateral_inset=dram_inset))
    return {"items": items}


def _unpack_layer_entry(entry) -> tuple[str, object]:
    """Validate a compact ``[material, thickness]`` entry."""
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        raise ValueError(
            f"layer entry must be [material, thickness], got {entry!r}")
    material, thickness = entry
    return str(material), thickness


def _stack_items_from_layers(layers, *, tags_factory, lateral_inset,
                              name_suffix: str) -> list:
    """Turn a compact ``[[material, thickness], ...]`` list into
    legacy layer / repeat entries."""
    out: list = []
    for entry in layers:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"layer entry must be [material, thickness], got {entry!r}")
        material, thickness = entry
        if isinstance(material, list) and material and isinstance(material[0], str):
            # Nested repeat: a list whose first item is a list of
            # layer entries.
            inner = _stack_items_from_layers(
                material, tags_factory=tags_factory,
                lateral_inset=lateral_inset, name_suffix=name_suffix)
            raise ValueError(
                "nested repeat is not supported; use the "
                "'hbm-style' sub-template (base/dram/top) instead")
        tags = tags_factory(material) if tags_factory else None
        out.append(_build_layer_entry(
            str(material), thickness,
            name=None,  # default to snake_case of material
            tags=tags, lateral_inset=lateral_inset,
        ))
    if name_suffix:
        # Apply a name suffix to all entries (e.g. "_top") so they
        # don't clash with the regular layer names earlier in the
        # stack.
        for item in out:
            base = _snake_case(item["material"])
            item["name"] = base + name_suffix
    return out


def _dram_tags(material: str) -> dict | None:
    """Tag a DRAM_BEOL layer with ``role: dram_beol`` so the
    power-source selector can find it without re-declaring the
    role in user YAML.
    """
    if material == "DRAM_BEOL":
        return {"role": "dram_beol"}
    return None


def _top_dram_tags(material: str) -> dict | None:
    """Same as :func:`_dram_tags` but the top-die tag (kept
    consistent with the legacy HBM tag set so the power selector
    still finds the top die's DRAM_BEOL).
    """
    if material == "DRAM_BEOL":
        return {"role": "dram_beol"}
    return None


def _build_legacy_horizontal(geometry: dict, stacks: dict) -> dict:
    """Expand the ``geometry.hbm.centers`` block into the legacy
    ``horizontal.memory_zone.columns`` list.

    Compact toy configs without an HBM sub-template use their first flat
    stack as the reference height; production HBM configs retain the
    canonical ``hbm_12hi``/thermal-silicon reference semantics.
    """
    hbm_block = geometry.get("hbm", {})
    hbm_centers = hbm_block.get("centers", {})
    hbm_fill_above = hbm_block.get("fill_above")
    has_hbm_subtemplate = any(
        ("base" in body or "dram" in body or "top" in body)
        and "layers" not in body
        for body in stacks.values()
    )
    reference_stack = (
        "thermal_silicon_stack" if hbm_fill_above is not None
        else "hbm_12hi" if has_hbm_subtemplate
        else next(iter(stacks.keys()), "gpu")
    )
    columns = []
    for col_name in hbm_centers:
        column = {
            "name": col_name,
            "footprint": col_name,
            "stack": "hbm_12hi",
            "priority": 10,
        }
        if hbm_fill_above is not None:
            column["fill_above"] = hbm_fill_above
        columns.append(column)
    if "thermal_silicon" in geometry:
        columns.append({
            "name": "thermal_silicon",
            "footprint": "thermal_silicon",
            "stack": "thermal_silicon_stack",
            "priority": 10,
        })
    background_material = (
        "Mold" if has_hbm_subtemplate
        else next(iter(_extract_material_names(stacks)), "Mold")
    )
    return {
        "foundation": {"footprint": "package", "stack": "foundation"},
        "gpu": {"footprint": "gpu", "stack": "gpu"},
        "memory_zone": {
            "footprint": "memory_zone",
            "reference_stack": reference_stack,
            "background_material": background_material,
            "background_priority": 0,
            "columns": columns,
        },
        "top": {"footprint": "memory_zone", "stack": "top"},
    }


def _extract_material_names(stacks: dict) -> set[str]:
    """Collect materials referenced by compact stack definitions."""
    materials: set[str] = set()

    def walk(entry):
        if isinstance(entry, (list, tuple)):
            if len(entry) == 2 and isinstance(entry[0], str):
                materials.add(entry[0])
            else:
                for subentry in entry:
                    walk(subentry)
        elif isinstance(entry, dict):
            for subentry in entry.values():
                walk(subentry)

    for body in stacks.values():
        walk(body)
    return materials


def _build_legacy_orthogonal_hbm(geometry: dict) -> dict:
    """Compile the compact parametric MOSAIC block without expanding dies."""
    block = geometry["orthogonal_hbm"]
    cube_size = block["cube_size"]
    if not isinstance(cube_size, (list, tuple)) or len(cube_size) != 3:
        raise ValueError("orthogonal_hbm.cube_size must be [x, y, z]")
    die = block["memory_die"]
    stack = die.get("stack", [])
    roles = ("si_substrate", "active_beol", "daa")
    layers = []
    for index, entry in enumerate(stack):
        if isinstance(entry, dict) and "repeat" in entry:
            repeat = int(entry["repeat"])
            material = str(entry["material"])
            thickness = entry["thickness"]
            role = str(entry["role"])
            base_name = str(entry.get("name", _snake_case(material)))
            if repeat <= 0:
                raise ValueError("orthogonal layer repeat must be positive")
            for repeat_index in range(1, repeat + 1):
                layers.append({
                    "name": f"{base_name}_{repeat_index:02d}",
                    "material": material,
                    "thickness": thickness,
                    "role": role,
                })
            continue
        if isinstance(entry, dict) and {
                "material", "thickness", "role"}.issubset(entry):
            material = str(entry["material"])
            thickness = entry["thickness"]
            role = str(entry["role"])
            layer_name = str(entry.get("name", _snake_case(material)))
        elif isinstance(entry, dict) and len(entry) == 1:
            material, thickness = next(iter(entry.items()))
            role = roles[index] if index < len(roles) else _snake_case(
                str(material))
            layer_name = _snake_case(str(material))
        else:
            material, thickness = _unpack_layer_entry(entry)
            role = roles[index] if index < len(roles) else _snake_case(
                str(material))
            layer_name = _snake_case(str(material))
        layers.append({
            "name": layer_name,
            "material": str(material),
            "thickness": thickness,
            "role": role,
        })
    return {
        "cube_footprint": "mosaic_cube",
        "cube_height": cube_size[2],
        "background_material": block.get("background_material", "Mold"),
        "foundation": {"footprint": "package", "stack": "foundation"},
        "gpu": {"footprint": "gpu", "stack": "gpu"},
        "top": {"footprint": "mosaic_cube", "stack": "top"},
        "adhesive": {
            "material": block["adhesive"]["material"],
            "thickness": block["adhesive"]["thickness"],
        },
        "memory_die": {
            "count": die["count"],
            "width": die["width"],
            "height": die["height"],
            "layers": layers,
            "power_per_die": die.get("power_per_die", 0.0),
        },
    }


def _build_legacy_discretization(mesh: dict) -> dict:
    if "dz_max" in mesh and "z" not in mesh:
        mesh = {**mesh, "z": mesh["dz_max"]}
    return {
        "max_cell_size": {
            "x": mesh["dx"],
            "y": mesh["dy"],
            "z": mesh["z"],
        },
        "preserve_box_boundaries": True,
    }


def _build_legacy_thermal_conductance() -> dict:
    return {
        "rotation_policy": "axis_aligned_only",
        "default_interface_areal_resistance": "0 m^2*K/W",
        "interfaces": [],
    }


def _build_legacy_thermal_boundary_conditions(boundary: dict) -> dict:
    """Build the ``thermal_boundary_conditions`` block from the
    compact ``boundary`` section.

    Two convection rules (lid top, laminate bottom) are emitted; the
    default is adiabatic. The lid and laminate components / materials
    are matched against the legacy ``HorizontalColumnsBuilder`` tags
    so the same convection cells are picked as before.
    """
    ambient_raw = boundary.get("ambient", "293.15 K")
    ambient = _format_temperature(ambient_raw)
    rules: list = []
    if "lid_top_htc" in boundary:
        rules.append({
            "name": "lid_top_convection",
            "kind": "convection",
            "selector": {
                "component": "top",
                "material": "Lid",
                "axis": "z",
                "side": "plus",
                "priority": 100,
            },
            "heat_transfer_coefficient": boundary["lid_top_htc"],
            "ambient_temperature": ambient,
            "metadata": {"status": "PAPER_REPORTED"},
        })
    if "laminate_bottom_htc" in boundary:
        rules.append({
            "name": "laminate_bottom_convection",
            "kind": "convection",
            "selector": {
                "component": "foundation",
                "material": "Laminate",
                "axis": "z",
                "side": "minus",
                "priority": 100,
            },
            "heat_transfer_coefficient": boundary["laminate_bottom_htc"],
            "ambient_temperature": ambient,
            "metadata": {"status": "PAPER_REPORTED"},
        })
    return {
        "default": "adiabatic",
        "rules": rules,
    }


def _son23_component_sources(
    power: dict,
    hbm_columns: list[str],
    *,
    include_logic: bool = True,
    source_power_model: str = "son23split",
) -> list[dict]:
    """Expand the Son et al. EDAPS 2023 component partition.

    Functional components remain separate power sources even when two sources
    share the same existing active-side BEOL carrier. No lateral PHY/TSV/bank
    geometry is inferred here.
    """
    reference = power.get("son23_reference")
    if not isinstance(reference, dict):
        raise ValueError(
            "power.model='son23split' requires power.son23_reference")
    required = (
        "stack_total", "logic_phy", "logic_tsv",
        "dram_bank_per_die", "dram_tsv_per_die", "dram_die_count",
    )
    missing = [key for key in required if key not in reference]
    if missing:
        raise ValueError(
            f"power.son23_reference is missing required keys {missing}")

    reference_stack_W = parse_power(reference["stack_total"])
    logic_phy_reference_W = parse_power(reference["logic_phy"])
    logic_tsv_reference_W = parse_power(reference["logic_tsv"])
    dram_bank_reference_W = parse_power(reference["dram_bank_per_die"])
    dram_tsv_reference_W = parse_power(reference["dram_tsv_per_die"])
    dram_die_count = int(reference["dram_die_count"])
    if dram_die_count != 12:
        raise ValueError(
            "Son23 conventional 12Hi model requires dram_die_count=12")
    accounted_reference_W = (
        logic_phy_reference_W + logic_tsv_reference_W
        + dram_die_count * (dram_bank_reference_W + dram_tsv_reference_W))
    if not abs(accounted_reference_W - reference_stack_W) <= 1e-12:
        raise ValueError(
            "Son23 reference partition does not sum to stack_total: "
            f"{accounted_reference_W} W != {reference_stack_W} W")

    target_stack_W = parse_power(power["hbm_each"])
    reference_dram_W = dram_die_count * (
        dram_bank_reference_W + dram_tsv_reference_W)
    scaling_reference_W = (
        reference_stack_W if include_logic else reference_dram_W)
    scale = target_stack_W / scaling_reference_W
    component_power_W = {
        "logic_phy": logic_phy_reference_W * scale,
        "logic_tsv": logic_tsv_reference_W * scale,
        "dram_bank": dram_bank_reference_W * scale,
        "dram_tsv": dram_tsv_reference_W * scale,
    }
    scaled_total_W = (
        (component_power_W["logic_phy"] + component_power_W["logic_tsv"]
         if include_logic else 0.0)
        + dram_die_count * (
            component_power_W["dram_bank"] + component_power_W["dram_tsv"]))
    if not abs(scaled_total_W - target_stack_W) <= max(
            1e-12, 1e-12 * target_stack_W):
        raise ValueError(
            "scaled Son23 partition does not sum to hbm_each: "
            f"{scaled_total_W} W != {target_stack_W} W")

    common_metadata = {
        "power_model": source_power_model,
        "partition_provenance": "PAPER_REPORTED",
        "scaling_provenance": "DERIVED_FROM_REFERENCE",
        "placement_provenance": "MODELING_CHOICE",
        "placement": "existing active-side BEOL carrier",
        "reference_stack_power_W": reference_stack_W,
        "scaling_reference_power_W": scaling_reference_W,
        "scale_from_reference": scale,
    }
    sources: list[dict] = []
    for stack_name in hbm_columns:
        component = f"memory_column:{stack_name}"
        if include_logic:
            for function, power_W in (
                    ("phy", component_power_W["logic_phy"]),
                    ("tsv", component_power_W["logic_tsv"])):
                sources.append({
                    "name": f"hbm_{stack_name}_logic_{function}",
                    "total_power": power_W,
                    "selector": {
                        "component": component,
                        "material": "HBM_Base_BEOL",
                        "tags": {"role": "hbm_base"},
                    },
                    "distribution": "uniform_volume",
                    "metadata": {
                        **common_metadata,
                        "stack": stack_name,
                        "component_class": "logic",
                        "functional_component": function,
                    },
                })
        for die_index in range(1, dram_die_count + 1):
            layer_name = (
                f"dram_beol_{die_index:02d}"
                if die_index < dram_die_count else "top_dram_beol")
            for function, power_W in (
                    ("bank", component_power_W["dram_bank"]),
                    ("tsv", component_power_W["dram_tsv"])):
                sources.append({
                    "name": f"hbm_{stack_name}_dram{die_index:02d}_{function}",
                    "total_power": power_W,
                    "selector": {
                        "component": component,
                        "layer": f"{component}.{layer_name}",
                    },
                    "distribution": "uniform_volume",
                    "metadata": {
                        **common_metadata,
                        "stack": stack_name,
                        "component_class": "dram",
                        "functional_component": function,
                        "dram_die_index": die_index,
                    },
                })
    return sources


def _build_legacy_thermal_power_sources(power: dict,
                                          geometry: dict) -> dict:
    """Build the ``thermal_power_sources.sources`` list from the
    compact ``power`` section. GPU power goes to the FEOL layer;
    HBM power goes to the dram_beol layer of each HBM column.
    """
    power_model = str(power.get("model", "uniform"))
    if power_model not in {
            "uniform", "son23split", "son23_dram_only",
            "m3d_operation_energy"}:
        raise ValueError(
            f"unsupported power.model {power_model!r}; expected "
            "'uniform', 'son23split', 'son23_dram_only', or "
            "'m3d_operation_energy'")
    sources: list = []
    if "gpu" in power:
        sources.append({
            "name": "gpu_total",
            "total_power": power["gpu"],
            "selector": {"component": "gpu", "material": "FEOL"},
            "distribution": "uniform_volume",
            "metadata": {
                "status": "PAPER_REPORTED",
                "power_model": "uniform",
                "component_class": "gpu",
            },
        })
    hbm_columns = list(geometry.get("hbm", {}).get("centers", {}).keys())
    if "hbm_each" in power:
        if power_model in {"son23split", "son23_dram_only"}:
            sources.extend(_son23_component_sources(
                power, hbm_columns,
                include_logic=(power_model == "son23split"),
                source_power_model=power_model,
            ))
        else:
            for col in hbm_columns:
                sources.append({
                    "name": col,
                    "total_power": power["hbm_each"],
                    "selector": {
                        "component": f"memory_column:{col}",
                        "tags": {"role": "dram_beol"},
                    },
                    "distribution": "uniform_volume",
                    "metadata": {
                        "status": "PAPER_REPORTED",
                        "power_model": "uniform",
                        "component_class": "dram",
                        "stack": col,
                    },
                })
    orthogonal = geometry.get("orthogonal_hbm")
    if orthogonal is not None:
        die = orthogonal["memory_die"]
        if power_model == "m3d_operation_energy":
            from .thermal.m3d_power import calculate_array_read_power
            read_probability = power["read_state_probability"]
            memory_total_W = calculate_array_read_power(
                delivered_bandwidth_bit_per_s=float(
                    power["delivered_bandwidth_bit_per_s"]),
                read_fraction=float(power["read_fraction"]),
                state_0_probability=float(read_probability["p0"]),
                state_1_probability=float(read_probability["p1"]),
                read_0_energy_fJ_per_bit=float(
                    power["operation_energy_fJ_per_bit"]["read_0"]),
                read_1_energy_fJ_per_bit=float(
                    power["operation_energy_fJ_per_bit"]["read_1"]),
            )
            if float(power["write_fraction"]) != 0.0:
                raise ValueError(
                    "M3D-v1 array-core nominal requires write_fraction=0")
            per_die_power = memory_total_W / int(die["count"])
            target_role = "m3d_bitcell_stack"
        else:
            per_die_power = die["power_per_die"]
            target_role = "active_beol"
        for index in range(1, int(die["count"]) + 1):
            die_name = f"die_{index:03d}"
            sources.append({
                "name": f"hbm_{die_name}",
                "total_power": per_die_power,
                "selector": {
                    "component": f"orthogonal_hbm:{die_name}",
                    "tags": {"role": target_role},
                },
                "distribution": "uniform_volume",
                "metadata": {
                    "status": (
                        "DERIVED_FROM_OPERATION_ENERGY"
                        if power_model == "m3d_operation_energy"
                        else "PAPER_REPORTED"),
                    "power_model": power_model,
                    "component_class": "dram",
                    "stack": die_name,
                    "modeling_choice": (
                        "uniform within homogenized 8-layer M3D bit-cell stack"
                        if power_model == "m3d_operation_energy"
                        else "uniform within die BEOL"),
                },
            })
    return {"sources": sources}


def _build_legacy_metadata(solver: dict, supplied: dict | None = None) -> dict:
    """Build a minimal ``metadata`` block (no paper text; that lives
    in ``docs/benchmarks/``)."""
    md: dict = dict(supplied or {})
    if solver:
        md["solver"] = dict(solver)
    return md


def compile_user_config(data: dict) -> dict:
    """Expand the compact user-facing YAML into the legacy
    :class:`SimulationConfig` dict.

    The legacy form is detected by the absence of any of the
    compact markers (:data:`_COMPACT_MARKERS`); it is returned
    unchanged so older configs keep working as fixtures / for
    reference.
    """
    if not is_compact_user_config(data):
        return data
    out: dict = {"name": data.get("name", "untitled")}
    out["package_footprint"] = "package"
    if "materials" in data:
        out["materials"] = _build_legacy_materials(data["materials"])
    geometry = dict(data.get("geometry", {}))
    if "orthogonal_hbm" in data:
        if "orthogonal_hbm" in geometry:
            raise ValueError(
                "orthogonal_hbm must be declared either at top level or under geometry, not both")
        geometry["orthogonal_hbm"] = data["orthogonal_hbm"]
    if geometry:
        out["footprints"] = _build_legacy_footprints(geometry)
    stacks = data.get("stacks", {})
    if stacks:
        out["stack_templates"] = _build_legacy_stack_templates(
            stacks, geometry=data.get("geometry"))
    if "orthogonal_hbm" in geometry:
        out["orthogonal_hbm"] = _build_legacy_orthogonal_hbm(geometry)
    elif geometry or stacks:
        out["horizontal"] = _build_legacy_horizontal(geometry, stacks)
    if "mesh" in data:
        out["discretization"] = _build_legacy_discretization(data["mesh"])
    out["thermal_conductance"] = _build_legacy_thermal_conductance()
    if "boundary" in data:
        out["thermal_boundary_conditions"] = (
            _build_legacy_thermal_boundary_conditions(data["boundary"]))
    thermal_power_sources = None
    if "power" in data:
        thermal_power_sources = _build_legacy_thermal_power_sources(
            data["power"], geometry)
        out["thermal_power_sources"] = thermal_power_sources
    metadata = _build_legacy_metadata(
        data.get("solver", {}), data.get("metadata"))
    if (thermal_power_sources is not None
            and data.get("power", {}).get("model")
            == "m3d_operation_energy"):
        derived_memory_W = sum(
            float(source["total_power"])
            for source in thermal_power_sources["sources"]
            if source.get("metadata", {}).get("power_model")
            == "m3d_operation_energy")
        bookkeeping = metadata.setdefault("architecture_bookkeeping", {})
        bookkeeping["array_read_power_W"] = derived_memory_W
        bookkeeping["memory_power_derivation"] = (
            "delivered_bandwidth_bit_per_s * read_1_energy_fJ_per_bit * 1e-15")
    out["metadata"] = metadata
    return out
