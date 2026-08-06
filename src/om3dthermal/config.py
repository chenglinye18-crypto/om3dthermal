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
    horizontal: HorizontalStructureConfig
    discretization: DiscretizationConfig | None = None
    thermal_conductance: ThermalConductanceConfig | None = None
    thermal_boundary_conditions: ThermalBoundaryConditionsConfig | None = None
    thermal_power_sources: ThermalPowerSourcesConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def references_and_bounds(self):
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
        used_stacks = [self.horizontal.foundation.stack, self.horizontal.gpu.stack,
                       self.horizontal.top.stack, self.horizontal.memory_zone.reference_stack]
        used_footprints = [self.horizontal.foundation.footprint, self.horizontal.gpu.footprint,
                           self.horizontal.top.footprint, self.horizontal.memory_zone.footprint]
        for column in self.horizontal.memory_zone.columns:
            used_footprints.append(column.footprint)
            if column.stack:
                used_stacks.append(column.stack)
            if column.match_height_of:
                used_stacks.append(column.match_height_of)
        missing_stacks = sorted(set(used_stacks) - self.stack_templates.keys())
        missing_footprints = sorted(set(used_footprints) - self.footprints.keys())
        if missing_stacks:
            raise ValueError(f"unknown stack reference(s): {missing_stacks}")
        if missing_footprints:
            raise ValueError(f"unknown footprint reference(s): {missing_footprints}")
        referenced_materials = {
            layer.material for stack in self.stack_templates.values() for layer in stack.expand()
        } | {self.horizontal.memory_zone.background_material}
        for column in self.horizontal.memory_zone.columns:
            if column.material:
                referenced_materials.add(column.material)
            if column.fill_above:
                referenced_materials.add(column.fill_above)
        missing_materials = sorted(referenced_materials - self.materials.keys())
        if missing_materials:
            raise ValueError(f"unknown material reference(s): {missing_materials}")
        reference_height = self.stack_templates[self.horizontal.memory_zone.reference_stack].total_thickness
        for column in self.horizontal.memory_zone.columns:
            if column.stack:
                height = self.stack_templates[column.stack].total_thickness
            else:
                height = self.stack_templates[column.match_height_of].total_thickness
            if height > reference_height + 1e-15:
                raise ValueError(f"column {column.name!r} match height exceeds memory zone")
            if height < reference_height - 1e-15 and not column.fill_above:
                raise ValueError(f"short column {column.name!r} requires fill_above")
        return self


def load_config(path: str | Path) -> SimulationConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return SimulationConfig.model_validate(data)
