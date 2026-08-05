"""Pydantic v2 configuration schema and YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from .geometry.primitives import Footprint
from .materials import Material
from .units import parse_length

Length = Annotated[float, BeforeValidator(parse_length)]


class Layer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["layer"] = "layer"
    name: str
    material: str
    thickness: Length
    tags: dict[str, Any] = Field(default_factory=dict)

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
                    tags=dict(item.tags), source_suffix=f"items[{item_index}]"))
            else:
                for repeat_index in range(1, item.count + 1):
                    for layer_index, layer in enumerate(item.layers):
                        expanded.append(ExpandedLayer(
                            name=f"{layer.name}_{repeat_index:02d}", material=layer.material,
                            thickness=layer.thickness,
                            tags={**layer.tags, "repeat_index": repeat_index},
                            source_suffix=(f"items[{item_index}].layers[{layer_index}]"
                                           f"#repeat={repeat_index}")))
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


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    package_footprint: str
    materials: dict[str, Material]
    footprints: dict[str, Footprint]
    stack_templates: dict[str, StackTemplate]
    horizontal: HorizontalStructureConfig
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
