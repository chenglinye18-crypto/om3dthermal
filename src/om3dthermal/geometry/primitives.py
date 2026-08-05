"""Geometry primitives shared by present and future builders."""

from __future__ import annotations

from typing import Any, Annotated
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, BeforeValidator, model_validator

from om3dthermal.units import parse_length

Length = Annotated[float, BeforeValidator(parse_length)]
IDENTITY_ROTATION = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class Footprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    center_x: Length
    center_y: Length
    size_x: Length
    size_y: Length

    @model_validator(mode="after")
    def positive_size(self):
        if self.size_x <= 0 or self.size_y <= 0:
            raise ValueError("footprint sizes must be strictly positive")
        return self

    @property
    def x0(self) -> float:
        return self.center_x - self.size_x / 2

    @property
    def x1(self) -> float:
        return self.center_x + self.size_x / 2

    @property
    def y0(self) -> float:
        return self.center_y - self.size_y / 2

    @property
    def y1(self) -> float:
        return self.center_y + self.size_y / 2


class AxisAlignedBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    material: str
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float
    tags: dict[str, Any] = Field(default_factory=dict)
    source_path: str
    rotation: tuple[tuple[float, float, float], ...] = IDENTITY_ROTATION

    @model_validator(mode="after")
    def positive_dimensions(self):
        if not (self.x1 > self.x0 and self.y1 > self.y0 and self.z1 > self.z0):
            raise ValueError(f"box {self.name!r} must have strictly positive dimensions")
        if len(self.rotation) != 3 or any(len(row) != 3 for row in self.rotation):
            raise ValueError("rotation must be a 3x3 matrix")
        return self

    @property
    def dimensions(self) -> tuple[float, float, float]:
        return self.x1 - self.x0, self.y1 - self.y0, self.z1 - self.z0
