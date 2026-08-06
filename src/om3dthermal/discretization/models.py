"""Pydantic data models for the block-structured thermal discretisation.

The ``DiscretizationConfig`` and ``CellSizeConfig`` data classes live in
``om3dthermal.config`` to avoid an import cycle (the discretisation
algorithms import from the configuration models). The cell / edge /
boundary models live here and depend only on primitives + pydantic.

The models are deliberately read-only at the geometry level: they hold
spatial coordinates, the parent box reference, and the material. They
do **not** store temperature, power, conductance or any other thermal
state. A future KCL / steady-state solver would attach those to the
``ThermalCell.id`` separately.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config import CellSizeConfig, DiscretizationConfig  # re-exported below
from ..geometry.primitives import IDENTITY_ROTATION

__all__ = [
    "AdjacencyEdge",
    "BoundaryFace",
    "CellSizeConfig",
    "DiscretizationConfig",
    "GeometryOverlapError",
    "ThermalCell",
]


class GeometryOverlapError(ValueError):
    """Raised when two ``AxisAlignedBox`` regions claim the same grid
    voxel during cell generation. Carries the parent box names and the
    voxel coordinate so the offender is easy to locate.
    """

    def __init__(self, *, box_a: str, box_b: str, ix: int, iy: int, iz: int,
                 x_range: tuple[float, float], y_range: tuple[float, float],
                 z_range: tuple[float, float]):
        self.box_a = box_a
        self.box_b = box_b
        self.ix, self.iy, self.iz = ix, iy, iz
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        super().__init__(
            f"geometry overlap: box {box_a!r} and box {box_b!r} both claim "
            f"voxel ({ix}, {iy}, {iz}) with x in {x_range}, y in {y_range}, "
            f"z in {z_range}")


class ThermalCell(BaseModel):
    """A single block-structured cell produced by the discretiser.

    Cells are unitless with respect to thermal state. They carry enough
    geometry and provenance to be paired with a future solver and to
    trace back to the originating ``AxisAlignedBox``.
    """

    model_config = ConfigDict(extra="forbid")
    id: int = Field(ge=0)
    ix: int = Field(ge=0)
    iy: int = Field(ge=0)
    iz: int = Field(ge=0)
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float
    material: str
    parent_box_id: str
    parent_box_name: str
    component: str | None = None
    source_path: str
    rotation: tuple[tuple[float, float, float], ...] = IDENTITY_ROTATION
    tags: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def positive_extents(self):
        if not (self.x1 > self.x0):
            raise ValueError(f"cell {self.id} has non-positive x extent "
                             f"({self.x0}, {self.x1})")
        if not (self.y1 > self.y0):
            raise ValueError(f"cell {self.id} has non-positive y extent "
                             f"({self.y0}, {self.y1})")
        if not (self.z1 > self.z0):
            raise ValueError(f"cell {self.id} has non-positive z extent "
                             f"({self.z0}, {self.z1})")
        return self

    @property
    def size_x(self) -> float:
        return self.x1 - self.x0

    @property
    def size_y(self) -> float:
        return self.y1 - self.y0

    @property
    def size_z(self) -> float:
        return self.z1 - self.z0

    @property
    def center_x(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def center_y(self) -> float:
        return 0.5 * (self.y0 + self.y1)

    @property
    def center_z(self) -> float:
        return 0.5 * (self.z0 + self.z1)

    @property
    def volume(self) -> float:
        return self.size_x * self.size_y * self.size_z


class AdjacencyEdge(BaseModel):
    """Face-shared adjacency between two ``ThermalCell`` nodes.

    The edge always points along one of the three world axes
    (``"x"`` / ``"y"`` / ``"z"``), the interface coordinate is the
    shared plane's position on that axis, and ``face_area`` is the area
    of that plane. ``center_distance`` is the distance between the
    centres of the two cells; the two half-distances are the half-extent
    of each cell along the normal axis.
    """

    model_config = ConfigDict(extra="forbid")
    id: int = Field(ge=0)
    cell_a: int = Field(ge=0)
    cell_b: int = Field(ge=0)
    axis: Literal["x", "y", "z"]
    interface_coordinate: float
    face_area: float
    center_distance: float
    half_distance_a: float
    half_distance_b: float
    material_a: str
    material_b: str
    is_material_interface: bool

    @field_validator("face_area", "center_distance", "half_distance_a", "half_distance_b")
    @classmethod
    def strictly_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("face_area / center_distance / half_distance_? must be > 0")
        return value


class BoundaryFace(BaseModel):
    """A face of a cell that is not shared with another cell.

    ``classification`` distinguishes the two cases the future solver
    will care about:
    - ``"scene_outer_boundary"``: the face lies on the global scene
      bounding box; HTC / fixed temperature / adiabatic flags would be
      applied here in a later stage.
    - ``"exposed_internal_boundary"``: the face lies strictly inside
      the scene bounding box but the neighbour voxel is empty. This
      is the geometric signature of a cavity (e.g. the mold-filled
      lateral gaps in the benchmark).
    """

    model_config = ConfigDict(extra="forbid")
    id: int = Field(ge=0)
    cell_id: int = Field(ge=0)
    axis: Literal["x", "y", "z"]
    side: Literal["minus", "plus"]
    coordinate: float
    area: float = Field(gt=0)
    normal: tuple[float, float, float]
    component: str | None = None
    material: str
    classification: Literal["scene_outer_boundary", "exposed_internal_boundary"]
