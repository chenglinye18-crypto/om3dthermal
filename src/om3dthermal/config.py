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
    expanded = compile_user_config(data)
    return SimulationConfig.model_validate(expanded)


# ---------------------------------------------------------------------------
# Compact-format compiler
# ---------------------------------------------------------------------------

_COMPACT_MARKERS = ("geometry", "stacks", "mesh", "solver")


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
    (``geometry.hbm.inset``) which is shared by every DRAM-layer
    entry (the 11 repeats and the top die).
    """
    dram_inset = None
    if geometry is not None:
        hbm_block = geometry.get("hbm", {})
        if "inset" in hbm_block:
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
    """
    hbm_block = geometry.get("hbm", {})
    hbm_centers = hbm_block.get("centers", {})
    columns = []
    for col_name in hbm_centers:
        columns.append({
            "name": col_name,
            "footprint": col_name,
            "stack": "hbm_12hi",
            "priority": 10,
        })
    if "thermal_silicon" in geometry:
        columns.append({
            "name": "thermal_silicon",
            "footprint": "thermal_silicon",
            "stack": "thermal_silicon_stack",
            "priority": 10,
        })
    return {
        "foundation": {"footprint": "package", "stack": "foundation"},
        "gpu": {"footprint": "gpu", "stack": "gpu"},
        "memory_zone": {
            "footprint": "memory_zone",
            "reference_stack": "hbm_12hi",
            "background_material": "Mold",
            "background_priority": 0,
            "columns": columns,
        },
        "top": {"footprint": "memory_zone", "stack": "top"},
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


def _build_legacy_thermal_power_sources(power: dict,
                                          geometry: dict) -> dict:
    """Build the ``thermal_power_sources.sources`` list from the
    compact ``power`` section. GPU power goes to the FEOL layer;
    HBM power goes to the dram_beol layer of each HBM column.
    """
    sources: list = []
    if "gpu" in power:
        sources.append({
            "name": "gpu_total",
            "total_power": power["gpu"],
            "selector": {"component": "gpu", "material": "FEOL"},
            "distribution": "uniform_volume",
            "metadata": {"status": "PAPER_REPORTED"},
        })
    hbm_columns = list(geometry.get("hbm", {}).get("centers", {}).keys())
    if "hbm_each" in power:
        for col in hbm_columns:
            sources.append({
                "name": col,
                "total_power": power["hbm_each"],
                "selector": {
                    "component": f"memory_column:{col}",
                    "tags": {"role": "dram_beol"},
                },
                "distribution": "uniform_volume",
                "metadata": {"status": "PAPER_REPORTED"},
            })
    return {"sources": sources}


def _build_legacy_metadata(solver: dict) -> dict:
    """Build a minimal ``metadata`` block (no paper text; that lives
    in ``docs/benchmarks/``)."""
    md: dict = {}
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
    geometry = data.get("geometry", {})
    if geometry:
        out["footprints"] = _build_legacy_footprints(geometry)
    stacks = data.get("stacks", {})
    if stacks:
        out["stack_templates"] = _build_legacy_stack_templates(
            stacks, geometry=data.get("geometry"))
    if geometry or stacks:
        out["horizontal"] = _build_legacy_horizontal(geometry, stacks)
    if "mesh" in data:
        out["discretization"] = _build_legacy_discretization(data["mesh"])
    out["thermal_conductance"] = _build_legacy_thermal_conductance()
    if "boundary" in data:
        out["thermal_boundary_conditions"] = (
            _build_legacy_thermal_boundary_conditions(data["boundary"]))
    if "power" in data:
        out["thermal_power_sources"] = (
            _build_legacy_thermal_power_sources(data["power"], geometry))
    out["metadata"] = _build_legacy_metadata(data.get("solver", {}))
    return out
