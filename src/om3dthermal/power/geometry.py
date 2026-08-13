"""Memory-footprint constraints sourced from existing thermal configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from om3dthermal.units import parse_length

from .config import (
    CanonicalCaseConfig,
    GeometrySourceInput,
    resolve_project_path,
)


@dataclass(frozen=True)
class GeometryFit:
    configured_x_mm: float
    configured_y_mm: float
    required_x_mm: float
    required_y_mm: float
    x_utilization: float
    y_utilization: float
    geometry_feasible: bool

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class M3DGeometry:
    layers: int
    layer_pitch_um: float
    slab_x_um: float
    slab_y_um: float
    cell_area_um2: float


@dataclass(frozen=True)
class ResolvedGeometry:
    source: str
    memory_region: str
    configured_x_mm: float
    configured_y_mm: float
    m3d: M3DGeometry | None = None


def resolve_case_geometry(case: CanonicalCaseConfig) -> ResolvedGeometry:
    """Resolve canonical geometry without opening another YAML file."""
    geometry = case.geometry
    m3d = None
    if geometry.type == "orthogonal_m3d":
        assert geometry.m3d_stack is not None
        assert geometry.orthogonal is not None
        x_mm = geometry.orthogonal.slab_plane_y_mm
        y_mm = geometry.orthogonal.slab_height_z_mm
        m3d = M3DGeometry(
            layers=geometry.m3d_stack.bitcell_layers,
            layer_pitch_um=(
                geometry.m3d_stack.bitcell_layer_pitch_nm * 1e-3),
            slab_x_um=x_mm * 1e3,
            slab_y_um=y_mm * 1e3,
            cell_area_um2=geometry.m3d_stack.cell_area_um2,
        )
        region = "orthogonal_m3d_slab"
    else:
        assert geometry.memory_region is not None
        x_mm = geometry.memory_region.width_mm
        y_mm = geometry.memory_region.height_mm
        region = "hbm_dram_die"
    return ResolvedGeometry(
        source=f"canonical_case:{case.name}",
        memory_region=region,
        configured_x_mm=x_mm,
        configured_y_mm=y_mm,
        m3d=m3d,
    )


def resolve_legacy_geometry(
        project_root: Path, source: GeometrySourceInput) -> ResolvedGeometry:
    path, x_mm, y_mm = load_memory_region_size(project_root, source)
    m3d = (
        load_m3d_geometry(project_root, source)
        if source.memory_region == "orthogonal_m3d_slab" else None)
    return ResolvedGeometry(
        source=str(path),
        memory_region=source.memory_region,
        configured_x_mm=x_mm,
        configured_y_mm=y_mm,
        m3d=m3d,
    )


def _length_mm(value: Any) -> float:
    return parse_length(value) * 1e3


def load_memory_region_size(
        project_root: Path, source: GeometrySourceInput,
        ) -> tuple[Path, float, float]:
    """Read a die/slab plane from an existing thermal geometry config."""
    path = resolve_project_path(project_root, source.config).resolve()
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"thermal geometry config root must be a mapping: {path}")

    try:
        if source.memory_region == "hbm_dram_die":
            size = raw["geometry"]["hbm"]["dram_size"]
            x_mm, y_mm = _length_mm(size[0]), _length_mm(size[1])
        elif source.memory_region == "orthogonal_memory_slab":
            die = raw["orthogonal_hbm"]["memory_die"]
            x_mm, y_mm = _length_mm(die["width"]), _length_mm(die["height"])
        elif source.memory_region == "orthogonal_m3d_slab":
            orthogonal = raw["orthogonal"]
            x_mm = float(orthogonal["slab_plane_y_mm"])
            y_mm = float(orthogonal["slab_height_z_mm"])
        else:  # protected by the config Literal, retained for direct callers
            raise ValueError(f"unsupported memory region {source.memory_region!r}")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"cannot resolve {source.memory_region!r} from thermal geometry "
            f"config {path}") from exc
    if x_mm <= 0.0 or y_mm <= 0.0:
        raise ValueError(f"memory region dimensions must be positive in {path}")
    return path, x_mm, y_mm


def load_m3d_geometry(
        project_root: Path, source: GeometrySourceInput) -> M3DGeometry:
    """Read monolithic layer topology from the existing M3D geometry config."""
    if source.memory_region != "orthogonal_m3d_slab":
        raise ValueError("M3D topology requires orthogonal_m3d_slab geometry")
    path = resolve_project_path(project_root, source.config).resolve()
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    try:
        layers = int(raw["m3d_beol"]["bitcell_layers"])
        memory_layers = int(raw["m3d_memory"]["layers"])
        pitch_um = float(raw["m3d_beol"]["bitcell_layer_pitch_nm"]) * 1e-3
        slab_x_um = float(raw["orthogonal"]["slab_plane_y_mm"]) * 1e3
        slab_y_um = float(raw["orthogonal"]["slab_height_z_mm"]) * 1e3
        cell_area_um2 = float(raw["m3d_memory"]["cell_area_um2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot resolve M3D layer geometry from {path}") from exc
    if (layers <= 0 or pitch_um <= 0.0 or slab_x_um <= 0.0
            or slab_y_um <= 0.0 or cell_area_um2 <= 0.0):
        raise ValueError("M3D geometry values must be positive")
    if memory_layers != layers:
        raise ValueError(
            "m3d_memory.layers must equal m3d_beol.bitcell_layers")
    return M3DGeometry(
        layers=layers,
        layer_pitch_um=pitch_um,
        slab_x_um=slab_x_um,
        slab_y_um=slab_y_um,
        cell_area_um2=cell_area_um2,
    )


def evaluate_geometry_fit(
        *, configured_x_mm: float, configured_y_mm: float,
        required_x_mm: float, required_y_mm: float,
        ) -> GeometryFit:
    """Evaluate independent X/Y fit without changing DreamRAM organization."""
    values = (configured_x_mm, configured_y_mm, required_x_mm, required_y_mm)
    if any(value <= 0.0 for value in values):
        raise ValueError("configured and required geometry dimensions must be positive")
    return GeometryFit(
        configured_x_mm=configured_x_mm,
        configured_y_mm=configured_y_mm,
        required_x_mm=required_x_mm,
        required_y_mm=required_y_mm,
        x_utilization=required_x_mm / configured_x_mm,
        y_utilization=required_y_mm / configured_y_mm,
        geometry_feasible=(
            required_x_mm <= configured_x_mm
            and required_y_mm <= configured_y_mm),
    )
