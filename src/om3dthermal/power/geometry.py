"""Memory-footprint constraints sourced from existing thermal configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from om3dthermal.units import parse_length

from .config import GeometrySourceInput, resolve_project_path


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
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot resolve M3D layer geometry from {path}") from exc
    if layers <= 0 or pitch_um <= 0.0:
        raise ValueError("M3D layers and layer pitch must be positive")
    if memory_layers != layers:
        raise ValueError(
            "m3d_memory.layers must equal m3d_beol.bitcell_layers")
    return M3DGeometry(layers=layers, layer_pitch_um=pitch_um)


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
