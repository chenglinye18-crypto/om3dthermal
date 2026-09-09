"""Memory-footprint constraints sourced from existing thermal configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass


from .config import (
    CanonicalCaseConfig,
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
    memory_region_count: int = 1
    memory_dies_per_region: int = 1
    m3d: M3DGeometry | None = None


def resolve_case_geometry(case: CanonicalCaseConfig) -> ResolvedGeometry:
    """Resolve canonical geometry without opening another YAML file."""
    geometry = case.geometry
    m3d = None
    if geometry.type in {"orthogonal_si", "orthogonal_m3d"}:
        assert geometry.orthogonal is not None
        x_mm = geometry.orthogonal.slab_plane_y_mm
        y_mm = geometry.orthogonal.slab_height_z_mm
        region_count = geometry.orthogonal.slab_count
        if geometry.type == "orthogonal_m3d":
            assert geometry.m3d_stack is not None
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
            region = "orthogonal_memory_slab"
    else:
        assert geometry.memory_region is not None
        capacity_region = (
            geometry.capacity_instance_region or geometry.memory_region)
        x_mm = capacity_region.width_mm
        y_mm = capacity_region.height_mm
        region = "hbm_dram_die"
        region_count = int(
            geometry.layout.get("total_physical_stack_equivalents", 1))
        dies_per_region = int(geometry.layout.get("dram_dies_per_stack", 1))
        if dies_per_region <= 0:
            raise ValueError("geometry.layout.dram_dies_per_stack must be positive")
    if geometry.type in {"orthogonal_si", "orthogonal_m3d"}:
        dies_per_region = 1
    return ResolvedGeometry(
        source=f"canonical_case:{case.name}",
        memory_region=region,
        configured_x_mm=x_mm,
        configured_y_mm=y_mm,
        memory_region_count=region_count,
        memory_dies_per_region=dies_per_region,
        m3d=m3d,
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
