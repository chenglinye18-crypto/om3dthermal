"""Physical slab identities; the obsolete die-only NMP thermal path is removed."""
from dataclasses import dataclass
from ..power.config import CanonicalCaseConfig

@dataclass(frozen=True)
class NMPDieThermalRegion:
    die_id: int
    region_id: str
    geometry_die_index: int
    center_x_m: float
    memory_role: str = "m3d_bitcell_beol_stack"
    nmp_role: str = "feol"

def physical_nmp_die_regions(case: CanonicalCaseConfig) -> tuple[NMPDieThermalRegion, ...]:
    """Return the explicit zero-based architecture ID to geometry identity."""
    orth = case.geometry.orthogonal
    if orth is None:
        raise ValueError("NMP die mapping requires orthogonal geometry")
    count = orth.slab_count
    pitch_m = orth.slab_pitch_x_um * 1e-6
    array_x0_m = 0.5 * (orth.cube_length_x_mm * 1e-3 - count * pitch_m)
    return tuple(
        NMPDieThermalRegion(
            die_id=i,
            region_id=f"orthogonal_hbm:die_{i + 1:03d}",
            geometry_die_index=i + 1,
            center_x_m=array_x0_m + (i + 0.5) * pitch_m,
        )
        for i in range(count)
    )
