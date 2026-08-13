"""Integer packing of DreamRAM physical primitives into orthogonal Si slabs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SiPrimitivePacking:
    primitive_type: str
    primitive_width_um: float
    primitive_height_um: float
    primitive_bits: int
    rotated_90_deg: bool
    packed_nx: int
    packed_ny: int
    primitives_per_slab: int
    packed_width_um: float
    packed_height_um: float
    packing_utilization: float
    bits_per_slab: int
    slab_count: int
    total_system_bits: int
    gib_per_slab: float
    total_system_gib: float
    gross_density_Mb_mm2: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _orientation(
        slab_width_um: float, slab_height_um: float,
        primitive_width_um: float, primitive_height_um: float,
        ) -> tuple[int, int, int]:
    nx = math.floor(slab_width_um / primitive_width_um)
    ny = math.floor(slab_height_um / primitive_height_um)
    return nx, ny, nx * ny


def pack_si_primitive(
        *, slab_width_um: float, slab_height_um: float, slab_count: int,
        primitive_type: str, primitive_width_um: float,
        primitive_height_um: float, primitive_bits: int,
        ) -> SiPrimitivePacking:
    """Pack complete primitives, choosing native or 90-degree orientation."""
    if min(slab_width_um, slab_height_um, primitive_width_um,
           primitive_height_um) <= 0:
        raise ValueError("slab and primitive dimensions must be positive")
    if slab_count <= 0 or primitive_bits <= 0:
        raise ValueError("slab count and primitive capacity must be positive")

    native = _orientation(
        slab_width_um, slab_height_um,
        primitive_width_um, primitive_height_um)
    rotated = _orientation(
        slab_width_um, slab_height_um,
        primitive_height_um, primitive_width_um)
    use_rotated = rotated[2] > native[2]
    nx, ny, count = rotated if use_rotated else native
    width = primitive_height_um if use_rotated else primitive_width_um
    height = primitive_width_um if use_rotated else primitive_height_um
    if count == 0:
        raise ValueError(
            f"no complete {primitive_type} fits in the configured Si slab")

    packed_width = nx * width
    packed_height = ny * height
    slab_area = slab_width_um * slab_height_um
    bits_per_slab = count * primitive_bits
    total_bits = bits_per_slab * slab_count
    return SiPrimitivePacking(
        primitive_type=primitive_type,
        primitive_width_um=primitive_width_um,
        primitive_height_um=primitive_height_um,
        primitive_bits=primitive_bits,
        rotated_90_deg=use_rotated,
        packed_nx=nx,
        packed_ny=ny,
        primitives_per_slab=count,
        packed_width_um=packed_width,
        packed_height_um=packed_height,
        packing_utilization=(count * width * height / slab_area),
        bits_per_slab=bits_per_slab,
        slab_count=slab_count,
        total_system_bits=total_bits,
        gib_per_slab=bits_per_slab / 8 / (2 ** 30),
        total_system_gib=total_bits / 8 / (2 ** 30),
        gross_density_Mb_mm2=bits_per_slab / 1e6 / (slab_area * 1e-6),
    )
