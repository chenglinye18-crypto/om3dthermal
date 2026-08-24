from pathlib import Path

import pytest

from om3dthermal.architecture import resolve_packing_from_legacy_power_result
from om3dthermal.architecture_capacity import resolve_architecture_capacity
from om3dthermal.power import (
    load_case_config,
    resolve_case_geometry,
    resolve_system_power,
)


ROOT = Path(__file__).parents[1]
CASES = (
    "conventional_hbm_2x1",
    "orthogonal_si",
    "orthogonal_m3d_igzo",
)


@pytest.mark.parametrize("name", CASES)
def test_resolved_packing_is_bit_exact_with_existing_capacity(name: str) -> None:
    case = load_case_config(ROOT / "configs" / "cases" / f"{name}.yaml")
    geometry = resolve_case_geometry(case)
    system = resolve_system_power(case, project_root=ROOT, geometry=geometry)
    assert system.memory_result is not None

    packing = resolve_packing_from_legacy_power_result(
        case, geometry, system.memory_result)
    legacy = resolve_architecture_capacity(case, geometry, system)

    assert packing.architecture_id == legacy.architecture
    assert packing.instance_count == legacy.instance_count
    assert packing.bits_per_instance == legacy.bits_per_instance
    assert packing.total_bits == legacy.total_bits
    assert packing.bits_per_plane == legacy.bits_per_plane
    assert packing.system_capacity_bytes == legacy.system_capacity_bytes
    assert packing.memory_plane_area_mm2 == legacy.memory_plane_area_mm2
    assert (packing.architecture_footprint_area_mm2
            == legacy.architecture_footprint_area_mm2)
