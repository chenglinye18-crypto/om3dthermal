from pathlib import Path

import pytest

from om3dthermal.adapters import (
    extract_architecture_facts,
    resolve_architecture_spec,
)
from om3dthermal.architecture import resolve_packing_from_legacy_power_result
from om3dthermal.architecture_capacity import resolve_architecture_capacity
from om3dthermal.experiment.config import load_architecture_spec
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


@pytest.mark.parametrize("name", CASES)
def test_resolved_architecture_facts_are_exact_adapter_views(name: str) -> None:
    spec = load_architecture_spec(
        ROOT / "configs" / "architecture" / f"{name}.yaml",
        project_root=ROOT,
    )
    resolved = resolve_architecture_spec(spec, project_root=ROOT)

    facts = extract_architecture_facts(resolved)

    assert facts.architecture_id == spec.architecture_id
    assert facts.packing == resolved.packing
    assert facts.energy_primitives.read_access_energy_pj_per_bit == (
        resolved.system_power.memory_access_energy_pJ_per_bit
    )
    assert facts.energy_primitives.memory_internal_pj_per_bit == (
        resolved.system_power.memory_result.E_memory_internal_pj_bit
    )
    assert facts.provenance == spec.provenance
    assert facts.static_power.fixed_gpu_power_W == resolved.system_power.gpu_power_W
    assert facts.static_power.refresh_power_W == (
        resolved.system_power.memory_result.P_refresh_W
    )


def test_m3d_architecture_facts_preserve_unresolved_logic_background() -> None:
    spec = load_architecture_spec(
        ROOT / "configs" / "architecture" / "orthogonal_m3d_igzo.yaml",
        project_root=ROOT,
    )
    facts = extract_architecture_facts(
        resolve_architecture_spec(spec, project_root=ROOT)
    )

    assert facts.static_power.logic_background_power_W is None
    assert facts.static_power.completeness_status == (
        "UNRESOLVED_LOGIC_BACKGROUND"
    )
