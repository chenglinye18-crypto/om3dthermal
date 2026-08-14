"""Targeted closure tests for the nominal three-architecture comparison."""

from pathlib import Path

import pytest

from om3dthermal.cli import build_scene
from om3dthermal.architecture_comparison import (
    _resolved_capacity,
    compile_case_thermal,
)
from om3dthermal.power import (
    load_case_config,
    map_system_power_to_thermal,
    resolve_case_geometry,
    resolve_system_power,
)


ROOT = Path(__file__).parents[1]
CASES = ROOT / "configs" / "cases"
NAMES = (
    "conventional_hbm_2x1",
    "orthogonal_si",
    "orthogonal_m3d_igzo",
)


def _resolved(name):
    case = load_case_config(CASES / f"{name}.yaml")
    geometry = resolve_case_geometry(case)
    system = resolve_system_power(case, project_root=ROOT, geometry=geometry)
    return case, geometry, system


def test_system_scope_capacity_and_refresh_close():
    conventional = _resolved(NAMES[0])
    orth_si = _resolved(NAMES[1])
    m3d = _resolved(NAMES[2])
    capacity = _resolved_capacity(*conventional)
    assert capacity["system_capacity_GiB"] == 114.75
    assert capacity["capacity_per_instance_GiB"] == 57.375
    assert _resolved_capacity(*orth_si)["system_capacity_GiB"] == 234.28125
    assert _resolved_capacity(*m3d)["system_capacity_GiB"] == 428.75
    assert conventional[2].refresh_power_W == pytest.approx(
        0.8172465768010997)
    assert orth_si[2].refresh_power_W == pytest.approx(1.6685450943022453)
    assert m3d[2].refresh_power_W == 98 * 0.0003484694872064


def test_access_energy_regressions_and_system_bandwidth_are_frozen():
    expected = (1.3970979848163718, 1.3676557831180527,
                0.8552605756733209)
    for name, energy in zip(NAMES, expected):
        _, _, system = _resolved(name)
        assert system.read_bandwidth_gbps == 39200.0
        assert system.memory_access_energy_pJ_per_bit == energy
        assert system.memory_access_power_W == pytest.approx(
            energy * 39200.0 * 1e-3)


def test_resolved_to_thermal_source_closure_and_same_case_compile():
    for name in NAMES:
        case, _, system = _resolved(name)
        mapping = map_system_power_to_thermal(case, system)
        expected = system.gpu_power_W + system.resolved_total_memory_power_W
        assert mapping.total_mapped_power_W == pytest.approx(expected)
        assert mapping.unresolved is False
        thermal = compile_case_thermal(case, system)
        assert sum(source.total_power for source in
                   thermal.thermal_power_sources.sources) == pytest.approx(expected)
        assert thermal.metadata["case_id"] == case.name


@pytest.mark.parametrize("name", NAMES[1:])
def test_orthogonal_adhesive_thickness_comes_from_canonical_case(name):
    case, _, system = _resolved(name)
    thermal = compile_case_thermal(case, system)
    assert case.thermal["adhesive"]["thickness_um"] == 1.0
    assert thermal.orthogonal_hbm.adhesive.thickness == pytest.approx(1e-6)


def test_conventional_physical_geometry_drives_capacity_and_thermal_stack():
    case, geometry, system = _resolved(NAMES[0])
    diagnostics = system.diagnostics
    assert geometry.memory_region_count == 2
    assert geometry.memory_dies_per_region == 12
    assert (geometry.configured_x_mm, geometry.configured_y_mm) == (10.8, 21.8)
    assert case.geometry.layout["visible_group_footprint_mm"] == [11.0, 22.0]
    assert diagnostics["packed_banks_per_die"] == 306
    assert diagnostics["rotated_90_deg"] is True
    assert diagnostics["bits_per_stack"] == diagnostics["bits_per_die"] * 12
    assert diagnostics["total_stored_bits"] == diagnostics["bits_per_stack"] * 2
    thermal = compile_case_thermal(case, system)
    hbm = thermal.stack_templates["hbm_12hi"].model_dump()
    repeated = next(item for item in hbm["items"] if item["kind"] == "repeat")
    assert repeated["count"] == 11
    assert hbm["items"][1]["material"] == "HBM_Base_BEOL"
    assert hbm["items"][1]["thickness"] == pytest.approx(5e-6)
    assert hbm["items"][2]["material"] == "Silicon"
    assert hbm["items"][2]["thickness"] == pytest.approx(50e-6)


def test_conventional_base_route_maps_only_to_physical_base_beol():
    case, _, system = _resolved(NAMES[0])
    result = system.memory_result
    assert result is not None
    assert result.E_base_route_pj_bit > 0.0
    mapping = map_system_power_to_thermal(case, system)
    base = [source for source in mapping.sources
            if source.name.startswith("base_route_group_")]
    dram = [source for source in mapping.sources
            if source.name.startswith("dram_group_")]
    expected_base_W = (
        result.E_base_route_pj_bit * system.read_bandwidth_gbps * 1e-3)
    assert len(base) == len(dram) == 2
    assert sum(source.power_W for source in base) == pytest.approx(expected_base_W)
    assert sum(source.power_W for source in dram) == pytest.approx(
        system.resolved_total_memory_power_W - expected_base_W)
    thermal = compile_case_thermal(case, system)
    sources = {source.name: source for source in
               thermal.thermal_power_sources.sources}
    assert all(sources[source.name].selector.material == "HBM_Base_BEOL"
               for source in base)
    assert all(sources[source.name].selector.material == "DRAM_BEOL"
               for source in dram)
    assert result.P_logic_background_W == 0.0
    assert system.diagnostics["P_base_FEOL_logic_W"] == 0.0
    assert system.diagnostics["P_base_FEOL_logic_status"] == (
        "NOT_SEPARATELY_MODELED")


def test_conventional_has_two_physical_base_dies_and_775um_stack():
    case, _, system = _resolved(NAMES[0])
    thermal = compile_case_thermal(case, system)
    scene = build_scene(thermal)
    base_beol = [box for box in scene.boxes
                 if box.material == "HBM_Base_BEOL"]
    base_si = [box for box in scene.boxes
               if box.tags.get("layer") == "hbm_base_si"]
    assert len(base_beol) == len(base_si) == 2
    assert thermal.stack_templates["hbm_12hi"].total_thickness == pytest.approx(
        775e-6)
    assert all(box.z1 - box.z0 == pytest.approx(5e-6)
               for box in base_beol)
    assert all(box.z1 - box.z0 == pytest.approx(50e-6)
               for box in base_si)


def test_unified_memory_mapping_targets_complete_beol_only():
    for name in NAMES:
        case, _, system = _resolved(name)
        mapping = map_system_power_to_thermal(case, system)
        memory = [source for source in mapping.sources if source.name != "gpu"]
        assert sum(source.power_W for source in memory) == pytest.approx(
            system.resolved_total_memory_power_W)
        assert all(source.power_W > 0.0 for source in memory)
        assert all(source.target_region != "M3D_FEOL" for source in memory)
        thermal = compile_case_thermal(case, system)
        selectors = [source.selector for source in
                     thermal.thermal_power_sources.sources
                     if source.name != "gpu"]
        assert all(selector.material != "M3D_FEOL" for selector in selectors)
        assert all(selector.tags.get("role") != "feol" for selector in selectors)

    case, _, system = _resolved(NAMES[2])
    memory = map_system_power_to_thermal(case, system).sources[1:]
    assert {source.target_region for source in memory} == {
        "M3D_BITCELL_STACK", "M3D_BEOL_INTERCONNECT"}
    bitcell, interconnect = memory
    assert bitcell.power_W / interconnect.power_W == pytest.approx(2.304 / 3.0)


def test_density_denominators_are_geometry_derived():
    conventional = _resolved_capacity(*_resolved(NAMES[0]))
    orth_si = _resolved_capacity(*_resolved(NAMES[1]))
    m3d = _resolved_capacity(*_resolved(NAMES[2]))
    assert conventional["memory_plane_area_mm2"] == 10.8 * 21.8
    assert conventional["architecture_footprint_area_mm2"] == 2 * 11 * 22
    assert orth_si["memory_plane_area_mm2"] == 22 * 5.5
    assert orth_si["architecture_footprint_area_mm2"] == 30 * 22
    assert m3d["memory_plane_area_mm2"] == 22 * 5.5
    assert m3d["architecture_footprint_area_mm2"] == 30 * 22
