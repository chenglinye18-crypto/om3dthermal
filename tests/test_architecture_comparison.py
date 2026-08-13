"""Targeted closure tests for the nominal three-architecture comparison."""

from pathlib import Path

import pytest

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
    "conventional_hbm_2x1_nologic",
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
    assert _resolved_capacity(*conventional)["system_capacity_GiB"] == 64.0
    assert _resolved_capacity(*orth_si)["system_capacity_GiB"] == 234.28125
    assert _resolved_capacity(*m3d)["system_capacity_GiB"] == 428.75
    assert conventional[2].refresh_power_W == 4 * 0.11395159240799647
    assert orth_si[2].refresh_power_W == pytest.approx(1.6685450943022453)
    assert m3d[2].refresh_power_W == 98 * 0.0003484694872064


def test_access_energy_regressions_and_system_bandwidth_are_frozen():
    expected = (1.1589766571414901, 1.3676557831180527,
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
