"""Geometry, energy and zero-refresh contracts for the No-NMP sensitivity."""

from pathlib import Path

import pytest

from om3dthermal.architecture import resolve_packing_from_legacy_power_result
from om3dthermal.bandwidth_thermal_sweep import _base_case, _point_simulation
from om3dthermal.power import calculate_memory_power, load_case_config, resolve_case_geometry
from om3dthermal.power.config import RefreshInput
from om3dthermal.thermal.setup_cache import build_scene_and_signature
from om3dthermal.thermal_sensitivity import (
    SENSITIVITY_ARCHITECTURES, prepare_case, summarize_curve,
)


ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def cases():
    return [prepare_case(ROOT, spec) for spec in SENSITIVITY_ARCHITECTURES]


@pytest.mark.parametrize("index", (0, 2))
def test_nominal_simulation_and_energy_are_unchanged(cases, index):
    case, simulation, memory, _, _ = cases[index]
    old_case, old_simulation, old_memory = _base_case(
        ROOT, SENSITIVITY_ARCHITECTURES[index])
    assert case == old_case
    assert simulation == old_simulation
    assert memory.as_dict() == old_memory.as_dict()


def test_24hi_changes_only_depth_and_doubles_capacity(cases):
    nominal, baseline, _, nominal_packing, _ = cases[0]
    case, simulation, _, packing, _ = cases[1]
    assert packing.total_bits == 2 * nominal_packing.total_bits
    expected = nominal.geometry.model_dump()
    expected["layout"]["dram_dies_per_stack"] = 24
    assert case.geometry.model_dump() == expected
    assert case.thermal == nominal.thermal
    assert simulation.thermal_boundary_conditions == baseline.thermal_boundary_conditions
    assert simulation.discretization == baseline.discretization
    nominal_boxes, _ = build_scene_and_signature(baseline)
    boxes, _ = build_scene_and_signature(simulation)
    for group in ("hbm_left", "hbm_right"):
        prefix = f"memory_column:{group}."
        layers = [box for box in boxes if box.name.startswith(prefix)
                  and box.material == "DRAM_BEOL"]
        old_layers = [box for box in nominal_boxes if box.name.startswith(prefix)
                      and box.material == "DRAM_BEOL"]
        assert len(layers) == 24
        assert len(old_layers) == 12
        assert {(b.x0, b.x1, b.y0, b.y1) for b in layers} == {
            (b.x0, b.x1, b.y0, b.y1) for b in old_layers}
        assert max(b.z1 for b in layers) - max(b.z1 for b in old_layers) == (
            pytest.approx(12 * 46e-6))


def test_hbm_row_states_recompute_dreamram_at_24hi(cases):
    case, _, memory, _, row = cases[1]
    raw = load_case_config(ROOT / "configs/cases" / SENSITIVITY_ARCHITECTURES[1].case_file)
    assert raw.memory.nominal_read_energy is None
    full, closed = row["full_row"], row["closed_row"]
    for state in (full, closed):
        assert state["diagnostics"]["electrical_resolved_stack_die_count"] == 24
        assert state["diagnostics"]["resolved_average_tsv_layers_crossed"] == 12
    assert full["diagnostics"]["effective_rd_per_act"] == 64
    assert closed["diagnostics"]["effective_rd_per_act"] == 1
    assert full["E_access_total_pj_bit"] == pytest.approx(1.419208075375544)
    assert closed["E_access_total_pj_bit"] == pytest.approx(3.9551503687074323)
    assert memory.E_access_total_pj_bit == pytest.approx(
        (full["E_access_total_pj_bit"] + closed["E_access_total_pj_bit"]) / 2)
    assert memory.E_access_total_pj_bit != pytest.approx(1.9955)
    assert case.memory.nominal_read_energy.nominal_pj_per_bit == memory.E_access_total_pj_bit


def test_300um_slab_closure_capacity_and_shared_physics(cases):
    nominal, baseline, old_memory, old_packing, _ = cases[2]
    case, simulation, memory, packing, _ = cases[3]
    stack = case.geometry.m3d_stack
    orth = case.geometry.orthogonal
    assert stack.si_substrate_um == 292.546
    assert stack.bitcell_layers == 8
    assert (stack.si_substrate_um + stack.feol_um + stack.beol_interconnect_um
            + stack.daa_um + stack.bitcell_layers * stack.bitcell_layer_pitch_nm / 1000) == (
                pytest.approx(300.0, abs=1e-12))
    assert orth.slab_count == 106
    assert orth.slab_pitch_x_um == 300.0
    assert orth.slab_count * orth.slab_pitch_x_um == pytest.approx(31800.0)
    assert packing.total_bits * 3 == old_packing.total_bits
    assert packing.total_bits / 8 == 497_947_770_880
    assert case.thermal == nominal.thermal
    assert case.thermal["edge_strip_material"] == "Cu"
    assert simulation.discretization == baseline.discretization
    assert simulation.thermal_boundary_conditions == baseline.thermal_boundary_conditions
    assert memory.E_access_total_pj_bit == old_memory.E_access_total_pj_bit


@pytest.mark.parametrize("index", (0, 1, 2, 3))
def test_mapped_power_has_no_memory_static_refresh_or_nmp(cases, index):
    _, simulation, memory, _, _ = cases[index]
    family = "conventional_hbm_2x1" if index < 2 else "orthogonal_m3d_igzo"
    if index in (1, 3):
        assert memory.P_refresh_W == memory.P_memory_background_W == memory.P_logic_background_W == 0
    for bandwidth in (0.0, 2.4):
        point, gpu, access = _point_simulation(simulation, family, memory, bandwidth)
        assert gpu == pytest.approx(74.0 + 8 * 11.68 * bandwidth)
        assert access == pytest.approx(8 * memory.E_access_total_pj_bit * bandwidth)
        assert sum(source.total_power for source in point.thermal_power_sources.sources) == (
            pytest.approx(gpu + access))
        assert all("nmp" not in source.name.lower() for source in point.thermal_power_sources.sources)


@pytest.mark.parametrize("name", ("conventional_hbm_2x1", "orthogonal_m3d_igzo", "orthogonal_si"))
def test_disabling_refresh_preserves_physical_capacity_and_access(name):
    case = load_case_config(ROOT / f"configs/cases/{name}.yaml")
    geometry = resolve_case_geometry(case)
    enabled = calculate_memory_power(case, project_root=ROOT, geometry=geometry, read_bandwidth_gbps=100)
    disabled_case = case.model_copy(update={"power": case.power.model_copy(
        update={"refresh": RefreshInput(enabled=False)})})
    disabled = calculate_memory_power(disabled_case, project_root=ROOT,
                                      geometry=geometry, read_bandwidth_gbps=100)
    assert disabled.P_refresh_W == 0
    assert enabled.P_refresh_W > 0
    assert disabled.P_access_W == enabled.P_access_W
    assert disabled.physical_capacity_layout == enabled.physical_capacity_layout
    assert resolve_packing_from_legacy_power_result(disabled_case, geometry, disabled) == (
        resolve_packing_from_legacy_power_result(case, geometry, enabled))


def test_each_geometry_has_distinct_power_independent_cache(cases):
    signatures = []
    for index, (_, simulation, memory, _, _) in enumerate(cases):
        _, signature = build_scene_and_signature(simulation)
        family = "conventional_hbm_2x1" if index < 2 else "orthogonal_m3d_igzo"
        point, _, _ = _point_simulation(simulation, family, memory, 2.4)
        assert build_scene_and_signature(point)[1] == signature
        signatures.append(signature)
    assert len(set(signatures)) == 4


def test_slope_uses_common_domain_and_local_matched_difference():
    rows = [{"bandwidth_TBps": round(index * 0.2, 10),
             "Tmax_C": 30 + 12 * round(index * 0.2, 10)} for index in range(25)]
    rows.append({"bandwidth_TBps": 6.7, "Tmax_C": 150.0})
    result = summarize_curve(rows)
    assert result["slope_K_per_TBps"] == pytest.approx(12.0)
    assert result["local_2p4_slope_K_per_TBps"] == pytest.approx(12.0)
    assert result["matched_2p4"]["Tmax_C"] == pytest.approx(58.8)
