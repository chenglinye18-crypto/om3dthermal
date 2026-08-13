"""Targeted tests for the config-driven Memory Power v0 framework."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from om3dthermal.power import (
    calculate_memory_power,
    load_case_config,
    load_power_config,
    resolve_case_geometry,
    run_memory_power,
)
from om3dthermal.power.backends import DreamRAMBackend, OperationTableCellModel
from om3dthermal.power.cell_model import (
    MissingCellReplacementError,
    ONE_T_ONE_C_SPECIFIC,
    REUSABLE_STRUCTURE,
)
from om3dthermal.power.config import MemoryPowerConfig, RowPolicy
from om3dthermal.power.geometry import (
    evaluate_geometry_fit,
    load_m3d_geometry,
)
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.power.miv import build_miv_topology
from om3dthermal.power.refresh import calculate_refresh_power


ROOT = Path(__file__).parents[1]
POWER_CONFIGS = ROOT / "configs" / "power"
CASE_CONFIGS = ROOT / "configs" / "cases"


def _with_row_utilization(config, value: float):
    workload = config.workload.model_copy(
        update={"row_policy": RowPolicy(
            activated_row_data_utilization=value)})
    return config.model_copy(update={"workload": workload})


def _with_component_replacement(
        config, *, required=("bl-act",), replacements=None):
    raw = config.model_dump()
    raw["memory"]["cell_model"] = {
        "type": "component_replacement",
        "replacement": {
            "mapping_status": "validated",
            "components": list(required),
            "component_energy_pj_per_bit": replacements or {},
        },
    }
    return MemoryPowerConfig.model_validate(raw)


def _m3d_subarray(config):
    geometry = load_m3d_geometry(ROOT, config.architecture.geometry_source)
    assert config.architecture.m3d_subarray is not None
    return calculate_m3d_subarray(
        config.architecture.m3d_subarray, geometry)


def _m3d_backend(config):
    return DreamRAMBackend(ROOT).calculate(
        config, m3d_subarray=_m3d_subarray(config))


@pytest.fixture(scope="module")
def conventional():
    return load_power_config(POWER_CONFIGS / "hbm3_si.yaml")


def test_all_four_configs_parse():
    for name in (
        "hbm3_si.yaml",
        "hbm3_si_logic_remove.yaml",
        "orthogonal_si.yaml",
        "orthogonal_m3d_igzo.yaml",
    ):
        load_power_config(POWER_CONFIGS / name)


def test_igzo_cell_geometry_comes_only_from_thermal_geometry_source():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    assert config.memory.cell_model.geometry is None
    geometry = load_m3d_geometry(ROOT, config.architecture.geometry_source)
    assert geometry.cell_area_um2 == pytest.approx(0.023)
    topology = _m3d_subarray(config)
    expected_F = 0.5 * 0.023 ** 0.5
    assert topology.F_um == pytest.approx(expected_F)
    assert topology.cell_pitch_x_um == pytest.approx(2 * expected_F)
    assert topology.cell_pitch_y_um == pytest.approx(2 * expected_F)
    assert topology.shared_row_selection_band_um == pytest.approx(
        4 * expected_F)
    assert topology.shared_column_write_selection_band_um == pytest.approx(
        2 * expected_F)


def test_igzo_geometry_does_not_change_operation_energy_or_hold():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    cell_model = config.memory.cell_model
    energy = cell_model.operations
    assert energy is not None
    assert (
        energy.read_0_pj_per_bit, energy.read_1_pj_per_bit,
    ) == pytest.approx((0.00060, 0.36800))
    assert (
        energy.write_00_pj_per_bit, energy.write_01_pj_per_bit,
        energy.write_10_pj_per_bit, energy.write_11_pj_per_bit,
    ) == pytest.approx((0.00030, 0.00037, 0.00058, 0.00024))
    assert (
        energy.refresh_0_pj_per_bit, energy.refresh_1_pj_per_bit,
    ) == pytest.approx((0.00090, 0.37000))
    assert cell_model.background is not None
    assert cell_model.background.type == "per_row"
    assert cell_model.background.value_w == pytest.approx(4.26e-15)


def test_igzo_table_i_operation_values_and_provenance_are_frozen():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    cell_model = config.memory.cell_model
    operations = cell_model.operations
    provenance = cell_model.operation_energy_provenance
    replacement = cell_model.replacement
    assert cell_model.source == "IEDM2026_HaotongZhu_V5"
    assert operations is not None
    assert operations.model_dump() == pytest.approx({
        "read_0_pj_per_bit": 0.00060,
        "read_1_pj_per_bit": 0.36800,
        "write_00_pj_per_bit": 0.00030,
        "write_01_pj_per_bit": 0.00037,
        "write_10_pj_per_bit": 0.00058,
        "write_11_pj_per_bit": 0.00024,
        "refresh_0_pj_per_bit": 0.00090,
        "refresh_1_pj_per_bit": 0.37000,
    })
    assert provenance is not None
    assert provenance.source == "IEDM2026_HaotongZhu_V5"
    assert provenance.classification == "PAPER_REPORTED"
    assert provenance.accounting_level == (
        "SPICE_EXTRACTED_MAT_LOCAL_OPERATION_ENERGY")
    assert provenance.sensing_included is True
    assert provenance.distributed_rc_included is True
    assert replacement is not None
    assert replacement.mapping_status == "validated"
    assert replacement.energy_source == "operation_table"
    assert replacement.components == ()
    assert replacement.component_energy_pj_per_bit == {}


def test_subarray_cluster_auto_floorplan_and_explicit_override():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    auto = _m3d_subarray(config)
    assert auto.cluster_count_x == int(auto.slab_x_um // auto.cluster_width_um)
    assert auto.cluster_count_y == int(
        auto.slab_y_um // auto.cluster_height_um)
    assert auto.placed_width_um <= auto.slab_x_um
    assert auto.placed_height_um <= auto.slab_y_um
    assert auto.cluster_grid_x_source == "auto_floor"
    assert auto.cluster_grid_y_source == "auto_floor"

    cluster = config.architecture.m3d_subarray.subarray_cluster
    grid = cluster.grid.model_copy(update={
        "nx": auto.cluster_count_x - 1,
        "ny": auto.cluster_count_y - 1,
    })
    explicit_cluster = cluster.model_copy(update={"grid": grid})
    explicit_spec = config.architecture.m3d_subarray.model_copy(
        update={"subarray_cluster": explicit_cluster})
    geometry = load_m3d_geometry(ROOT, config.architecture.geometry_source)
    explicit = calculate_m3d_subarray(explicit_spec, geometry)
    assert explicit.cluster_count_x == auto.cluster_count_x - 1
    assert explicit.cluster_count_y == auto.cluster_count_y - 1
    assert explicit.cluster_grid_x_source == "explicit_override"
    assert explicit.cluster_grid_y_source == "explicit_override"

    invalid_grid = cluster.grid.model_copy(
        update={"nx": auto.cluster_count_x + 1})
    invalid_cluster = cluster.model_copy(update={"grid": invalid_grid})
    invalid_spec = config.architecture.m3d_subarray.model_copy(
        update={"subarray_cluster": invalid_cluster})
    with pytest.raises(ValueError, match="exceeds fit limit"):
        calculate_m3d_subarray(invalid_spec, geometry)


def test_m3d_subarray_schema_defaults_counts_to_auto():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    raw = config.architecture.m3d_subarray.subarray_cluster.grid.model_dump()
    raw.pop("nx")
    raw.pop("ny")
    validated = type(
        config.architecture.m3d_subarray.subarray_cluster.grid).model_validate(raw)
    assert validated.nx == "auto"
    assert validated.ny == "auto"


def test_subarray_cluster_config_rejects_illegal_dimensions():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    raw = config.model_dump()
    raw["architecture"]["m3d_subarray"]["subarray_cluster"][
        "subarrays_x"] = 0
    with pytest.raises(ValueError, match="greater than 0"):
        MemoryPowerConfig.model_validate(raw)


def test_nrow_ncol_change_topology_and_routing():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    baseline = _m3d_subarray(config)
    core = config.architecture.m3d_subarray.subarray.model_copy(
        update={"n_rows": 256, "n_cols": 256})
    spec = config.architecture.m3d_subarray.model_copy(
        update={"subarray": core})
    geometry = load_m3d_geometry(ROOT, config.architecture.geometry_source)
    changed = calculate_m3d_subarray(spec, geometry)
    assert changed.subarray_width_um != pytest.approx(
        baseline.subarray_width_um)
    assert changed.subarray_height_um != pytest.approx(
        baseline.subarray_height_um)
    assert changed.cluster_width_um != pytest.approx(baseline.cluster_width_um)
    assert changed.local_rbl_route_length_um != pytest.approx(
        baseline.local_rbl_route_length_um)
    assert changed.local_rbl_energy_pj_per_bit == 0.0


def test_shared_bands_apply_once_per_cluster_and_mux_per_subarray():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    topology = _m3d_subarray(config)
    assert topology.subarray_width_um == pytest.approx(topology.core_width_um)
    assert topology.subarray_height_um == pytest.approx(
        topology.core_height_um + topology.local_mux_footprint_height_um)
    assert topology.cluster_width_um == pytest.approx(
        topology.cluster_array_width_um
        + topology.shared_row_selection_band_um)
    assert topology.cluster_height_um == pytest.approx(
        topology.cluster_array_height_um
        + topology.shared_column_write_selection_band_um)
    assert topology.cluster_subarrays_x == 8
    assert topology.cluster_subarrays_y == 8
    assert topology.subarrays_per_cluster == 64
    assert topology.subarray_spacing_width_overhead_um == pytest.approx(
        7 * topology.subarray_gap_x_um)
    assert topology.subarray_spacing_height_overhead_um == pytest.approx(
        7 * topology.subarray_gap_y_um)
    assert topology.cluster_array_width_um == pytest.approx(
        topology.cluster_array_width_without_spacing_um
        + topology.subarray_spacing_width_overhead_um)
    assert topology.cluster_array_height_um == pytest.approx(
        topology.cluster_array_height_without_spacing_um
        + topology.subarray_spacing_height_overhead_um)
    assert topology.placed_width_um == pytest.approx(
        topology.cluster_count_x * topology.cluster_width_um
        + (topology.cluster_count_x - 1) * topology.cluster_gap_x_um)
    assert topology.placed_height_um == pytest.approx(
        topology.cluster_count_y * topology.cluster_height_um
        + (topology.cluster_count_y - 1) * topology.cluster_gap_y_um)
    assert topology.global_rwl_route_length_um_per_cluster < 1000.0
    assert topology.global_wwl_route_length_um_per_cluster < 1000.0
    assert topology.global_wbl_route_length_um_per_cluster < 1000.0
    assert topology.global_peripheral_instances_per_layer == (
        topology.clusters_per_layer)
    assert topology.local_mux_instances_per_layer == (
        topology.subarrays_per_layer)


def test_global_energy_uses_active_clusters_and_local_energy_uses_subarrays():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    baseline = _m3d_subarray(config)
    cluster = config.architecture.m3d_subarray.subarray_cluster
    smaller_grid = cluster.grid.model_copy(update={
        "nx": baseline.cluster_count_x - 1,
        "ny": baseline.cluster_count_y,
    })
    smaller_cluster = cluster.model_copy(update={"grid": smaller_grid})
    smaller_spec = config.architecture.m3d_subarray.model_copy(
        update={"subarray_cluster": smaller_cluster})
    geometry = load_m3d_geometry(ROOT, config.architecture.geometry_source)
    smaller = calculate_m3d_subarray(smaller_spec, geometry)
    # Grid affects physical shared-route length, but energy is not multiplied
    # by total instantiated subarray count.
    expected_raw = (
        smaller.interconnect_electrical["global_rwl"]["active_line_count"]
        * smaller.interconnect_electrical["global_rwl"]["activity_factor"]
        * smaller.interconnect_electrical["global_rwl"][
            "capacitance_fF_per_um"]
        * smaller.global_rwl_route_length_um_per_cluster
        * smaller.interconnect_electrical["global_rwl"]["voltage_V"] ** 2
        * 1e-3
        * smaller.accessed_clusters_per_access)
    assert smaller.global_rwl_raw_energy_pJ_per_access == pytest.approx(expected_raw)
    assert smaller.global_control_routing_energy_pj_per_bit == pytest.approx(
        baseline.global_control_routing_energy_pj_per_bit)

    doubled_access = config.architecture.m3d_subarray.access.model_copy(
        update={
            "accessed_subarrays_per_access": 512,
            "accessed_clusters_per_access": 8,
        })
    access_spec = config.architecture.m3d_subarray.model_copy(
        update={"access": doubled_access})
    doubled = calculate_m3d_subarray(access_spec, geometry)
    assert doubled.local_rbl_raw_energy_pJ_per_access == pytest.approx(
        2 * baseline.local_rbl_raw_energy_pJ_per_access)
    assert doubled.local_mux_raw_energy_pJ_per_access == pytest.approx(
        2 * baseline.local_mux_raw_energy_pJ_per_access)
    assert doubled.global_rwl_raw_energy_pJ_per_access == pytest.approx(
        2 * baseline.global_rwl_raw_energy_pJ_per_access)

    five_clusters = config.architecture.m3d_subarray.access.model_copy(
        update={"accessed_clusters_per_access": 5})
    five_spec = config.architecture.m3d_subarray.model_copy(
        update={"access": five_clusters})
    five = calculate_m3d_subarray(five_spec, geometry)
    assert five.global_rwl_raw_energy_pJ_per_access == pytest.approx(
        1.25 * baseline.global_rwl_raw_energy_pJ_per_access)


def test_cluster_geometry_controls_global_route_not_slab_size():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    geometry = load_m3d_geometry(ROOT, config.architecture.geometry_source)
    baseline = _m3d_subarray(config)
    cluster = config.architecture.m3d_subarray.subarray_cluster.model_copy(
        update={"subarrays_x": 4, "subarrays_y": 4})
    access = config.architecture.m3d_subarray.access.model_copy(
        update={"accessed_clusters_per_access": 16})
    changed_spec = config.architecture.m3d_subarray.model_copy(
        update={"subarray_cluster": cluster, "access": access})
    changed = calculate_m3d_subarray(changed_spec, geometry)
    assert changed.global_rwl_route_length_um_per_cluster == pytest.approx(
        4 * changed.subarray_width_um + 3 * changed.subarray_gap_x_um)
    assert changed.global_wbl_route_length_um_per_cluster == pytest.approx(
        4 * changed.subarray_height_um + 3 * changed.subarray_gap_y_um)
    assert changed.global_control_routing_energy_pj_per_bit != pytest.approx(
        baseline.global_control_routing_energy_pj_per_bit)

    larger_slab = replace(geometry, slab_x_um=30000.0, slab_y_um=7000.0)
    larger = calculate_m3d_subarray(
        config.architecture.m3d_subarray, larger_slab)
    assert larger.clusters_per_layer > baseline.clusters_per_layer
    assert larger.global_rwl_route_length_um_per_cluster == pytest.approx(
        baseline.global_rwl_route_length_um_per_cluster)
    assert larger.global_wbl_route_length_um_per_cluster == pytest.approx(
        baseline.global_wbl_route_length_um_per_cluster)
    assert larger.global_control_routing_energy_pj_per_bit == pytest.approx(
        baseline.global_control_routing_energy_pj_per_bit)


def test_spacing_hierarchy_changes_only_its_physical_scope():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    geometry = load_m3d_geometry(ROOT, config.architecture.geometry_source)
    baseline = _m3d_subarray(config)

    one_wide_cluster = config.architecture.m3d_subarray.subarray_cluster.model_copy(
        update={"subarrays_x": 1, "subarrays_y": 1})
    one_access = config.architecture.m3d_subarray.access.model_copy(update={
        "accessed_clusters_per_access": 256,
    })
    one_spec = config.architecture.m3d_subarray.model_copy(update={
        "subarray_cluster": one_wide_cluster,
        "access": one_access,
    })
    one = calculate_m3d_subarray(one_spec, geometry)
    assert one.subarray_spacing_width_overhead_um == 0.0
    assert one.subarray_spacing_height_overhead_um == 0.0

    wider_spacing = config.architecture.m3d_subarray.spacing.model_copy(update={
        "subarray_gap_x_f": 8.0,
        "subarray_gap_y_f": 8.0,
    })
    wider_spec = config.architecture.m3d_subarray.model_copy(
        update={"spacing": wider_spacing})
    wider = calculate_m3d_subarray(wider_spec, geometry)
    assert wider.cluster_width_um > baseline.cluster_width_um
    assert wider.cluster_height_um > baseline.cluster_height_um
    assert wider.global_rwl_route_length_um_per_cluster > (
        baseline.global_rwl_route_length_um_per_cluster)
    assert wider.global_wbl_route_length_um_per_cluster > (
        baseline.global_wbl_route_length_um_per_cluster)
    assert wider.global_control_routing_energy_pj_per_bit > (
        baseline.global_control_routing_energy_pj_per_bit)

    large_cluster_gap = config.architecture.m3d_subarray.spacing.model_copy(
        update={"cluster_gap_x_f": 500.0, "cluster_gap_y_f": 500.0})
    gap_spec = config.architecture.m3d_subarray.model_copy(
        update={"spacing": large_cluster_gap})
    gapped = calculate_m3d_subarray(gap_spec, geometry)
    assert gapped.clusters_per_layer < baseline.clusters_per_layer
    assert gapped.layout_utilization != pytest.approx(baseline.layout_utilization)
    assert gapped.global_rwl_route_length_um_per_cluster == pytest.approx(
        baseline.global_rwl_route_length_um_per_cluster)
    assert gapped.global_wbl_route_length_um_per_cluster == pytest.approx(
        baseline.global_wbl_route_length_um_per_cluster)
    assert gapped.global_control_routing_energy_pj_per_bit == pytest.approx(
        baseline.global_control_routing_energy_pj_per_bit)
    assert gapped.local_read_routing_energy_pj_per_bit == pytest.approx(
        baseline.local_read_routing_energy_pj_per_bit)


def test_nominal_spacing_diagnostics_density_and_miv_independence():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    baseline_topology = _m3d_subarray(config)
    assert baseline_topology.subarray_gap_x_f == pytest.approx(2.0)
    assert baseline_topology.subarray_gap_y_f == pytest.approx(2.0)
    assert baseline_topology.cluster_gap_x_f == pytest.approx(2.0)
    assert baseline_topology.cluster_gap_y_f == pytest.approx(2.0)
    assert baseline_topology.spacing_provenance == "MODELING_CHOICE"
    expected_density = (
        baseline_topology.bits_per_layer / 1e6
        / (baseline_topology.slab_x_um * baseline_topology.slab_y_um * 1e-6))
    assert baseline_topology.effective_density_Mb_per_mm2 == pytest.approx(
        expected_density)

    zero_cluster_gap = config.architecture.m3d_subarray.spacing.model_copy(
        update={"cluster_gap_x_f": 0.0, "cluster_gap_y_f": 0.0})
    zero_spec = config.architecture.m3d_subarray.model_copy(
        update={"spacing": zero_cluster_gap})
    zero_architecture = config.architecture.model_copy(
        update={"m3d_subarray": zero_spec})
    zero_config = config.model_copy(update={"architecture": zero_architecture})
    baseline_result = calculate_memory_power(config, project_root=ROOT)
    zero_result = calculate_memory_power(zero_config, project_root=ROOT)
    assert zero_result.diagnostics["cluster_count_x"] == (
        baseline_result.diagnostics["cluster_count_x"])
    assert zero_result.diagnostics["cluster_count_y"] == (
        baseline_result.diagnostics["cluster_count_y"])
    assert zero_result.E_vertical_pj_bit == pytest.approx(
        baseline_result.E_vertical_pj_bit, abs=0.0)


def test_impossible_cluster_access_fails_loudly():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    geometry = load_m3d_geometry(ROOT, config.architecture.geometry_source)
    access = config.architecture.m3d_subarray.access.model_copy(update={
        "accessed_clusters_per_access": 4,
        "accessed_subarrays_per_access": 257,
    })
    spec = config.architecture.m3d_subarray.model_copy(
        update={"access": access})
    with pytest.raises(ValueError, match="capacity of accessed clusters"):
        calculate_m3d_subarray(spec, geometry)


def test_existing_geometry_sources_are_resolved_for_all_power_configs():
    expected = {
        "hbm3_si.yaml": (10.8, 10.8, "hbm_dram_die"),
        "hbm3_si_logic_remove.yaml": (10.8, 10.8, "hbm_dram_die"),
        "orthogonal_si.yaml": (22.0, 5.5, "orthogonal_memory_slab"),
        "orthogonal_m3d_igzo.yaml": (
            22.0, 5.5, "orthogonal_m3d_slab"),
    }
    for name, (configured_x, configured_y, region) in expected.items():
        config = load_power_config(POWER_CONFIGS / name)
        backend = (
            _m3d_backend(config)
            if config.architecture.m3d_subarray is not None
            else DreamRAMBackend(ROOT).calculate(config))
        assert backend.metadata["configured_x_mm"] == pytest.approx(configured_x)
        assert backend.metadata["configured_y_mm"] == pytest.approx(configured_y)
        assert backend.metadata["memory_region"] == region
        assert backend.metadata["x_utilization"] == pytest.approx(
            backend.metadata["required_x_mm"] / configured_x)
        assert backend.metadata["y_utilization"] == pytest.approx(
            backend.metadata["required_y_mm"] / configured_y)
        assert backend.metadata["geometry_feasible"] == (
            backend.metadata["required_x_mm"] <= configured_x
            and backend.metadata["required_y_mm"] <= configured_y)


@pytest.mark.parametrize(
    "configured_x, configured_y, expected",
    [(11.0, 12.0, True), (9.0, 12.0, False), (11.0, 9.0, False)],
)
def test_geometry_fit_checks_each_axis(configured_x, configured_y, expected):
    fit = evaluate_geometry_fit(
        configured_x_mm=configured_x,
        configured_y_mm=configured_y,
        required_x_mm=10.0,
        required_y_mm=10.0,
    )
    assert fit.geometry_feasible is expected


def test_dreamram_hbm3_full_row_regression(conventional):
    result = calculate_memory_power(conventional, project_root=ROOT)
    assert result.E_access_total_pj_bit == 0.9782367130708566
    assert result.P_access_W == pytest.approx(
        39200 * result.E_access_total_pj_bit * 1e-3)
    assert result.P_logic_background_W is None
    assert result.P_total_W is None
    assert result.diagnostics["activated_row_data_utilization"] == 1.0
    assert result.diagnostics[
        "activated_row_data_utilization_percent"] == 100.0
    assert result.diagnostics["minimum_activated_row_data_utilization"] == (
        1 / 64)
    assert result.diagnostics[
        "minimum_activated_row_data_utilization_percent"] == 1.5625
    assert result.diagnostics["effective_rd_per_act"] == 64.0
    assert result.diagnostics["atoms_per_page"] == 64


def test_dreamram_hbm3_closed_row_regression(conventional):
    result = calculate_memory_power(
        _with_row_utilization(conventional, 1 / 64), project_root=ROOT)
    assert result.E_access_total_pj_bit == pytest.approx(3.0133613062)


def test_dreamram_decomposition_closes(conventional):
    result = calculate_memory_power(conventional, project_root=ROOT)
    reconstructed = (
        result.E_memory_internal_pj_bit + result.E_vertical_pj_bit
        + result.E_base_route_pj_bit + result.E_interface_pj_bit)
    assert reconstructed == pytest.approx(result.E_access_total_pj_bit)


def test_internal_component_partition_closes(conventional):
    result = calculate_memory_power(conventional, project_root=ROOT)
    components = result.diagnostics["native_components_pj_bit"]
    assert set(components) == ONE_T_ONE_C_SPECIFIC | REUSABLE_STRUCTURE
    specific = sum(components[name] for name in ONE_T_ONE_C_SPECIFIC)
    reusable = sum(components[name] for name in REUSABLE_STRUCTURE)
    assert specific + reusable == pytest.approx(
        result.E_memory_internal_pj_bit, abs=1e-15)


@pytest.mark.parametrize(
    "utilization, expected_rd",
    [
        (1 / 64, 1), (2 / 64, 2), (4 / 64, 4), (8 / 64, 8),
        (16 / 64, 16), (32 / 64, 32), (64 / 64, 64),
    ],
)
def test_activated_row_utilization_resolves_effective_rd_count(
        conventional, utilization, expected_rd):
    result = calculate_memory_power(
        _with_row_utilization(conventional, utilization), project_root=ROOT)
    assert result.diagnostics["activated_row_data_utilization"] == utilization
    assert result.diagnostics["effective_rd_per_act"] == expected_rd
    expected_energy = (
        result.diagnostics["E_PRE_pJ"] + result.diagnostics["E_ACT_pJ"]
        + expected_rd * result.diagnostics["E_RD_pJ"]
    ) / (expected_rd * result.diagnostics["atom_size_bits"])
    assert result.E_access_total_pj_bit == pytest.approx(expected_energy)


@pytest.mark.parametrize("utilization", [0.0, -0.1, 1.01])
def test_activated_row_utilization_range_fails(utilization):
    with pytest.raises(ValueError):
        RowPolicy(activated_row_data_utilization=utilization)


@pytest.mark.parametrize(
    "utilization, expected_rd", [(0.30, 19.2), (0.10, 6.4)])
def test_fractional_effective_rd_per_act_succeeds(
        conventional, utilization, expected_rd):
    result = calculate_memory_power(
        _with_row_utilization(conventional, utilization), project_root=ROOT)
    assert result.diagnostics["effective_rd_per_act"] == pytest.approx(
        expected_rd)
    expected_energy = (
        result.diagnostics["E_PRE_pJ"] + result.diagnostics["E_ACT_pJ"]
        + expected_rd * result.diagnostics["E_RD_pJ"]
    ) / (expected_rd * result.diagnostics["atom_size_bits"])
    assert result.E_access_total_pj_bit == pytest.approx(expected_energy)


def test_utilization_below_one_transfer_fails(conventional):
    with pytest.raises(ValueError, match="at least 1/atoms_per_page=0.015625"):
        calculate_memory_power(
            _with_row_utilization(conventional, 0.01), project_root=ROOT)


def test_m3d_uses_control_address_reuse_not_hbm_row_policy():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    assert config.workload.row_policy is None
    assert config.workload.control_address_reuse == 64
    result = calculate_memory_power(config, project_root=ROOT)
    assert result.diagnostics["control_address_reuse"] == 64
    assert "activated_row_data_utilization" not in result.diagnostics
    assert "effective_rd_per_act" not in result.diagnostics
    assert result.E_access_total_pj_bit == pytest.approx(
        0.8552605756733209, abs=0.0)
    assert result.E_vertical_pj_bit == pytest.approx(
        0.002445862111816407, abs=0.0)
    assert result.P_refresh_W == pytest.approx(
        0.0003484694872064, abs=0.0)


def test_orthogonal_si_keeps_same_dreamram_internal_energy(conventional):
    hbm = calculate_memory_power(conventional, project_root=ROOT)
    orthogonal = calculate_memory_power(
        load_power_config(POWER_CONFIGS / "orthogonal_si.yaml"),
        project_root=ROOT)
    assert orthogonal.E_memory_internal_pj_bit == pytest.approx(
        hbm.E_memory_internal_pj_bit, abs=0.0)
    assert orthogonal.E_vertical_pj_bit == 0.0
    assert orthogonal.E_base_route_pj_bit == 0.0
    assert orthogonal.E_interface_pj_bit == pytest.approx(0.5)
    assert orthogonal.E_memory_internal_pj_bit == pytest.approx(0.6288729797)
    assert orthogonal.E_access_total_pj_bit == pytest.approx(1.1288729797)


def test_igzo_then_si_has_no_shared_tech_contamination(conventional):
    before = calculate_memory_power(conventional, project_root=ROOT)
    calculate_memory_power(
        load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml"),
        project_root=ROOT)
    after = calculate_memory_power(conventional, project_root=ROOT)
    assert after.E_access_total_pj_bit == pytest.approx(
        before.E_access_total_pj_bit, abs=0.0)
    assert after.diagnostics["pitch_bl_um"] == before.diagnostics["pitch_bl_um"]
    assert after.diagnostics["pitch_wl_um"] == before.diagnostics["pitch_wl_um"]


def test_logic_removed_only_drops_dreamram_base_route(conventional):
    baseline = calculate_memory_power(conventional, project_root=ROOT)
    removed = calculate_memory_power(
        load_power_config(POWER_CONFIGS / "hbm3_si_logic_remove.yaml"),
        project_root=ROOT)
    assert removed.E_memory_internal_pj_bit == pytest.approx(
        baseline.E_memory_internal_pj_bit, abs=0.0)
    assert removed.E_vertical_pj_bit == pytest.approx(
        baseline.E_vertical_pj_bit, abs=0.0)
    assert removed.E_interface_pj_bit == pytest.approx(
        baseline.E_interface_pj_bit, abs=0.0)
    assert removed.E_base_route_pj_bit == 0.0
    assert removed.P_logic_background_W == 0.0
    assert removed.P_total_W == pytest.approx(removed.P_access_W)


def test_synthetic_replacement_has_no_double_count(conventional):
    native = calculate_memory_power(conventional, project_root=ROOT)
    old_component = native.diagnostics["native_components_pj_bit"]["bl-act"]
    replacement = 0.125
    modified = calculate_memory_power(
        _with_component_replacement(
            conventional, replacements={"bl-act": replacement}),
        project_root=ROOT)
    expected = native.E_memory_internal_pj_bit - old_component + replacement
    assert modified.E_memory_internal_pj_bit == pytest.approx(
        expected, abs=1e-15)
    assert "bl-act" not in modified.diagnostics["native_components_pj_bit"]
    assert modified.diagnostics["replacement_components_pj_bit"] == {
        "bl-act": replacement}


def test_modified_internal_is_architecture_independent(conventional):
    modified = _with_component_replacement(
        conventional, replacements={"bl-act": 0.125})
    orthogonal_architecture = load_power_config(
        POWER_CONFIGS / "orthogonal_si.yaml").architecture
    orthogonal = modified.model_copy(
        update={"architecture": orthogonal_architecture})
    hbm_result = calculate_memory_power(modified, project_root=ROOT)
    orthogonal_result = calculate_memory_power(orthogonal, project_root=ROOT)
    assert orthogonal_result.E_memory_internal_pj_bit == pytest.approx(
        hbm_result.E_memory_internal_pj_bit, abs=0.0)
    assert hbm_result.E_vertical_pj_bit != orthogonal_result.E_vertical_pj_bit
    assert hbm_result.E_base_route_pj_bit != orthogonal_result.E_base_route_pj_bit
    assert hbm_result.E_interface_pj_bit != orthogonal_result.E_interface_pj_bit


def test_required_replacement_missing_fails_loudly(conventional):
    config = _with_component_replacement(
        conventional,
        required=("bl-act", "bl-pre"),
        replacements={"bl-act": 0.125})
    with pytest.raises(
            MissingCellReplacementError,
            match="required replacement components are unresolved: bl-pre"):
        calculate_memory_power(config, project_root=ROOT)


def test_operation_table_is_device_energy_not_complete_memory_energy():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    device = OperationTableCellModel().calculate(config)
    assert device.weighted_read(p0=0.5, p1=0.5) == pytest.approx(0.1843)
    assert device.weighted_write(
        p00=0.25, p01=0.25, p10=0.25, p11=0.25,
    ) == pytest.approx((0.00030 + 0.00037 + 0.00058 + 0.00024) / 4)
    assert device.background_value_W == pytest.approx(4.26e-15)


def test_m3d_internal_uses_zhu_tang_and_no_dreamram_hierarchy():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    result = calculate_memory_power(config, project_root=ROOT)
    operation = 0.5 * 0.00060 + 0.5 * 0.36800
    replacement = result.diagnostics["replacement_components_pj_bit"]
    assert result.E_memory_internal_pj_bit == pytest.approx(
        operation
        + replacement["tang_global_control_routing"], abs=1e-15)
    assert result.diagnostics["native_components_pj_bit"] == {}
    assert replacement["zhu_scaled_local_operation"] == pytest.approx(operation)
    assert result.diagnostics["dreamram_hierarchy_included"] is False
    assert result.diagnostics["global_control_scope"] == "SUBARRAY_CLUSTER"
    assert result.diagnostics["dreamram_planar_organization_used"] is False
    assert result.diagnostics["dreamram_internal_components_used"] is False
    assert not ({"mat_x_um", "bank_x_um", "wire_lengths_um"}
                & set(result.diagnostics))
    excluded = set(result.diagnostics["dreamram_internal_components_excluded"])
    assert excluded == {
        "row", "mwl", "lwl", "bl-act", "bl-pre", "col", "csl",
        "ldl", "mdl", "bgbus+gbus",
    }
    assert "current_sense_energy" not in result.diagnostics
    assert "CSA_energy" not in result.diagnostics
    assert "RBL_energy" not in result.diagnostics


def test_m3d_topology_uses_geometry_layers_and_uniform_access():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    metadata = _m3d_backend(config).metadata
    assert config.architecture.layers is None
    assert metadata["m3d_layers"] == 8
    assert metadata["layer_pitch_um"] == pytest.approx(0.288)
    assert metadata["m3d_layers_source"] == (
        "geometry_source.m3d_beol.bitcell_layers")
    assert metadata["layer_pitch_source"] == (
        "geometry_source.m3d_beol.bitcell_layer_pitch_nm")
    assert metadata["layer_access_assumption"] == "uniform"
    assert metadata["miv_length_per_layer_um"] == pytest.approx(
        tuple(0.288 * index for index in range(1, 9)))
    assert metadata["miv_average_length_um"] == pytest.approx(1.296)
    assert metadata["m3d_layers_independent_of_dies_stacked"] is True
    assert metadata["dies_stacked"] == 8


def test_miv_length_changes_with_layers_and_pitch_not_dies_stacked():
    common = {
        "data_width_before_vertical": 17,
        "vertical_serialization_factor": "unresolved",
        "row_miv_count": 24,
        "col_miv_count": 19,
    }
    four_layers = build_miv_topology(
        m3d_layers=4, layer_pitch_um=0.288, **common)
    eight_layers = build_miv_topology(
        m3d_layers=8, layer_pitch_um=0.288, **common)
    wider_pitch = build_miv_topology(
        m3d_layers=8, layer_pitch_um=0.4, **common)
    assert four_layers.miv_average_length_um == pytest.approx(0.72)
    assert eight_layers.miv_average_length_um == pytest.approx(1.296)
    assert wider_pitch.miv_average_length_um == pytest.approx(1.8)
    # No dies_stacked argument exists: physical die count cannot enter MIV
    # length, and 4-layer average is not DreamRAM dies_stacked/2 * pitch.
    assert four_layers.miv_average_length_um != pytest.approx(4 * 0.288)


@pytest.mark.parametrize(
    "layers,pitch_nm,expected_average_um",
    [(4, 288.0, 0.72), (8, 400.0, 1.8)],
)
def test_miv_adapter_tracks_existing_geometry_config(
        tmp_path, layers, pitch_nm, expected_average_um):
    source_path = ROOT / "configs" / "orthogonal_m3d_edram_v0.yaml"
    raw_geometry = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    stack_um = layers * pitch_nm * 1e-3
    raw_geometry["m3d_beol"].update({
        "bitcell_layers": layers,
        "bitcell_layer_pitch_nm": pitch_nm,
        "bitcell_stack_um": stack_um,
        "total_um": stack_um + raw_geometry["m3d_beol"]["interconnect_um"],
    })
    raw_geometry["m3d_memory"]["layers"] = layers
    raw_geometry["slab"]["si_substrate_um"] = (
        raw_geometry["slab"]["total_pitch_um"]
        - raw_geometry["slab"]["feol_um"]
        - stack_um
        - raw_geometry["m3d_beol"]["interconnect_um"]
        - raw_geometry["orthogonal"]["daa_um"])
    geometry_path = tmp_path / "m3d_geometry.yaml"
    geometry_path.write_text(
        yaml.safe_dump(raw_geometry, sort_keys=False), encoding="utf-8")

    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    geometry_source = config.architecture.geometry_source.model_copy(
        update={"config": geometry_path})
    architecture = config.architecture.model_copy(
        update={"geometry_source": geometry_source})
    modified_config = config.model_copy(update={"architecture": architecture})
    metadata = _m3d_backend(modified_config).metadata
    baseline = _m3d_backend(config).metadata
    assert metadata["m3d_layers"] == layers
    assert metadata["layer_pitch_um"] == pytest.approx(pitch_nm * 1e-3)
    assert metadata["miv_average_length_um"] == pytest.approx(
        expected_average_um)
    assert metadata["dies_stacked"] == 8
    expected_segments = (layers + 1) / 2
    assert metadata["miv_average_segments"] == pytest.approx(expected_segments)
    assert metadata["miv_average_effective_capacitance_pF"] == pytest.approx(
        metadata["miv_fixed_load_pF"]
        + metadata["miv_vertical_capacitance_pF_per_um"]
        * expected_average_um)
    assert metadata["miv_access_energy_pJ_per_bit"] == pytest.approx(
        baseline["miv_access_energy_pJ_per_bit"]
        * metadata["miv_average_effective_capacitance_pF"]
        / baseline["miv_average_effective_capacitance_pF"])


def test_miv_count_uses_dreamram_tsv_equivalent_serialization():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    metadata = _m3d_backend(config).metadata
    assert metadata["data_width_before_vertical"] == 256
    assert metadata["vertical_serialization_factor"] == 4
    assert metadata["vertical_serialization_status"] == "resolved"
    assert metadata["miv_serialization_factor"] == 4
    assert metadata["miv_serialization_source"] == (
        "DREAMRAM_TSV_EQUIVALENT")
    assert metadata["active_data_miv_count"] == 64
    assert metadata["row_miv_count"] == 15
    assert metadata["col_miv_count"] == 18
    resolved = build_miv_topology(
        m3d_layers=8,
        layer_pitch_um=0.288,
        data_width_before_vertical=metadata["data_width_before_vertical"],
        vertical_serialization_factor=5,
        row_miv_count=metadata["row_miv_count"],
        col_miv_count=metadata["col_miv_count"],
    )
    assert resolved.active_data_miv_count == 52
    assert resolved.active_data_miv_count == (
        metadata["data_width_before_vertical"] + 5 - 1) // 5


def test_m3d_path_excludes_hbm_vertical_base_dq_and_tsv_area():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    metadata = _m3d_backend(config).metadata
    assert metadata["vertical_interconnect_type"] == "MIV"
    assert metadata["miv_connection_model"] == (
        "per_layer_local_selection_to_shared_vertical")
    assert metadata["direct_bitline_to_feol"] is False
    assert metadata["miv_components"] == (
        "row-miv", "col-miv", "data-miv")
    assert metadata["tsv_energy_included"] is False
    assert metadata["base_route_included"] is False
    assert metadata["dq_included"] is False
    assert metadata["miv_dedicated_koz_area_modeled"] is False
    assert metadata["miv_planar_footprint_basis"] == (
        "tang_subarray_cluster")
    assert set(metadata["excluded_hbm_components"]) == {
        "row-tsv", "col-tsv", "tsv",
        "row-base", "col-base", "base",
        "row-dq", "col-dq", "dq",
    }


def test_length_scaled_miv_energy_resolves_and_access_closes():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    metadata = _m3d_backend(config).metadata
    assert metadata["miv_electrical_model"] == (
        "DREAMRAM_TSV_LENGTH_SCALED_REFERENCE")
    assert metadata["miv_modeling_class"] == "MODELING_CHOICE"
    assert metadata["miv_capacitance_status"] == "resolved"
    assert metadata["miv_energy_status"] == "resolved"
    expected_slope = (
        metadata["dreamram_reference_tsv_capacitance_pF"]
        / metadata["dreamram_reference_tsv_height_um"]
        * metadata["dreamram_tsv_pitch_capacitance_scale"])
    assert metadata["miv_vertical_capacitance_pF_per_um"] == pytest.approx(
        expected_slope)
    assert metadata["miv_vertical_capacitance_pF_per_um"] == pytest.approx(
        0.022)
    assert metadata["miv_vertical_capacitance_fF_per_um"] == pytest.approx(22)
    assert metadata["miv_fixed_load_pF"] == pytest.approx(0.006)
    assert metadata["miv_fixed_load_fF"] == pytest.approx(6)
    assert metadata["miv_fixed_load_classification"] == "MODELING_CHOICE"
    assert metadata["dreamram_reference_load_capacitance_pF"] == pytest.approx(
        0.120)
    assert metadata["dreamram_complete_scaled_tsv_capacitance_pF"] == (
        pytest.approx(0.78))
    assert metadata["dreamram_scaled_cap_tsv_used_per_m3d_segment"] is False
    assert metadata["dreamram_reference_load_used_as_miv_fixed_load"] is False
    assert metadata["miv_segments_energy_role"] == "GEOMETRY_DIAGNOSTIC_ONLY"
    assert metadata["row_miv_energy_pj_per_bit"] == pytest.approx(
        metadata["row_miv_access_energy_pJ_per_bit"])
    assert metadata["col_miv_energy_pj_per_bit"] == pytest.approx(
        metadata["col_miv_access_energy_pJ_per_bit"])
    assert metadata["data_miv_energy_pj_per_bit"] == pytest.approx(
        metadata["data_miv_access_energy_pJ_per_bit"])
    assert metadata["miv_length_per_layer_um"] == pytest.approx(
        tuple(0.288 * index for index in range(1, 9)))
    distributed = metadata["miv_distributed_capacitance_per_layer_pF"]
    effective = metadata["miv_effective_capacitance_per_layer_pF"]
    assert all(right > left for left, right in zip(distributed, distributed[1:]))
    assert all(
        total - partial == pytest.approx(0.006)
        for total, partial in zip(effective, distributed, strict=True))
    assert metadata["miv_average_distributed_capacitance_pF"] == pytest.approx(
        0.022 * 1.296)
    assert metadata["miv_average_effective_capacitance_pF"] == pytest.approx(
        0.006 + 0.022 * 1.296)
    assert metadata["miv_segments_per_layer"] == tuple(range(1, 9))
    assert metadata["miv_average_segments"] == pytest.approx(4.5)
    assert metadata["row_miv_voltage_source"] == (
        "DREAMRAM_TSV_EQUIVALENT")
    assert metadata["col_miv_voltage_source"] == (
        "DREAMRAM_TSV_EQUIVALENT")
    assert metadata["data_miv_voltage_source"] == (
        "DREAMRAM_TSV_EQUIVALENT")

    result = calculate_memory_power(config, project_root=ROOT)
    mat_local = result.diagnostics["replacement_components_pj_bit"][
        "zhu_scaled_local_operation"]
    routing = result.diagnostics["replacement_components_pj_bit"][
        "tang_global_control_routing"]
    assert mat_local == pytest.approx(0.1843)
    assert result.diagnostics["native_components_pj_bit"] == {}
    assert result.E_base_route_pj_bit == 0.0
    assert result.E_interface_pj_bit == pytest.approx(0.5)
    assert result.E_access_total_pj_bit == pytest.approx(
        mat_local + routing + result.E_vertical_pj_bit
        + result.E_feol_route_pj_bit + result.E_interface_pj_bit)


def test_layer_probability_weights_physical_miv_capacitance_and_energy():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    near_workload = config.workload.model_copy(
        update={"layer_access_probability": (1.0, 0, 0, 0, 0, 0, 0, 0)})
    far_workload = config.workload.model_copy(
        update={"layer_access_probability": (0, 0, 0, 0, 0, 0, 0, 1.0)})
    near = _m3d_backend(config.model_copy(update={"workload": near_workload}))
    far = _m3d_backend(config.model_copy(update={"workload": far_workload}))
    assert near.metadata["miv_average_length_um"] == pytest.approx(0.288)
    assert far.metadata["miv_average_length_um"] == pytest.approx(2.304)
    assert far.metadata["miv_average_effective_capacitance_pF"] > (
        near.metadata["miv_average_effective_capacitance_pF"])
    assert far.metadata["miv_access_energy_pJ_per_bit"] > (
        near.metadata["miv_access_energy_pJ_per_bit"])


def test_feol_route_config_and_centered_edge_ports():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    spec = config.architecture.feol_route
    assert spec is not None
    topology = _m3d_subarray(config)
    route = calculate_feol_route(spec, topology)
    assert route.feol_io_channel_count == 50
    assert route.feol_io_channel_pitch_um == pytest.approx(
        topology.slab_x_um / 50)
    assert route.feol_io_channel_pitch_um == pytest.approx(440.0)
    ports = route.feol_io_channel_coordinates_um
    assert len(ports) == 50
    assert ports[0] == pytest.approx((220.0, 0.0))
    assert ports[-1] == pytest.approx((21780.0, 0.0))
    assert all(
        right[0] - left[0] == pytest.approx(route.feol_io_channel_pitch_um)
        for left, right in zip(ports, ports[1:]))
    raw = config.model_dump()
    raw["architecture"]["feol_route"]["io_channels"] = 0
    with pytest.raises(ValueError, match="greater than 0"):
        MemoryPowerConfig.model_validate(raw)


def test_feol_cluster_to_port_mapping_is_nearest_manhattan():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    topology = _m3d_subarray(config)
    route = calculate_feol_route(config.architecture.feol_route, topology)
    assert route.feol_route_cluster_count == topology.clusters_per_layer
    assert len(route.feol_route_nearest_port_index) == topology.clusters_per_layer
    for index, ((xc, yc), port_index, length) in enumerate(zip(
            route.feol_route_cluster_centers_um,
            route.feol_route_nearest_port_index,
            route.feol_route_length_per_cluster_um,
            strict=True)):
        xp, yp = route.feol_io_channel_coordinates_um[port_index]
        assert length == pytest.approx(abs(xc - xp) + abs(yc - yp))
        assert length == pytest.approx(
            route.feol_route_lateral_component_per_cluster_um[index]
            + route.feol_route_perpendicular_component_per_cluster_um[index])
    assert route.feol_route_average_length_um > 1000.0
    assert route.feol_route_average_length_um < 10000.0
    # Same X column: farther cluster rows have a larger y_min route.
    first_column = route.feol_route_length_per_cluster_um[
        ::topology.cluster_count_x]
    assert all(right > left for left, right in zip(first_column, first_column[1:]))


def test_feol_io_count_and_slab_depth_control_physical_route():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    topology = _m3d_subarray(config)
    spec = config.architecture.feol_route
    baseline = calculate_feol_route(spec, topology)
    more_ports = spec.model_copy(update={"io_channels": 100})
    denser = calculate_feol_route(more_ports, topology)
    assert denser.feol_route_average_lateral_component_um <= (
        baseline.feol_route_average_lateral_component_um)

    geometry = load_m3d_geometry(ROOT, config.architecture.geometry_source)
    deeper_geometry = replace(geometry, slab_y_um=7000.0)
    deeper_topology = calculate_m3d_subarray(
        config.architecture.m3d_subarray, deeper_geometry)
    deeper = calculate_feol_route(spec, deeper_topology)
    assert deeper.feol_route_average_length_um > baseline.feol_route_average_length_um


@pytest.mark.parametrize(
    "wire_update, expected_factor",
    [
        ({"capacitance_fF_per_um": 0.40}, 2.0),
        ({"voltage_V": 1.60}, 4.0),
        ({"activity_factor": 1.0}, 2.0),
    ],
)
def test_feol_wire_energy_scaling(wire_update, expected_factor):
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    topology = _m3d_subarray(config)
    spec = config.architecture.feol_route
    baseline = calculate_feol_route(spec, topology)
    wire = spec.wire.model_copy(update=wire_update)
    modified = calculate_feol_route(spec.model_copy(update={"wire": wire}), topology)
    assert modified.feol_route_energy_pj_per_bit == pytest.approx(
        expected_factor * baseline.feol_route_energy_pj_per_bit)


def test_feol_boundary_is_additive_and_frozen_terms_are_unchanged():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    with_route = calculate_memory_power(config, project_root=ROOT)
    architecture = config.architecture.model_copy(update={"feol_route": None})
    without_route = calculate_memory_power(
        config.model_copy(update={"architecture": architecture}),
        project_root=ROOT)
    assert with_route.E_access_total_pj_bit == pytest.approx(
        without_route.E_access_total_pj_bit + with_route.E_feol_route_pj_bit)
    assert with_route.E_memory_internal_pj_bit == pytest.approx(
        without_route.E_memory_internal_pj_bit, abs=0.0)
    assert with_route.E_vertical_pj_bit == pytest.approx(
        without_route.E_vertical_pj_bit, abs=0.0)
    assert with_route.E_base_route_pj_bit == 0.0
    assert with_route.E_interface_pj_bit == pytest.approx(0.5, abs=0.0)
    assert with_route.diagnostics["miv_fixed_load_pF"] == pytest.approx(0.006)
    assert with_route.diagnostics["feol_route_start"] == "MIV_FEOL_LANDING"
    assert with_route.diagnostics["feol_route_end"] == "EDGE_IO_INTERFACE_INPUT"
    assert with_route.diagnostics["interface_boundary"] == (
        "EDGE_IO_INTERFACE_INPUT_TO_GPU_RECEIVER")
    assert with_route.diagnostics["interface_energy_pj_per_bit"] == 0.5
    assert with_route.diagnostics["feol_serialization_applied"] is False

    conventional = calculate_memory_power(
        load_power_config(POWER_CONFIGS / "hbm3_si.yaml"), project_root=ROOT)
    assert conventional.E_feol_route_pj_bit == 0.0
    assert conventional.E_vertical_pj_bit == pytest.approx(0.22048568115234377)
    assert conventional.E_access_total_pj_bit == 0.9782367130708566


def _with_m3d_subarray_size(config, *, n_rows=None, n_cols=None):
    core = config.architecture.m3d_subarray.subarray
    update = {}
    if n_rows is not None:
        update["n_rows"] = n_rows
    if n_cols is not None:
        update["n_cols"] = n_cols
    resized_core = core.model_copy(update=update)
    topology = config.architecture.m3d_subarray.model_copy(
        update={"subarray": resized_core})
    architecture = config.architecture.model_copy(
        update={"m3d_subarray": topology})
    return config.model_copy(update={"architecture": architecture})


def test_zhu_nrow_scaling_preserves_reference_and_formula():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    results = {
        rows: calculate_memory_power(
            _with_m3d_subarray_size(config, n_rows=rows), project_root=ROOT)
        for rows in (256, 512, 1024)
    }
    reference = results[512]
    assert reference.diagnostics["zhu_reference_n_rows"] == 512
    assert reference.diagnostics["zhu_reference_n_cols"] == 512
    assert reference.diagnostics["zhu_reference_read_0_pj_per_bit"] == 0.00060
    assert reference.diagnostics["zhu_reference_read_1_pj_per_bit"] == 0.36800
    assert reference.diagnostics["zhu_reference_energy_provenance"] == (
        "PAPER_REPORTED")
    assert reference.diagnostics["zhu_size_scaling_model"] == (
        "common_rc_linear_nrow")
    assert reference.diagnostics["zhu_size_scaling_provenance"] == (
        "MODELING_CHOICE")
    assert reference.diagnostics["zhu_scaled_weighted_read_pj_per_bit"] == (
        pytest.approx(0.1843, abs=0.0))
    assert (
        results[256].diagnostics["zhu_scaled_weighted_read_pj_per_bit"]
        < reference.diagnostics["zhu_scaled_weighted_read_pj_per_bit"]
        < results[1024].diagnostics["zhu_scaled_weighted_read_pj_per_bit"])
    for rows, result in results.items():
        ratio = rows / 512
        scaled_read_0 = 0.00060 * ratio
        scaled_read_1 = (0.36800 - 0.00060) + scaled_read_0
        assert result.diagnostics["zhu_nrow_scale_ratio"] == pytest.approx(ratio)
        assert result.diagnostics["zhu_scaled_read_0_pj_per_bit"] == (
            pytest.approx(scaled_read_0))
        assert result.diagnostics["zhu_scaled_read_1_pj_per_bit"] == (
            pytest.approx(scaled_read_1))
        assert result.diagnostics["zhu_scaled_weighted_read_pj_per_bit"] == (
            pytest.approx(0.5 * scaled_read_0 + 0.5 * scaled_read_1))


def test_zhu_v1_scaling_ignores_ncol_and_local_energy_is_zero():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    baseline = calculate_memory_power(config, project_root=ROOT)
    wider = calculate_memory_power(
        _with_m3d_subarray_size(config, n_cols=1024), project_root=ROOT)
    assert wider.diagnostics["zhu_scaled_weighted_read_pj_per_bit"] == (
        pytest.approx(
            baseline.diagnostics["zhu_scaled_weighted_read_pj_per_bit"],
            abs=0.0))
    assert baseline.diagnostics["local_mux_footprint_height_um"] > 0.0
    assert baseline.diagnostics["local_mux_geometry_modeled"] is True
    assert baseline.diagnostics["local_mux_energy_modeled"] is False
    assert baseline.diagnostics["local_mux_energy_status"] == (
        "NOT_SEPARATELY_MODELED")
    assert baseline.diagnostics["local_rbl_separate_energy_modeled"] is False
    assert baseline.diagnostics["local_rbl_energy_pj_per_bit"] == 0.0
    assert baseline.diagnostics["local_mux_energy_pj_per_bit"] == 0.0
    assert baseline.diagnostics["local_read_routing_energy_pj_per_bit"] == 0.0
    assert "tang_local_read_routing" not in (
        baseline.diagnostics["replacement_components_pj_bit"])
    assert baseline.E_memory_internal_pj_bit == pytest.approx(
        baseline.diagnostics["zhu_scaled_weighted_read_pj_per_bit"]
        + baseline.diagnostics["global_control_routing_energy_pj_per_bit"])


def test_local_energy_removal_preserves_frozen_transports_and_geometry():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    result = calculate_memory_power(config, project_root=ROOT)
    topology = _m3d_subarray(config)
    assert result.diagnostics["cluster_width_um"] == pytest.approx(
        topology.cluster_width_um, abs=0.0)
    assert result.diagnostics["cluster_height_um"] == pytest.approx(
        topology.cluster_height_um, abs=0.0)
    assert result.diagnostics["global_control_routing_energy_pj_per_bit"] == (
        pytest.approx(topology.global_control_routing_energy_pj_per_bit, abs=0.0))
    assert result.E_vertical_pj_bit == pytest.approx(
        0.002445862111816407, abs=0.0)
    assert result.E_interface_pj_bit == pytest.approx(0.5, abs=0.0)
    assert result.diagnostics["miv_fixed_load_pF"] == pytest.approx(0.006)


def test_igzo_refresh_operation_table_and_capacity_accounting():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    result = calculate_memory_power(config, project_root=ROOT)
    diagnostics = result.diagnostics
    assert diagnostics["refresh_reference_0_pj_per_bit"] == 0.00090
    assert diagnostics["refresh_reference_1_pj_per_bit"] == 0.37000
    assert diagnostics["refresh_weighted_energy_pj_per_bit"] == pytest.approx(
        0.18545, abs=0.0)
    assert diagnostics["zhu_refresh_size_scaling"] == "NOT_MODELED"
    assert diagnostics["retention_reference_s"] == 20.0
    assert diagnostics["retention_reference_source"] == (
        "TANG_IEDM2023_IGZO_2T0C")
    assert diagnostics["retention_provenance"] == "PAPER_REPORTED"
    assert diagnostics["refresh_interval_provenance"] == "MODELING_CHOICE"
    assert diagnostics["total_stored_bits"] == (
        diagnostics["bits_per_layer"] * diagnostics["memory_layer_count"])
    expected_energy_J = (
        diagnostics["total_stored_bits"] * 0.18545e-12)
    assert diagnostics["full_memory_refresh_energy_J"] == pytest.approx(
        expected_energy_J)
    assert result.P_refresh_W == pytest.approx(expected_energy_J / 20.0)
    assert diagnostics["refresh_route_boundary"] == "INTERNAL_MEMORY_ONLY"
    assert "dreamram_refresh_included_components" not in diagnostics


def test_igzo_refresh_scales_with_capacity_interval_and_safety_factor():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    topology = _m3d_subarray(config)
    backend = _m3d_backend(config)
    device = OperationTableCellModel().calculate(config)
    baseline = calculate_refresh_power(
        config, backend=backend, device=device, m3d_subarray=topology,
        m3d_layer_count=8)
    twice_capacity = calculate_refresh_power(
        config, backend=backend, device=device, m3d_subarray=topology,
        m3d_layer_count=16)
    assert twice_capacity.power_W == pytest.approx(2.0 * baseline.power_W)

    refresh = config.power.refresh.model_copy(
        update={"retention_reference_s": 40.0})
    slower = calculate_refresh_power(
        config.model_copy(update={
            "power": config.power.model_copy(update={"refresh": refresh})}),
        backend=backend, device=device, m3d_subarray=topology,
        m3d_layer_count=8)
    assert slower.power_W == pytest.approx(0.5 * baseline.power_W)

    refresh = config.power.refresh.model_copy(
        update={"refresh_safety_factor": 2.0})
    safer = calculate_refresh_power(
        config.model_copy(update={
            "power": config.power.model_copy(update={"refresh": refresh})}),
        backend=backend, device=device, m3d_subarray=topology,
        m3d_layer_count=8)
    assert safer.power_W == pytest.approx(2.0 * baseline.power_W)


def test_si_refresh_uses_internal_act_pre_components_and_organization():
    config = load_power_config(POWER_CONFIGS / "hbm3_si.yaml")
    result = calculate_memory_power(config, project_root=ROOT)
    diagnostics = result.diagnostics
    included = diagnostics["dreamram_refresh_included_components"]
    excluded = diagnostics["dreamram_refresh_excluded_components"]
    assert included == ["row", "mwl", "lwl", "bl-pre", "bl-act"]
    assert {"tsv", "base", "dq", "bgbus+gbus"}.issubset(excluded)
    organization = diagnostics["dreamram_refresh_organization"]
    expected_events = (
        organization["ranks"]
        * organization["channels"]
        * organization["pseudochannels"]
        * organization["horizontal_bankgroups"]
        * organization["vertical_bankgroups"]
        * organization["banks_per_bankgroup"]
        * organization["subarrays_per_bank"]
        * organization["rows_per_mat"]
        * organization["independent_row_pages"])
    assert diagnostics["refresh_events_per_full_memory_cycle"] == (
        expected_events)
    assert diagnostics["total_stored_bits"] == (
        expected_events * diagnostics["refresh_bits_per_event"])
    expected_full_energy = (
        diagnostics["refresh_internal_event_energy_pJ"] * expected_events)
    assert diagnostics["full_memory_refresh_energy_pJ"] == pytest.approx(
        expected_full_energy)
    assert result.P_refresh_W == pytest.approx(
        expected_full_energy * 1e-12 / 0.032)
    assert result.P_refresh_W > 0.0


@pytest.mark.parametrize("name", ["hbm3_si.yaml", "orthogonal_m3d_igzo.yaml"])
def test_refresh_is_independent_of_read_bandwidth(name):
    config = load_power_config(POWER_CONFIGS / name)
    baseline = calculate_memory_power(config, project_root=ROOT)
    workload = config.workload.model_copy(update={
        "read_bandwidth_gbps": config.workload.read_bandwidth_gbps / 2.0})
    slower = calculate_memory_power(
        config.model_copy(update={"workload": workload}), project_root=ROOT)
    assert slower.P_read_W == pytest.approx(0.5 * baseline.P_read_W)
    assert slower.P_refresh_W == pytest.approx(baseline.P_refresh_W, abs=0.0)


def test_si_refresh_window_scaling_and_read_regression():
    config = load_power_config(POWER_CONFIGS / "hbm3_si.yaml")
    baseline = calculate_memory_power(config, project_root=ROOT)
    refresh = config.power.refresh.model_copy(update={"refresh_window_s": 0.064})
    modified = calculate_memory_power(
        config.model_copy(update={
            "power": config.power.model_copy(update={"refresh": refresh})}),
        project_root=ROOT)
    assert modified.P_refresh_W == pytest.approx(0.5 * baseline.P_refresh_W)
    assert modified.E_access_total_pj_bit == 0.9782367130708566
    assert baseline.E_access_total_pj_bit == 0.9782367130708566


def test_refresh_disable_is_zero_and_read_paths_are_frozen():
    for name, expected_access in (
            ("hbm3_si.yaml", 0.9782367130708566),
            ("orthogonal_m3d_igzo.yaml", 0.8552605756733209)):
        config = load_power_config(POWER_CONFIGS / name)
        raw = config.model_dump(mode="json")
        raw["power"]["refresh"] = {"enabled": False}
        disabled = MemoryPowerConfig.model_validate(raw)
        result = calculate_memory_power(disabled, project_root=ROOT)
        assert result.P_refresh_W == 0.0
        assert result.E_access_total_pj_bit == pytest.approx(
            expected_access, abs=0.0)
        assert result.diagnostics["refresh_enabled"] is False


def test_refresh_does_not_change_m3d_read_transport_terms():
    config = load_power_config(POWER_CONFIGS / "orthogonal_m3d_igzo.yaml")
    result = calculate_memory_power(config, project_root=ROOT)
    assert result.E_access_total_pj_bit == pytest.approx(
        0.8552605756733209, abs=0.0)
    assert result.E_vertical_pj_bit == pytest.approx(
        0.002445862111816407, abs=0.0)
    assert result.E_feol_route_pj_bit == pytest.approx(
        0.16705631334524151, abs=0.0)
    assert result.E_interface_pj_bit == pytest.approx(0.5, abs=0.0)


def test_canonical_cases_parse_and_preserve_nominal_power():
    hbm_case = load_case_config(CASE_CONFIGS / "conventional_hbm3.yaml")
    hbm_geometry = resolve_case_geometry(hbm_case)
    hbm = calculate_memory_power(
        hbm_case, project_root=ROOT, geometry=hbm_geometry)
    assert hbm.E_access_total_pj_bit == 0.9782367130708566
    assert hbm.P_refresh_W == 0.11395159240799647
    assert hbm.diagnostics["geometry_source_config"] == (
        "canonical_case:conventional_hbm3")
    assert hbm_case.architecture.geometry_source is None
    assert hbm_case.architecture.layers is None
    assert hbm_case.provenance["analytical_stack_dies"] == 8
    assert hbm_case.provenance["legacy_thermal_stack_dies"] == 12
    assert hbm.diagnostics["dies_stacked"] == 8
    assert hbm.diagnostics["geometry_feasible"] is True
    assert hbm_case.provenance["memory_region_footprint"] == (
        "DERIVED_FROM_DREAMRAM_DATE2026")
    assert hbm_case.thermal["migration_status"] == "NOT_MIGRATED"

    m3d_case = load_case_config(CASE_CONFIGS / "orthogonal_m3d_igzo.yaml")
    m3d_geometry = resolve_case_geometry(m3d_case)
    m3d = calculate_memory_power(
        m3d_case, project_root=ROOT, geometry=m3d_geometry)
    assert m3d.E_access_total_pj_bit == 0.8552605756733209
    assert m3d.E_vertical_pj_bit == 0.002445862111816407
    assert m3d.E_feol_route_pj_bit == 0.16705631334524151
    assert m3d.E_interface_pj_bit == 0.5
    assert m3d.P_refresh_W == 0.0003484694872064
    assert m3d.diagnostics["geometry_source_config"] == (
        "canonical_case:orthogonal_m3d_igzo")


def test_canonical_m3d_has_single_geometry_and_operation_sources():
    path = CASE_CONFIGS / "orthogonal_m3d_igzo.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "geometry_source" not in raw["architecture"]
    assert "memory_region" not in raw["geometry"]
    assert "total_pitch_um" not in raw["geometry"]["m3d_stack"]
    assert "layers" not in raw["architecture"]
    assert "stored_bits" not in raw["workload"]
    assert "operations" not in raw["geometry"]
    assert "power_models" not in raw["geometry"]
    assert "bitcell_layers" not in raw["memory"]
    assert "bitcell_layer_pitch_nm" not in raw["memory"]
    assert set(raw["memory"]["cell_model"]["operations"]) == {
        "read_0_pj_per_bit", "read_1_pj_per_bit",
        "write_00_pj_per_bit", "write_01_pj_per_bit",
        "write_10_pj_per_bit", "write_11_pj_per_bit",
        "refresh_0_pj_per_bit", "refresh_1_pj_per_bit",
    }


def test_canonical_geometry_drives_capacity_and_miv_without_second_yaml():
    case = load_case_config(CASE_CONFIGS / "orthogonal_m3d_igzo.yaml")
    geometry = resolve_case_geometry(case)
    baseline = calculate_memory_power(case, project_root=ROOT, geometry=geometry)
    assert baseline.diagnostics["clusters_per_layer"] == 280
    assert baseline.diagnostics["subarrays_per_layer"] == 17920
    assert baseline.diagnostics["bits_per_layer"] == 4697620480
    assert baseline.diagnostics["total_stored_bits"] == 37580963840
    assert baseline.diagnostics["placed_width_um"] == pytest.approx(
        21794.548876360117)
    assert baseline.diagnostics["placed_height_um"] == pytest.approx(
        4999.693110067114)

    raw = case.model_dump(mode="json")
    raw["geometry"]["m3d_stack"]["bitcell_layers"] = 16
    raw["geometry"]["m3d_stack"]["si_substrate_um"] = 290.242
    doubled_case = type(case).model_validate(raw)
    doubled_geometry = resolve_case_geometry(doubled_case)
    doubled = calculate_memory_power(
        doubled_case, project_root=ROOT, geometry=doubled_geometry)
    assert doubled.diagnostics["total_stored_bits"] == (
        2 * baseline.diagnostics["total_stored_bits"])
    assert doubled.diagnostics["miv_average_length_um"] != (
        baseline.diagnostics["miv_average_length_um"])
    assert doubled_case.memory.cell_model == case.memory.cell_model

    raw = case.model_dump(mode="json")
    raw["geometry"]["m3d_stack"]["bitcell_layer_pitch_nm"] = 300.0
    raw["geometry"]["m3d_stack"]["si_substrate_um"] = 292.45
    wider_pitch_case = type(case).model_validate(raw)
    wider_geometry = resolve_case_geometry(wider_pitch_case)
    wider = calculate_memory_power(
        wider_pitch_case, project_root=ROOT, geometry=wider_geometry)
    assert wider.diagnostics["miv_average_length_um"] != (
        baseline.diagnostics["miv_average_length_um"])
    assert wider.E_vertical_pj_bit != baseline.E_vertical_pj_bit


def test_run_memory_power_accepts_one_canonical_case_path():
    case = load_case_config(CASE_CONFIGS / "orthogonal_m3d_igzo.yaml")
    geometry = resolve_case_geometry(case)
    assert case.thermal["geometry_source"] == "canonical_case_geometry"
    assert geometry.source == "canonical_case:orthogonal_m3d_igzo"
    result = run_memory_power(CASE_CONFIGS / "orthogonal_m3d_igzo.yaml")
    assert result.E_access_total_pj_bit == 0.8552605756733209
