"""Targeted tests for the frozen-A per-die thermal carrier mapping."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from om3dthermal.geometry.orthogonal_hbm import OrthogonalHBMBuilder
from om3dthermal.thermal.nmp_die_mapping import (
    _pearson, analyze_nmp_die_thermal_pipeline,
    compile_nmp_die_thermal_config, physical_nmp_die_regions)
from scripts.evaluate_nmp_thermal_baseline import _frozen_case_inputs


@pytest.fixture(scope="module")
def frozen_n8():
    return _frozen_case_inputs(8)


def test_die_identity_is_complete_deterministic_and_unique(frozen_n8):
    case, _, power_map, _, _ = frozen_n8
    first = physical_nmp_die_regions(case)
    second = physical_nmp_die_regions(case)
    assert first == second
    assert len(first) == len(power_map.die_powers) == 98
    assert [row.die_id for row in first] == list(range(98))
    assert [row.geometry_die_index for row in first] == list(range(1, 99))
    assert len({row.region_id for row in first}) == 98
    assert first[0].region_id == "orthogonal_hbm:die_001"
    assert first[-1].region_id == "orthogonal_hbm:die_098"


def test_each_identity_resolves_one_memory_and_one_feol_box(frozen_n8):
    case, system, power_map, _, _ = frozen_n8
    config, regions = compile_nmp_die_thermal_config(case, system, power_map)
    scene = OrthogonalHBMBuilder(config).build()
    for region in regions:
        boxes = scene.filter(component=region.region_id)
        assert len([box for box in boxes if box.tags.get("role") == region.memory_role]) == 1
        assert len([box for box in boxes if box.tags.get("role") == region.nmp_role]) == 1


def test_all_per_die_carriers_and_package_power_close(frozen_n8):
    case, system, power_map, gain, placement = frozen_n8
    config, regions = compile_nmp_die_thermal_config(case, system, power_map)
    sources = config.thermal_power_sources.sources
    assert len(regions) == 98
    assert len(sources) == 1 + 3 * 98
    assert system.gpu_power_W == pytest.approx(300.0)
    assert sources[0].total_power == pytest.approx(300.0)
    assert sum(x.thermal_memory_carrier_W for x in power_map.die_powers) == pytest.approx(
        power_map.aggregate_memory_read_dynamic_W
        + power_map.aggregate_memory_write_dynamic_W + power_map.aggregate_refresh_W)
    assert sum(x.thermal_nmp_carrier_W for x in power_map.die_powers) == pytest.approx(
        power_map.aggregate_mac_dynamic_W)
    assert sum(x.residual_external_W for x in power_map.die_powers) == pytest.approx(
        power_map.aggregate_residual_external_W)
    assert sum(source.total_power for source in sources[1:]) == pytest.approx(
        power_map.aggregate_total_W)
    assert sum(source.total_power for source in sources) == pytest.approx(
        300.0 + power_map.aggregate_total_W)
    assert power_map.aggregate_refresh_W == pytest.approx(power_map.refresh_total_W)
    assert all(row.nmp_logic_overhead_factor == 1.0 for row in power_map.die_powers)
    assert gain == pytest.approx(3.950, rel=2e-3)
    # This model exposes no direct die-to-die path: ownership is local and all
    # remaining bytes are explicitly attributed to the external boundary.
    assert placement.locality_constraint.startswith("LEXICOGRAPHIC_MINIMUM_DIE_SPAN")


def test_residual_external_mapping_is_explicit_coarse_feol(frozen_n8):
    case, system, power_map, _, _ = frozen_n8
    config, _ = compile_nmp_die_thermal_config(case, system, power_map)
    external = [source for source in config.thermal_power_sources.sources
                if source.metadata.get("component_class") == "residual_external"]
    assert len(external) == 98
    assert all(source.selector.tags == {"role": "feol"} for source in external)
    assert all(source.metadata["mapping_provenance"]
               == "RESIDUAL_EXTERNAL_THERMAL_MAPPING_APPROXIMATION"
               for source in external)
    assert sum(source.total_power for source in external) == pytest.approx(
        power_map.aggregate_residual_external_W)


def test_correlation_reports_undefined_only_for_zero_variance():
    assert _pearson([1.0, 1.0], [2.0, 3.0]) is None
    assert _pearson([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)
    assert math.isfinite(_pearson([1.0, 2.0, 4.0], [4.0, 1.0, 3.0]))


def test_baseline_observables_are_finite_and_have_valid_maxima(frozen_n8):
    case, _, power_map, _, _ = frozen_n8
    regions = physical_nmp_die_regions(case)
    cells = []
    temperatures = []
    for region in regions:
        for role, offset in ((region.memory_role, 0.0), (region.nmp_role, 0.02)):
            cells.append(SimpleNamespace(
                component=region.region_id, material="test",
                tags={"role": role}, center_x=region.center_x_m,
                center_y=0.0, center_z=0.0))
            temperatures.append(300.0 + 0.01 * region.die_id + offset)
    cells.append(SimpleNamespace(component="gpu", material="FEOL", tags={},
                                 center_x=0.0, center_y=0.0, center_z=0.0))
    temperatures.append(400.0)
    pipeline = SimpleNamespace(
        cells=cells,
        result=SimpleNamespace(temperature_K=np.array(temperatures), converged=True,
                               iterations=10, final_relative_residual=1e-9),
        power=SimpleNamespace(total_power_W=300.0 + power_map.aggregate_total_W))
    result = analyze_nmp_die_thermal_pipeline(
        requests=8, power_map=power_map, regions=regions, pipeline=pipeline,
        solver_backend="gpu_pcg")
    assert len(result.dies) == 98
    assert all(math.isfinite(row.die_temperature_degC) for row in result.dies)
    assert result.global_Tmax_degC >= 20.0
    assert result.hottest_m3d_die_id == 97
    assert 0 <= result.max_power_die_id < 98
    assert math.isfinite(result.power_temperature_correlation)
    assert result.thermal_power_mapping_closure == "PASS"
