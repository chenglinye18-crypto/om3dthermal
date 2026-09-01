"""Integration-only gates for the canonical frozen E2E runner."""
from __future__ import annotations

import math

import pytest

from om3dthermal.evaluator.canonical_e2e import (
    integrate_canonical_e2e, write_canonical_e2e_artifacts)
from scripts.evaluate_die_local_placement import _architecture
from scripts.evaluate_nmp_locality_placement import run as run_nmp


THERMAL = {
    1: (67.2205041644706, 68.9968672597728, 73.6042466206252,
        6.38374245615461, 83.8547853487576, 0, 0),
    8: (72.2062933880625, 72.4262528679868, 75.1013393983393,
        2.89504601027681, 85.2812342478641, 81, 97),
    16: (71.8779799257423, 72.2017334615123, 74.7191244802142,
         2.84114455447195, 84.9214507948154, 22, 97),
}


@pytest.fixture(scope="module")
def integrated(tmp_path_factory):
    nmp = run_nmp(tmp_path_factory.mktemp("canonical_nmp"))
    thermal_cases = []
    for row in nmp["rows"]:
        n = row["requests"]
        tmin, tmean, tmax, spread, global_t, power_die, hot_die = THERMAL[n]
        thermal_cases.append({
            "aggregate_m3d_power_W": row["B_PREP_DIE_POWER_MAP"]["aggregate_total_W"],
            "summary": {
                "requests": n, "temperature_min_degC": tmin,
                "temperature_mean_degC": tmean, "temperature_max_degC": tmax,
                "temperature_spread_degC": spread, "global_Tmax_degC": global_t,
                "global_Tmax_region": "gpu"},
            "baseline": {"converged": True,
                         "thermal_power_mapping_closure": "PASS",
                         "max_power_die_id": power_die,
                         "hottest_m3d_die_id": hot_die},
        })
    layout, _ = _architecture()
    return integrate_canonical_e2e(
        nmp_payload=nmp, thermal_payload={"cases": thermal_cases},
        total_capacity_bytes=layout.total_capacity_bytes)


def test_runner_integration_returns_only_canonical_points(integrated):
    assert [row.requests for row in integrated.cases] == [1, 8, 16]
    assert integrated.gates == {
        "E2E_CAPACITY_GATE": "PASS", "E2E_TRAFFIC_GATE": "PASS",
        "E2E_PERFORMANCE_GATE": "PASS", "E2E_POWER_GATE": "PASS",
        "E2E_THERMAL_GATE": "PASS", "E2E_CANONICAL_GATE": "CONDITIONAL_PASS"}


def test_capacity_residency_and_headroom_close(integrated):
    assert all(row.total_physical_capacity_bytes == 428.75 * 2**30
               for row in integrated.cases)
    assert all(row.logical_working_set_bytes <= row.allocated_working_set_bytes
               <= row.total_physical_capacity_bytes for row in integrated.cases)
    assert all(row.resident_fraction == 1.0 and row.capacity_feasible
               and row.capacity_headroom_bytes >= 0 for row in integrated.cases)


def test_traffic_boundary_and_reduction_close(integrated):
    for row in integrated.cases:
        assert row.local_nmp_bytes == pytest.approx(
            row.total_weight_read_bytes + row.total_kv_read_bytes
            + row.total_kv_write_bytes)
        assert 0 < row.residual_external_bytes < row.non_nmp_external_bytes
        assert row.external_traffic_reduction == pytest.approx(
            1 - row.residual_external_bytes / row.non_nmp_external_bytes)
        assert 0 <= row.external_traffic_reduction <= 1
        assert row.direct_die_to_die_bytes == 0


def test_all_four_performance_points_and_throughput_close(integrated):
    gains = {1: 2.56529347202469, 8: 3.95003584371152,
             16: 3.74409028630852}
    for row in integrated.cases:
        for step, throughput in (
                (row.non_nmp_step_ms, row.non_nmp_tokens_per_s),
                (row.locality_only_step_ms, row.locality_only_tokens_per_s),
                (row.balanced_step_ms, row.balanced_tokens_per_s),
                (row.ideal_step_ms, row.ideal_tokens_per_s)):
            assert throughput == pytest.approx(row.requests / (step * 1e-3))
        assert row.balanced_gain == pytest.approx(gains[row.requests])
        assert row.placement_incremental_speedup == pytest.approx(
            row.locality_only_step_ms / row.balanced_step_ms)
        assert row.balanced_step_ms >= row.ideal_step_ms


def test_power_energy_and_frozen_hardware_close(integrated):
    powers = {1: 32.4502005092269, 8: 60.5992178172216,
              16: 58.7575858482535}
    for row in integrated.cases:
        assert row.aggregate_m3d_nmp_power_W == pytest.approx(
            row.read_W + row.write_W + row.mac_W + row.refresh_W
            + row.residual_external_W)
        assert row.aggregate_m3d_nmp_power_W == pytest.approx(powers[row.requests])
        assert row.energy_per_decode_step_J == pytest.approx(
            row.aggregate_m3d_nmp_power_W * row.balanced_step_ms * 1e-3)
        assert row.energy_per_token_J == pytest.approx(
            row.energy_per_decode_step_J / row.requests)
        assert row.gamma_NMP == 1.0


def test_thermal_regression_and_all_metrics_finite(integrated):
    for row in integrated.cases:
        expected = THERMAL[row.requests]
        assert row.m3d_Tmin_degC == pytest.approx(expected[0])
        assert row.m3d_Tmean_degC == pytest.approx(expected[1])
        assert row.m3d_Tmax_degC == pytest.approx(expected[2])
        assert row.m3d_delta_T_degC == pytest.approx(expected[3])
        assert row.global_Tmax_degC == pytest.approx(expected[4])
        assert row.global_Tmax_region == "gpu"
        numeric = [value for value in row.as_dict().values()
                   if isinstance(value, (int, float))]
        assert all(math.isfinite(value) for value in numeric)


def test_artifacts_are_reproducible(integrated, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    write_canonical_e2e_artifacts(integrated, first)
    write_canonical_e2e_artifacts(integrated, second)
    expected = {"summary.json", "summary.csv", "architecture_comparison.csv",
                "placement_comparison.csv", "case_n1.json", "case_n8.json",
                "case_n16.json"}
    assert {path.name for path in first.iterdir()} == expected
    for name in expected:
        assert (first / name).read_bytes() == (second / name).read_bytes()
