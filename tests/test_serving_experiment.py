"""Formal three-architecture serving path without thermal execution."""

from pathlib import Path

import om3dthermal.case_runner as case_runner
import om3dthermal.thermal.gpu_pcg as gpu_pcg
import pytest

from om3dthermal.experiment import (
    run_serving_experiment,
    write_serving_experiment_csvs,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "experiment" / "capacity_aware_serving_v0.yaml"


def test_formal_serving_path_reads_current_architecture_capacity_and_skips_thermal(
    monkeypatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("serving C path must not invoke thermal")

    monkeypatch.setattr(case_runner, "run_steady_pipeline", forbidden)
    monkeypatch.setattr(gpu_pcg, "solve_pcg_gpu", forbidden)
    result = run_serving_experiment(CONFIG, project_root=ROOT)
    by_arch = {item.architecture: item for item in result.operating_points}
    assert set(by_arch) == {
        "conventional_hbm_2x1",
        "orthogonal_si",
        "orthogonal_m3d_igzo",
        "modern_high_capacity_hbm",
    }
    expected_max = {
        "conventional_hbm_2x1": 5,
        "orthogonal_si": 13,
        "orthogonal_m3d_igzo": 25,
        "modern_high_capacity_hbm": 15,
    }
    for architecture, maximum in expected_max.items():
        assert {row.max_resident_requests for row in by_arch[architecture].rows} == {
            maximum}
    assert by_arch["conventional_hbm_2x1"].optimal_requested_requests == 4
    assert by_arch["orthogonal_si"].optimal_requested_requests == 8
    assert by_arch["orthogonal_m3d_igzo"].optimal_requested_requests == 16
    assert by_arch["modern_high_capacity_hbm"].optimal_requested_requests == 8

    hbm_8 = next(
        row for row in by_arch["conventional_hbm_2x1"].rows
        if row.requested_requests == 8)
    m3d_8 = next(
        row for row in by_arch["orthogonal_m3d_igzo"].rows
        if row.requested_requests == 8)
    assert hbm_8.capacity_status == "CAPACITY_PRESSURED"
    assert hbm_8.evaluation_status == "EVALUATED"
    assert hbm_8.host_effective_bandwidth_bytes_per_second == 56.2e9
    assert hbm_8.host_transfer_time_ms == pytest.approx(
        hbm_8.host_transfer_bytes_per_step / 56.2e9 * 1e3)
    assert m3d_8.capacity_status == "FULLY_LOCAL"
    assert m3d_8.evaluation_status == "EVALUATED"
    assert hbm_8.aggregate_tokens_per_s < m3d_8.aggregate_tokens_per_s
    modern_8 = next(
        row for row in by_arch["modern_high_capacity_hbm"].rows
        if row.requested_requests == 8)
    modern_16 = next(
        row for row in by_arch["modern_high_capacity_hbm"].rows
        if row.requested_requests == 16)
    m3d_16 = next(
        row for row in by_arch["orthogonal_m3d_igzo"].rows
        if row.requested_requests == 16)
    assert modern_8.capacity_status == "FULLY_LOCAL"
    assert modern_16.spilled_requests == 1
    assert m3d_16.spilled_requests == 0
    assert result.unresolved_capacity_references == ()


def test_formal_sensitivity_is_monotonic_and_all_local_points_match() -> None:
    result = run_serving_experiment(CONFIG, project_root=ROOT)
    nominal = {item.architecture: item for item in result.operating_points}
    hbm_4 = next(row for row in nominal["conventional_hbm_2x1"].rows
                 if row.requested_requests == 4)
    m3d_4 = next(row for row in nominal["orthogonal_m3d_igzo"].rows
                 if row.requested_requests == 4)
    assert hbm_4.aggregate_tokens_per_s == m3d_4.aggregate_tokens_per_s

    sensitivity = {
        item.sensitivity_id: {point.architecture: point
                              for point in item.operating_points}
        for item in result.sensitivity_operating_points
    }
    low = next(row for row in sensitivity[
        "published_h100_h2d_51_32_no_overlap"]["conventional_hbm_2x1"].rows
               if row.requested_requests == 8)
    high = next(row for row in sensitivity[
        "pcie5_one_direction_upper_no_overlap"]["conventional_hbm_2x1"].rows
                if row.requested_requests == 8)
    nominal_8 = next(row for row in nominal["conventional_hbm_2x1"].rows
                     if row.requested_requests == 8)
    assert low.aggregate_tokens_per_s <= nominal_8.aggregate_tokens_per_s
    assert nominal_8.aggregate_tokens_per_s <= high.aggregate_tokens_per_s


def test_plot_ready_csv_export(tmp_path) -> None:
    result = run_serving_experiment(CONFIG, project_root=ROOT)
    experiment = result.experiment.model_copy(update={"output_dir": tmp_path / "c"})
    result = result.__class__(
        experiment=experiment,
        operating_points=result.operating_points,
        sensitivity_operating_points=result.sensitivity_operating_points,
        unresolved_capacity_references=result.unresolved_capacity_references,
    )
    nominal_path, sensitivity_path = write_serving_experiment_csvs(result)
    nominal = nominal_path.read_text(encoding="utf-8")
    sensitivity = sensitivity_path.read_text(encoding="utf-8")
    assert "host_total_GB_per_step" in nominal
    assert "host_effective_bandwidth_GBps" in nominal
    assert "modern_high_capacity_hbm" in nominal
    assert "nominal_h2d_full_overlap_upper" in sensitivity
