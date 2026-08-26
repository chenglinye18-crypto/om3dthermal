"""Formal three-architecture serving path without thermal execution."""

from pathlib import Path

import om3dthermal.case_runner as case_runner
import om3dthermal.thermal.gpu_pcg as gpu_pcg

from om3dthermal.experiment import run_serving_experiment


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
    }
    expected_max = {
        "conventional_hbm_2x1": 5,
        "orthogonal_si": 13,
        "orthogonal_m3d_igzo": 25,
    }
    for architecture, maximum in expected_max.items():
        assert {row.max_resident_requests for row in by_arch[architecture].rows} == {
            maximum}

    hbm_8 = next(
        row for row in by_arch["conventional_hbm_2x1"].rows
        if row.requested_requests == 8)
    m3d_8 = next(
        row for row in by_arch["orthogonal_m3d_igzo"].rows
        if row.requested_requests == 8)
    assert hbm_8.capacity_status == "CAPACITY_PRESSURED"
    assert hbm_8.evaluation_status == "UNRESOLVED_HOST_BANDWIDTH"
    assert m3d_8.capacity_status == "FULLY_LOCAL"
    assert m3d_8.evaluation_status == "EVALUATED"
    assert result.unresolved_capacity_references == (
        "modern_high_capacity_hbm",)
