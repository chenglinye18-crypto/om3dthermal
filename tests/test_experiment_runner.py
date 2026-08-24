import json
from pathlib import Path

import pytest

import om3dthermal.experiment.runner as runner_module
from om3dthermal.evaluator import LLMDecodeWorkloadThermalMetrics
from om3dthermal.experiment import RESULT_FILES, run_experiment


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "experiment" / "conditional_e2e_v0.yaml"


def _fake_thermal(mapping):
    package_t = 20.0 + mapping.expected_package_total_power_W * 0.1
    return LLMDecodeWorkloadThermalMetrics(
        architecture=mapping.architecture,
        rho=mapping.rho,
        mapped_package_power_W=mapping.expected_package_total_power_W,
        expected_package_power_W=mapping.expected_package_total_power_W,
        source_power_breakdown_W={
            source.name: source.power_W for source in mapping.sources},
        power_closure_absolute_error_W=0.0,
        power_closure_relative_error=0.0,
        memory_Tmax_degC=package_t - 0.5,
        gpu_Tmax_degC=package_t,
        package_Tmax_degC=package_t,
        converged=True,
        iterations=1,
        final_relative_residual=1e-6,
        max_temperature_update_K=1e-4,
        relative_power_imbalance=1e-6,
        cell_count=1,
        internal_edge_count=0,
        full_vector_d2h_during_iteration=0,
        thermal_backend="gpu_pcg",
        precision_status="FP64",
        preconditioner_status="JACOBI_DIAGONAL",
        initial_temperature_K=293.15,
        relative_residual_tolerance=0.001,
        max_temperature_update_tolerance_K=0.01,
        max_iterations=100000,
        check_interval=10,
        warm_start_status="FRESH_SOLVE_NO_WARM_START",
        write_spatial_distribution_status=(
            "WRITE_SPATIAL_DISTRIBUTION_READ_SHAPE_SENSITIVITY_ONLY"),
        memory_total_completeness_status=(
            mapping.memory_total_completeness_status),
        scenario_status="CONDITIONAL_MATCHED_REFERENCE_SENSITIVITY",
    )


@pytest.fixture
def formal_run(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        runner_module, "run_llm_decode_workload_thermal", _fake_thermal)
    return run_experiment(
        CONFIG,
        project_root=ROOT,
        output_dir_override=tmp_path / "bundle",
    )


def test_formal_runner_assembles_exact_three_by_four_table(formal_run) -> None:
    assert len(formal_run.rows) == 12
    assert [(row.architecture, row.rho) for row in formal_run.rows] == [
        (architecture, rho)
        for architecture in (
            "conventional_hbm_2x1", "orthogonal_si", "orthogonal_m3d_igzo")
        for rho in (0.0, 1.0, 100.0, 1000.0)
    ]
    assert all(row.aggregate_tokens_per_second == pytest.approx(
        147.67932375509625) for row in formal_run.rows)
    assert all(row.bandwidth_capability_status == "NOT_VALIDATED"
               for row in formal_run.rows)


def test_formal_runner_writes_stage_separated_checksummed_bundle(formal_run) -> None:
    output = formal_run.output_dir
    assert output is not None
    expected = {
        "resolved_config.yaml", "manifest.json", *RESULT_FILES}
    assert expected <= {path.name for path in output.iterdir()}

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    assert set(manifest["files"]) == expected - {"manifest.json"}
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert len(summary["rows"]) == 12


def test_result_bundle_preserves_conditional_claim_boundaries(formal_run) -> None:
    output = formal_run.output_dir
    assert output is not None
    rows = json.loads((output / "summary.json").read_text(encoding="utf-8"))["rows"]
    m3d = [row for row in rows if row["architecture"] == "orthogonal_m3d_igzo"]
    assert all(row["m3d_logic_background_status"] == "CONDITIONAL_LOWER_BOUND"
               for row in m3d)
    assert all(row["system_j_token_status"] == "NOT_AVAILABLE" for row in rows)
    assert all(row["write_energy_model_status"] == "NOT_VALIDATED" for row in rows)


def test_formal_runner_rejects_nonempty_output_before_evaluation(
        tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("existing result", encoding="utf-8")

    def must_not_run(_mapping):
        raise AssertionError("thermal evaluation must not start")

    monkeypatch.setattr(
        runner_module, "run_llm_decode_workload_thermal", must_not_run)
    with pytest.raises(FileExistsError, match="ERROR_IF_EXISTS"):
        run_experiment(
            CONFIG,
            project_root=ROOT,
            output_dir_override=output,
        )
    assert (output / "keep.txt").read_text(encoding="utf-8") == "existing result"
