import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import om3dthermal.experiment.runner as runner_module
from om3dthermal.evaluator import LLMDecodeWorkloadThermalMetrics
from om3dthermal.experiment import (
    RESULT_FILES,
    MatchedBandwidthDerivationSpec,
    run_experiment,
)
from om3dthermal.power import load_case_config


ROOT = Path(__file__).parents[1]
CONFIG = (
    ROOT / "configs" / "experiment" /
    "m3d_igzo_llama31_8b_decode_conditional_v0.yaml")


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
        74.55887865736948) for row in formal_run.rows)
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


def test_formal_runner_evaluates_gpu_decode_energy_stage(formal_run) -> None:
    """E8 supplies the GPU power used by E5, E6 and the E7 rows."""
    gpu_rows = formal_run.gpu_decode_energy
    assert gpu_rows is not None
    assert len(gpu_rows) == len(formal_run.rows) == 12
    e7 = {(row.architecture, row.rho): row for row in formal_run.rows}
    for gpu in gpu_rows:
        assert gpu.evaluation_status == (
            "EVALUATED_ANALYTICAL_GPU_DECODE_ENERGY")
        # Performance consumes the separately resolved 2.4 TB/s sustained
        # service.  Until measurement distinguishes the power interpretation,
        # formal GPU power conservatively remains at the 4.8 TB/s ceiling OP.
        assert gpu.memory_bandwidth_utilization == pytest.approx(0.5)
        assert gpu.utilization_clamped is False
        assert gpu.bandwidth_demand_bytes_per_s == pytest.approx(2.4e12)
        assert gpu.bandwidth_actual_bytes_per_s == pytest.approx(2.4e12)
        assert gpu.bandwidth_saturated is False
        assert gpu.gpu_dynamic_power_W == pytest.approx(224.256)
        assert gpu.gpu_decode_power_W == pytest.approx(298.256)
        assert gpu.gpu_energy_j_per_token == pytest.approx(
            298.256 * gpu.token_time_s)
        row = e7[(gpu.architecture, gpu.rho)]
        assert row.gpu_power_W == gpu.gpu_decode_power_W
        assert gpu.system_energy_j_per_token == pytest.approx(
            gpu.gpu_energy_j_per_token
            + row.memory_dynamic_energy_j_per_token)

    output = formal_run.output_dir
    assert output is not None
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["gpu_decode_energy"]) == 12
    provenance = json.loads(
        (output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["environment"]["gpu_decode_energy_stage_status"] == (
        "EVALUATED_ANALYTICAL_GPU_DECODE_ENERGY")


@pytest.mark.parametrize("bandwidth_scale", [1.0, 0.5])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_runner_shares_gpu_operating_point_in_energy_power_and_thermal(
        tmp_path, monkeypatch, bandwidth_scale, batch_size):
    original_experiment = runner_module.load_experiment_spec
    original_workload = runner_module.load_workload_spec

    def experiment(*args, **kwargs):
        spec = original_experiment(*args, **kwargs)
        return spec.model_copy(update={"scenario": spec.scenario.model_copy(update={
            "matched_payload_bandwidth_bits_per_second": 39.2e12 * bandwidth_scale,
            "matched_bandwidth_derivation": None})})

    def workload(*args, **kwargs):
        spec = original_workload(*args, **kwargs)
        return spec.model_copy(update={"decode": spec.decode.model_copy(update={
            "batch_size": batch_size})})

    monkeypatch.setattr(runner_module, "load_experiment_spec", experiment)
    monkeypatch.setattr(runner_module, "load_workload_spec", workload)
    monkeypatch.setattr(runner_module, "run_llm_decode_workload_thermal", _fake_thermal)
    result = run_experiment(CONFIG, project_root=ROOT,
                            output_dir_override=tmp_path / "shared_gpu")
    power_rows = json.loads((result.output_dir / "power.json").read_text())
    thermal_rows = json.loads((result.output_dir / "thermal.json").read_text())
    bandwidth_demand = 4.9e12 * bandwidth_scale
    bandwidth_ceiling = min(bandwidth_demand, 4.8e12)
    bandwidth_actual = bandwidth_ceiling
    expected_gpu = 74.0 + 11.68e-12 * 8.0 * bandwidth_actual
    for row, gpu, power, thermal in zip(
            result.rows, result.gpu_decode_energy, power_rows, thermal_rows):
        assert row.gpu_power_W == pytest.approx(expected_gpu)
        assert gpu.bandwidth_demand_bytes_per_s == pytest.approx(
            bandwidth_actual)
        assert gpu.bandwidth_actual_bytes_per_s == pytest.approx(
            bandwidth_actual)
        assert gpu.bandwidth_saturated is False
        assert power["gpu_power_W"] == gpu.gpu_decode_power_W == row.gpu_power_W
        assert thermal["source_power_breakdown_W"]["gpu"] == row.gpu_power_W
        assert gpu.gpu_energy_j_per_token * row.aggregate_tokens_per_second == (
            pytest.approx(row.gpu_power_W))
        assert row.package_power_W == pytest.approx(
            row.gpu_power_W + row.memory_total_power_W)
        # M3D dynamic power and throughput consume the same sustained service,
        # so per-token dynamic energy closes directly without a second scale.
        static_memory = (power["refresh_power_W"] + power["memory_background_power_W"]
                         + power["logic_background_effective_W"])
        assert row.package_power_W == pytest.approx(
            (gpu.gpu_energy_j_per_token
             + gpu.memory_dynamic_energy_j_per_token)
            * row.aggregate_tokens_per_second
            + static_memory)
        assert thermal["mapped_package_power_W"] == pytest.approx(row.package_power_W)


def test_runner_rejects_missing_canonical_gpu_model(monkeypatch):
    original = runner_module.load_platform_spec

    def platform(*args, **kwargs):
        return original(*args, **kwargs).model_copy(update={"gpu_decode_power": None})

    monkeypatch.setattr(runner_module, "load_platform_spec", platform)
    with pytest.raises(ValueError, match="requires gpu_decode_power"):
        run_experiment(CONFIG, project_root=ROOT, write_bundle=False)


def test_single_platform_coefficient_propagates_to_power_and_thermal(
        monkeypatch):
    original = runner_module.load_platform_spec

    def run_with(coefficient):
        mappings = []

        def platform(*args, **kwargs):
            loaded = original(*args, **kwargs)
            decode = loaded.gpu_decode_power.model_copy(
                update={"e_decode_J_per_bit": coefficient})
            return loaded.model_copy(update={"gpu_decode_power": decode})

        def thermal(mapping):
            mappings.append(mapping)
            return _fake_thermal(mapping)

        monkeypatch.setattr(runner_module, "load_platform_spec", platform)
        monkeypatch.setattr(
            runner_module, "run_llm_decode_workload_thermal", thermal)
        result = run_experiment(CONFIG, project_root=ROOT, write_bundle=False)
        return result, mappings

    nominal, nominal_mappings = run_with(11.68e-12)
    changed, changed_mappings = run_with(12.68e-12)
    expected_delta_W = 1.0e-12 * 8.0 * 2.4e12
    for before, after, before_map, after_map in zip(
            nominal.rows, changed.rows,
            nominal_mappings[:len(nominal.rows)],
            changed_mappings[:len(changed.rows)]):
        assert after.gpu_power_W - before.gpu_power_W == pytest.approx(
            expected_delta_W)
        assert after.package_power_W - before.package_power_W == pytest.approx(
            expected_delta_W)
        before_gpu = next(
            source.power_W for source in before_map.sources
            if source.name == "gpu")
        after_gpu = next(
            source.power_W for source in after_map.sources
            if source.name == "gpu")
        assert after_gpu - before_gpu == pytest.approx(expected_delta_W)


def test_m3d_sensitivity_uses_same_reduced_gpu_power_as_main_rows(monkeypatch, sensitivity_config):
    original = runner_module.load_experiment_spec
    mappings = []

    def experiment(*args, **kwargs):
        spec = original(*args, **kwargs)
        return spec.model_copy(update={"scenario": spec.scenario.model_copy(update={
            "matched_payload_bandwidth_bits_per_second": 19.6e12,
            "matched_bandwidth_derivation": None})})

    def thermal(mapping):
        mappings.append(mapping)
        return _fake_thermal(mapping)

    monkeypatch.setattr(runner_module, "load_experiment_spec", experiment)
    monkeypatch.setattr(runner_module, "run_llm_decode_workload_thermal", thermal)
    result = run_experiment(sensitivity_config, project_root=ROOT, write_bundle=False)
    assert len(mappings) == 5  # nominal plus four logic-background points
    for mapping in mappings:
        gpu = next(source for source in mapping.sources if source.name == "gpu")
        assert gpu.power_W == pytest.approx(302.928)
        assert "SHARED_WITH_ENERGY" in gpu.mapping_provenance
    for row in result.m3d_parameter_sensitivity.logic_background_rows:
        assert row.package_total_power_W - row.memory_total_power_W == pytest.approx(302.928)


def test_result_bundle_persists_workload_demand_boundary(formal_run) -> None:
    output = formal_run.output_dir
    assert output is not None
    workload = json.loads(
        (output / "workload.json").read_text(encoding="utf-8")
    )
    demand = workload["demand"]
    metrics = workload["metrics"]
    assert demand["required_capacity_bytes"] == metrics["required_capacity_bytes"]
    assert demand["read_bytes_per_output"] == metrics["read_bytes_per_token"]
    assert demand["traffic_scope_status"] == (
        "ALGORITHMIC_WORKLOAD_TRAFFIC_NOT_PHYSICAL_DRAM_TRAFFIC"
    )


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


def test_runner_executes_configured_m3d_parameter_sensitivity(monkeypatch, sensitivity_config) -> None:
    monkeypatch.setattr(
        runner_module, "run_llm_decode_workload_thermal", _fake_thermal)
    result = run_experiment(
        sensitivity_config, project_root=ROOT, write_bundle=False)
    sensitivity = result.m3d_parameter_sensitivity
    assert sensitivity is not None
    assert len(result.rows) == 1
    assert [row.interface_energy_pj_per_bit
            for row in sensitivity.interface_rows] == [0.25, 0.5, 1.0]
    assert [row.logic_background_power_W
            for row in sensitivity.logic_background_rows] == [0.0, 5.0, 10.0, 20.0]


def test_runner_records_derived_matched_bandwidth_provenance(formal_run) -> None:
    environment = formal_run.provenance.environment
    assert environment["matched_bandwidth_resolution"] == (
        "DERIVED_FROM_ORTHOGONAL_SLAB_IO")
    assert environment["matched_bandwidth_bits_per_second"] == (
        pytest.approx(1.92e13))
    assert environment["matched_bandwidth_capability_bits_per_second"] == (
        pytest.approx(1.272e14))
    transfer = environment["memory_gpu_transfer_operating_points"][
        "orthogonal_m3d_igzo"]
    assert transfer["bandwidth_demand_bytes_per_s"] == pytest.approx(2.4e12)
    assert transfer["memory_capability_bytes_per_s"] == pytest.approx(15.9e12)
    assert transfer["gpu_peak_bandwidth_bytes_per_s"] == pytest.approx(4.8e12)
    assert transfer["bandwidth_actual_bytes_per_s"] == pytest.approx(2.4e12)
    assert transfer["bottleneck"] == "DEMAND"
    service = environment["gpu_bandwidth_service_operating_points"][
        "orthogonal_m3d_igzo"]
    assert service["transfer_ceiling_bytes_per_s"] == pytest.approx(2.4e12)
    assert service["sustained_bandwidth_bytes_per_s"] == pytest.approx(2.4e12)
    assert service["service_status"] == "DIRECT_TRANSFER_CEILING"


def test_capped_derivation_pins_bandwidth_below_derived_capability() -> None:
    case = load_case_config(ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml")
    orthogonal_106 = case.geometry.orthogonal.model_copy(
        update={"slab_count": 106})
    case_106 = case.model_copy(update={
        "geometry": case.geometry.model_copy(
            update={"orthogonal": orthogonal_106})})
    resolved = SimpleNamespace(
        spec=SimpleNamespace(architecture_id="orthogonal_m3d_igzo"),
        case=case_106)

    capped = SimpleNamespace(
        matched_payload_bandwidth_bits_per_second=None,
        matched_bandwidth_derivation=MatchedBandwidthDerivationSpec(
            derivation="ORTHOGONAL_SLAB_IO",
            architecture_id="orthogonal_m3d_igzo",
            cap_bits_per_second=3.92e13))
    applied, capability = (
        runner_module._resolve_matched_bandwidth_bits_per_second(
            capped, (resolved,)))
    assert capability == pytest.approx(106 * 50 * 8.0e9)
    assert applied == pytest.approx(3.92e13)

    uncapped = SimpleNamespace(
        matched_payload_bandwidth_bits_per_second=None,
        matched_bandwidth_derivation=MatchedBandwidthDerivationSpec(
            derivation="ORTHOGONAL_SLAB_IO",
            architecture_id="orthogonal_m3d_igzo"))
    applied, capability = (
        runner_module._resolve_matched_bandwidth_bits_per_second(
            uncapped, (resolved,)))
    assert capability == pytest.approx(4.24e13)
    assert applied == pytest.approx(4.24e13)


@pytest.fixture
def sensitivity_config(tmp_path):
    import yaml
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["architecture_configs"] = ["configs/architecture/orthogonal_m3d_igzo.yaml"]
    raw["scenario"]["rho_values"] = [1]
    raw["scenario"]["unresolved_logic_background_policy"] = {"orthogonal_m3d_igzo": "EXISTING_PLACEHOLDER_ZERO"}
    raw["scenario"]["m3d_parameter_sensitivity"] = {
        "architecture_id": "orthogonal_m3d_igzo",
        "interface_energy_pj_per_bit": [0.25, 0.5, 1.0],
        "logic_background_w": [0, 5, 10, 20],
        "status": "PARAMETRIC_SENSITIVITY",
    }
    path = tmp_path / "sensitivity.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path
