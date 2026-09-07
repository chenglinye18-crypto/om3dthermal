from pathlib import Path
import copy
import csv

import pytest
import yaml

from om3dthermal.experiment import (
    derive_orthogonal_slab_io_bandwidth_bits_per_second,
    load_architecture_spec,
    load_experiment_spec,
    load_platform_spec,
    load_workload_spec,
)
from om3dthermal.power import load_case_config


ROOT = Path(__file__).parents[1]
EXPERIMENT = (
    ROOT / "configs" / "experiment" /
    "m3d_igzo_llama31_8b_decode_conditional_v0.yaml")
AUDIT_EXPERIMENT = (
    ROOT / "configs" / "experiment" / "m3d_semantic_boundary_audit_v0.yaml")


def test_formal_experiment_config_resolves_three_separate_layers() -> None:
    experiment = load_experiment_spec(EXPERIMENT, project_root=ROOT)
    workload = load_workload_spec(
        experiment.workload_config, project_root=ROOT)
    platform = load_platform_spec(
        experiment.platform_config, project_root=ROOT)
    architectures = [
        load_architecture_spec(path, project_root=ROOT)
        for path in experiment.architecture_configs
    ]

    assert [item.architecture_id for item in architectures] == [
        "conventional_hbm_2x1", "orthogonal_si", "orthogonal_m3d_igzo"]
    assert workload.workload_id == "LLaMA-3.1-8B-class-B1-S131072-v0"
    assert workload.decode.batch_size == 1
    assert workload.decode.context_length == 131072
    assert workload.decode.d_head == 128
    assert workload.decode.model_dump()["d_head"] == 128
    assert any(
        item.record_id == "derived_attention_head_dimension"
        and item.classification == "SOFTWARE_DERIVED"
        for item in workload.provenance
    )
    # H200-anchored range-midpoint bandwidth-saturated operating point:
    # 74 W + 7.645 pJ/bit x 4.8 TB/s x 8 = 367.568 W.
    assert platform.gpu_decode_power.static_power_W == 74.0
    assert platform.gpu_decode_power.e_decode_J_per_bit == 7.645e-12
    assert platform.gpu_decode_power.peak_memory_bandwidth_bytes_per_s == 4.8e12
    assert platform.gpu_decode_power.derived_peak_decode_power_W == pytest.approx(
        367.568)
    raw_platform = yaml.safe_load(experiment.platform_config.read_text())
    assert set(raw_platform["gpu_decode_power"]) == {
        "model", "static_power_W", "e_decode_J_per_bit",
        "peak_memory_bandwidth_bytes_per_s", "static_power_status",
        "bandwidth_status", "coefficient_range_status",
        "coefficient_nominal_status", "model_form_status", "provenance",
    }
    compute = platform.gpu_compute_power
    assert compute.static_power_W == platform.gpu_decode_power.static_power_W
    assert compute.peak_compute_BF16_dense_flops_per_s == 989.5e12
    assert compute.compute_bound_power_W_min == 525.0
    assert compute.compute_bound_power_W_max == 700.0
    assert compute.e_compute_dynamic_J_per_FLOP_min == pytest.approx(
        (525.0 - 74.0) / 989.5e12)
    assert compute.e_compute_dynamic_J_per_FLOP_max == pytest.approx(
        (700.0 - 74.0) / 989.5e12)
    host = platform.host_offload
    assert host.effective_bandwidth_bytes_per_second == 56.2e9
    assert host.power_model_status == "INCREMENTAL_DYNAMIC_OFFLOAD_POWER"
    assert host.host_static_power_status == "UNRESOLVED"
    assert host.e_pcie_dynamic_J_per_bit == 167.9e-12
    assert host.e_pcie_dynamic_uncertainty_J_per_bit == 10.5e-12
    assert host.e_ddr_dynamic_J_per_bit == pytest.approx(
        4.93 / (24.94e9 * 8.0))
    assert host.e_host_offload_dynamic_J_per_bit == pytest.approx(
        192.6093023255814e-12)
    assert experiment.scenario.rho_values == (0.0, 1.0, 100.0, 1000.0)
    assert not hasattr(experiment.scenario, "thermal")
    assert experiment.output_policy == "ERROR_IF_EXISTS"
    assert experiment.experiment_id == (
        "m3d_igzo_llama31_8b_decode_conditional_v0")


def test_no_obsolete_parameter_path_remains() -> None:
    forbidden = (
        "_".join(("fixed", "gpu", "power", "W")),
        "_".join(("legacy", "gpu", "power")),
        "_".join(("compatibility", "gpu", "power")),
    )
    for directory in ("configs", "src", "tests", "scripts"):
        for path in (ROOT / directory).rglob("*"):
            if path.suffix.lower() not in {".py", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8")
            assert all(token not in text for token in forbidden), path
def test_gpu_platform_ledger_matches_decode_reference_range_and_nominal() -> None:
    path = ROOT / "docs" / "research" / "gpu_platform_table_2026-09-06.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = {row["name"]: row for row in csv.DictReader(stream)}
    for name in (
        "H200 SXM",
        "IOM3D baseline GPU+HBM (rev v2, planned)",
        "IOM3D-HBM proposed (rev v2, planned)",
    ):
        row = rows[name]
        assert float(row["e_decode_dynamic_pJ_per_bit_min"]) == 6.28
        assert float(row["e_decode_dynamic_pJ_per_bit_nominal"]) == 7.645
        assert float(row["e_decode_dynamic_pJ_per_bit_max"]) == 9.01


def test_architecture_descriptors_do_not_duplicate_workload_or_scenario() -> None:
    for path in (ROOT / "configs" / "architecture").glob("*.yaml"):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "workload" not in raw
        assert "bandwidth" not in raw
        assert "rho" not in raw
        assert "thermal_solver" not in raw


def test_workload_config_does_not_contain_hardware_or_thermal_fields() -> None:
    raw = yaml.safe_load((
        ROOT / "configs" / "workload" /
        "llama31_8b_decode_b1_s131072.yaml").read_text(encoding="utf-8"))
    assert not set(raw).intersection({"architecture", "power", "thermal"})
    assert "matched_payload_bandwidth_bits_per_second" not in raw["decode"]
    assert "d_head" not in raw["decode"]


def test_duplicate_or_negative_rho_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))
    raw["scenario"]["rho_values"] = [0, 1, 1]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_experiment_spec(path, project_root=ROOT)

    raw["scenario"]["rho_values"] = [-1]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="non-negative"):
        load_experiment_spec(path, project_root=ROOT)


def test_frozen_thermal_execution_cannot_be_overridden_in_experiment(
        tmp_path: Path) -> None:
    raw = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))
    assert "thermal" not in raw["scenario"]
    raw["scenario"]["thermal"] = {"backend": "gpu_pcg"}
    path = tmp_path / "thermal_override.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="thermal"):
        load_experiment_spec(path, project_root=ROOT)


def test_m3d_semantic_audit_declares_only_parametric_sensitivities() -> None:
    experiment = load_experiment_spec(AUDIT_EXPERIMENT, project_root=ROOT)
    sensitivity = experiment.scenario.m3d_parameter_sensitivity
    assert sensitivity is not None
    assert sensitivity.interface_energy_pj_per_bit == (0.25, 0.5, 1.0)
    assert sensitivity.logic_background_w == (0.0, 5.0, 10.0, 20.0)
    assert sensitivity.status == "PARAMETRIC_SENSITIVITY"


def test_formal_experiment_derives_matched_bandwidth_from_slab_io() -> None:
    experiment = load_experiment_spec(EXPERIMENT, project_root=ROOT)
    scenario = experiment.scenario
    assert scenario.matched_payload_bandwidth_bits_per_second is None
    derivation = scenario.matched_bandwidth_derivation
    assert derivation is not None
    assert derivation.derivation == "ORTHOGONAL_SLAB_IO"
    assert derivation.architecture_id == "orthogonal_m3d_igzo"
    assert derivation.cap_bits_per_second == pytest.approx(3.92e13)


def test_matched_bandwidth_requires_exactly_one_source(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))

    both = copy.deepcopy(raw)
    both["scenario"]["matched_payload_bandwidth_bits_per_second"] = 3.92e13
    path = tmp_path / "both.yaml"
    path.write_text(yaml.safe_dump(both), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_experiment_spec(path, project_root=ROOT)

    neither = copy.deepcopy(raw)
    del neither["scenario"]["matched_bandwidth_derivation"]
    path = tmp_path / "neither.yaml"
    path.write_text(yaml.safe_dump(neither), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_experiment_spec(path, project_root=ROOT)


def test_slab_io_bandwidth_derivation_scales_with_slab_count() -> None:
    case = load_case_config(
        ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml")
    orthogonal = case.geometry.orthogonal
    assert orthogonal is not None
    assert orthogonal.io_channels_per_slab == 50
    assert orthogonal.io_channel_rate_gbps == 8.0
    assert orthogonal.io_provenance == "MODELING_CHOICE"

    derived = derive_orthogonal_slab_io_bandwidth_bits_per_second(
        orthogonal, architecture_id=case.name)
    # Rev v2: slab_count 98 -> 106 on the 32 mm GPU die.
    assert derived == pytest.approx(106 * 50 * 8.0e9)
    assert derived == pytest.approx(4.24e13)

    more_slabs = orthogonal.model_copy(update={"slab_count": 112})
    assert derive_orthogonal_slab_io_bandwidth_bits_per_second(
        more_slabs, architecture_id=case.name) == pytest.approx(4.48e13)

    no_io = orthogonal.model_copy(
        update={"io_channels_per_slab": None, "io_channel_rate_gbps": None,
                "io_provenance": None})
    with pytest.raises(ValueError, match="slab IO fields"):
        derive_orthogonal_slab_io_bandwidth_bits_per_second(
            no_io, architecture_id=case.name)
