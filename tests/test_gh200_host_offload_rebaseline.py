from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from om3dthermal.platform import HostOffloadSpec, load_platform_spec_file
from om3dthermal.serving import (
    FormalInferenceWorkload, WorkspaceExecutionConfig,
    evaluate_formal_inference_workload,
)
from om3dthermal.workload import load_dense_model_registry


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def profiles():
    raw = yaml.safe_load((
        ROOT/"configs/experiment/gh200_host_offload_rebaseline.yaml"
    ).read_text(encoding="utf-8"))
    return {item["host_path_id"]: HostOffloadSpec.model_validate(item)
            for item in raw["host_paths"]}


@pytest.fixture(scope="module")
def evaluated(profiles):
    raw = yaml.safe_load((
        ROOT/"configs/experiment/gh200_host_offload_rebaseline.yaml"
    ).read_text(encoding="utf-8"))
    model = load_dense_model_registry(
        ROOT/raw["model_registry_dir"])["llama31_8b"]
    workspace = WorkspaceExecutionConfig.model_validate(raw["workspace"])
    workload = FormalInferenceWorkload(
        model_id=model.model_id, batch_size=8,
        prompt_tokens=131072, generation_tokens=4)
    return {name: evaluate_formal_inference_workload(
        project_root=ROOT, model=model, workload=workload,
        system="CONVENTIONAL_HBM_GPU", policy="HOST_KV_OFFLOAD",
        workspace_config=workspace, host_offload_spec=profile)
        for name, profile in profiles.items()}


def test_gh200_nominal_bandwidth_is_direct_measured_416_34e9():
    host = load_platform_spec_file(
        ROOT/"configs/platform/gpu_package_h200_reference.yaml").host_offload
    assert host.host_path_id == "GH200_GRACE_NVLINK_C2C"
    assert host.path_role == "PRIMARY_HOST_OFFLOAD_BASELINE"
    assert host.direct_effective_bandwidth_bytes_per_second == 416.34e9
    assert host.effective_bandwidth_bytes_per_second == 416.34e9
    assert host.host_offload_efficiency is None
    assert host.effective_bandwidth_status == (
        "MEASURED_REFERENCE_GH200_H2D_EFFECTIVE_BANDWIDTH")


def test_450_is_only_upper_bound_sensitivity(profiles):
    profile = profiles["GH200_IDEAL_NVLINK_C2C_450_GBPS"]
    assert profile.effective_bandwidth_bytes_per_second == 450e9
    assert profile.path_role == "SENSITIVITY_ONLY"
    assert profile.effective_bandwidth_status == (
        "IDEAL_NVLINK_C2C_ONE_DIRECTION_UPPER_BOUND")


def test_legacy_56_2_is_explicit_sensitivity(profiles):
    profile = profiles["PCIE_DDR_LEGACY_56_2_GBPS"]
    assert profile.effective_bandwidth_bytes_per_second == 56.2e9
    assert profile.path_role == "SENSITIVITY_ONLY"
    assert profile.total_energy_status == "LEGACY_PCIE_DDR_REFERENCE"
    assert profile.e_host_offload_dynamic_J_per_bit == pytest.approx(
        192.6093023255814e-12)


def test_bandwidth_change_does_not_change_host_bytes(evaluated):
    signatures = {(
        item.historical_host_read_bytes, item.host_append_write_bytes,
        item.migration_bytes, item.total_host_transfer_bytes)
        for item in evaluated.values()}
    assert len(signatures) == 1
    assert all(item.historical_host_read_bytes > 0
               for item in evaluated.values())


def test_gh200_transfer_is_faster_than_legacy(evaluated):
    assert (evaluated["GH200_GRACE_NVLINK_C2C"].host_transfer_time_s
            < evaluated["PCIE_DDR_LEGACY_56_2_GBPS"].host_transfer_time_s)


def test_gh200_energy_components_close_without_legacy_pcie(profiles):
    profile = profiles["GH200_GRACE_NVLINK_C2C"]
    assert profile.host_memory_dynamic_J_per_bit == 4.0e-12
    assert profile.host_link_dynamic_J_per_bit == 1.3e-12
    assert profile.e_host_offload_dynamic_J_per_bit == 5.3e-12
    assert profile.e_pcie_dynamic_J_per_bit is None
    assert profile.e_ddr_dynamic_J_per_bit is None
    assert math.isclose(
        profile.host_memory_dynamic_J_per_bit
        + profile.host_link_dynamic_J_per_bit,
        profile.e_host_offload_dynamic_J_per_bit, rel_tol=1e-12)


def test_host_energy_is_bytes_times_eight_times_energy_per_bit(
    profiles, evaluated,
):
    profile = profiles["GH200_GRACE_NVLINK_C2C"]
    result = evaluated["GH200_GRACE_NVLINK_C2C"]
    bits = 8.0*result.total_host_transfer_bytes
    components = result.known_energy_components
    assert components["host_memory_dynamic_J"] == pytest.approx(
        bits*profile.host_memory_dynamic_J_per_bit)
    assert components["host_link_dynamic_J"] == pytest.approx(
        bits*profile.host_link_dynamic_J_per_bit)
    assert components["host_total_dynamic_J"] == pytest.approx(
        components["host_memory_dynamic_J"]+components["host_link_dynamic_J"])
    assert result.known_energy_J == pytest.approx(sum(
        value for key, value in components.items()
        if key != "host_total_dynamic_J"))
