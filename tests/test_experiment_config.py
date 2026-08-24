from pathlib import Path

import pytest
import yaml

from om3dthermal.experiment import (
    load_architecture_spec,
    load_experiment_spec,
    load_platform_spec,
    load_workload_spec,
)


ROOT = Path(__file__).parents[1]
EXPERIMENT = (
    ROOT / "configs" / "experiment" /
    "m3d_igzo_llama31_8b_decode_conditional_v0.yaml")


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
    assert platform.fixed_gpu_power_W == 300.0
    assert experiment.scenario.rho_values == (0.0, 1.0, 100.0, 1000.0)
    assert experiment.output_policy == "ERROR_IF_EXISTS"
    assert experiment.experiment_id == (
        "m3d_igzo_llama31_8b_decode_conditional_v0")


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
