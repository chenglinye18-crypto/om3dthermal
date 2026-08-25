"""Targeted unit tests for the memory sweep benchmark framework.

The tests are deliberately small: each one pins down a single contract that
the rest of the benchmark pipeline depends on. They do not exercise the full
thermal solver. Use the canonical native-Windows environment documented in
``AGENTS.md``.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, "src")

from om3dthermal.sweep import (  # noqa: E402
    ALLOWED_ALIASES,
    SweepConfig,
    SweepAxis,
    CaseRef,
    _enum_axes,
    _nominal_value_for_alias,
    apply_override,
    load_sweep_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Sweep config parsing
# ---------------------------------------------------------------------------

def test_load_sweep_config_basic():
    cfg = load_sweep_config(
        REPO_ROOT / "configs" / "legacy" / "sweeps" /
        "memory_internal_v0.yaml")
    assert cfg.name == "memory_internal_v0"
    assert cfg.mode == "ofat"
    assert cfg.thermal is True
    # The official benchmark pins the device-resident GPU PCG backend so a
    # direct invocation cannot fall back to relaxation.
    assert cfg.thermal_backend == "gpu_pcg"
    assert set(cfg.cases.keys()) == {
        "conventional_hbm", "orthogonal_si", "orthogonal_m3d_igzo",
    }
    names = {s.name for s in cfg.sweeps}
    assert names == {
        "rd_per_act", "mat_rows", "mat_cols",
        "m3d_subarray_rows", "m3d_subarray_cols",
    }


def test_thermal_backend_explicit_gpu_parsed():
    raw = {
        "name": "gpu_pass_through",
        "mode": "ofat",
        "thermal": True,
        "thermal_backend": "gpu",
        "output_dir": "runs/sweeps/_gpu_pass_through",
        "cases": {
            "conventional_hbm": {
                "alias": "conventional_hbm",
                "path": "configs/cases/conventional_hbm_2x1.yaml",
            },
        },
        "sweeps": [{
            "name": "rd_per_act",
            "cases": ["conventional_hbm"],
            "parameter": "activated_row_data_utilization",
            "values": [0.10],
        }],
    }
    cfg = SweepConfig.model_validate(raw)
    assert cfg.thermal_backend == "gpu"


def test_thermal_backend_rejects_unknown_value():
    raw = {
        "name": "bad_backend",
        "mode": "ofat",
        "thermal": True,
        "thermal_backend": "tpu",
        "output_dir": "runs/sweeps/_bad_backend",
        "cases": {
            "x": {"alias": "x", "path": "configs/cases/x.yaml"},
        },
        "sweeps": [{
            "name": "n",
            "cases": ["x"],
            "parameter": "activated_row_data_utilization",
            "values": [0.10],
        }],
    }
    with pytest.raises(Exception):
        SweepConfig.model_validate(raw)


def test_run_sweep_applies_backend_and_output_overrides(tmp_path, monkeypatch):
    import om3dthermal.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "_enum_axes", lambda config: iter(()))
    destination = tmp_path / "pcg_results"
    result = sweep_module.run_sweep(
        REPO_ROOT / "configs" / "legacy" / "sweeps" /
        "memory_internal_v0.yaml",
        repo_root=REPO_ROOT,
        thermal_backend_override="gpu_pcg",
        output_dir_override=destination,
    )
    assert Path(result.output_dir) == destination.resolve()
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["thermal_backend"] == "gpu_pcg"
    resolved_text = (
        destination / "sweep_config.resolved.yaml").read_text()
    assert f"output_dir: {destination}" in resolved_text


def test_unknown_alias_rejected_at_config_load_time():
    raw = {
        "name": "bad",
        "mode": "ofat",
        "thermal": True,
        "output_dir": "runs/sweeps/_bad",
        "cases": {"x": {"alias": "x", "path": "configs/cases/x.yaml"}},
        "sweeps": [{
            "name": "weird",
            "cases": ["x"],
            "parameter": "totally_made_up",
            "values": [1.0],
        }],
    }
    with pytest.raises(Exception):
        SweepConfig.model_validate(raw)


def test_system_metrics_fails_loudly_without_memory_result(
        monkeypatch, tmp_path):
    """``_system_metrics`` must fail loudly if the analytical
    ``MemoryPowerResult`` is missing, instead of fabricating energy
    decomposition fields."""
    from om3dthermal.power.config import load_case_config
    from om3dthermal.power.geometry import ResolvedGeometry
    from om3dthermal.power.system import ResolvedSystemPower
    from om3dthermal.sweep import _system_metrics

    case = load_case_config(
        REPO_ROOT / "configs" / "cases" / "conventional_hbm_2x1.yaml")
    geom = ResolvedGeometry(
        source="fake",
        memory_region="hbm_dram_die",
        configured_x_mm=14.0, configured_y_mm=12.0,
        memory_region_count=1, memory_dies_per_region=1, m3d=None)
    sys_pow = ResolvedSystemPower(
        case_name="fake", architecture_type="dreamram_hbm",
        gpu_power_W=300.0, memory_power_model="reference_fixed",
        memory_power_status="placeholder",
        read_bandwidth_gbps=819.2,
        memory_access_energy_pJ_per_bit=0.0,
        memory_access_power_W=0.0, refresh_power_W=0.0,
        resolved_total_memory_power_W=0.0,
        memory_result=None,
        diagnostics={},
    )
    monkeypatch.setattr(
        "om3dthermal.sweep.resolve_case_geometry", lambda c: geom)
    monkeypatch.setattr(
        "om3dthermal.sweep.resolve_system_power",
        lambda c, project_root, geometry: sys_pow)
    monkeypatch.setattr(
        "om3dthermal.sweep._resolved_capacity",
        lambda c, g, s: {
            "system_capacity_GiB": 0.0,
            "capacity_per_instance_GiB": 0.0,
            "instance_count": 0,
            "memory_plane_area_mm2": 0.0,
            "memory_plane_density_Mb_mm2": 0.0,
            "architecture_footprint_area_mm2": 0.0,
            "architecture_footprint_density_Gb_mm2": 0.0,
        })
    with pytest.raises(RuntimeError, match="memory_result=None"):
        _system_metrics(case, REPO_ROOT)


def test_system_metrics_extracts_energy_and_power_from_memory_result(
        monkeypatch):
    """Component metrics come from ``MemoryPowerResult``, never aliases in
    ``ResolvedSystemPower`` or its diagnostics mapping.
    """
    from om3dthermal.power.config import load_case_config
    from om3dthermal.sweep import _system_metrics

    case = load_case_config(
        REPO_ROOT / "configs" / "cases" / "conventional_hbm_2x1.yaml")
    memory = SimpleNamespace(
        E_memory_internal_pj_bit=1.0,
        E_vertical_pj_bit=2.0,
        E_feol_route_pj_bit=3.0,
        E_base_route_pj_bit=4.0,
        E_interface_pj_bit=5.0,
        E_access_total_pj_bit=15.0,
        P_access_W=6.0,
        P_refresh_W=7.0,
        P_memory_background_W=8.0,
        P_logic_background_W=9.0,
        P_total_W=30.0,
    )
    sys_pow = SimpleNamespace(
        memory_result=memory,
        # Deliberately conflicting legacy/system aliases catch regressions.
        memory_access_energy_pJ_per_bit=999.0,
        memory_access_power_W=998.0,
        refresh_power_W=997.0,
        resolved_total_memory_power_W=30.0,
        gpu_power_W=300.0,
        diagnostics={
            "E_vertical_pj_bit": 996.0,
            "E_feol_route_pj_bit": 995.0,
            "E_base_route_pj_bit": 994.0,
            "E_interface_pj_bit": 993.0,
        },
    )
    capacity = {
        "system_capacity_GiB": 1.0,
        "capacity_per_instance_GiB": 1.0,
        "instance_count": 1,
        "memory_plane_area_mm2": 1.0,
        "memory_plane_density_Mb_mm2": 1.0,
        "architecture_footprint_area_mm2": 1.0,
        "architecture_footprint_density_Gb_mm2": 1.0,
    }
    monkeypatch.setattr(
        "om3dthermal.sweep.resolve_case_geometry", lambda c: object())
    monkeypatch.setattr(
        "om3dthermal.sweep.resolve_system_power",
        lambda c, project_root, geometry: sys_pow)
    monkeypatch.setattr(
        "om3dthermal.sweep._resolved_capacity",
        lambda c, g, s: capacity)

    metrics = _system_metrics(case, REPO_ROOT)

    assert metrics["E_memory_internal_pJ_per_bit"] == 1.0
    assert metrics["E_vertical_pJ_per_bit"] == 2.0
    assert metrics["E_feol_route_pJ_per_bit"] == 3.0
    assert metrics["E_base_route_pJ_per_bit"] == 4.0
    assert metrics["E_interface_pJ_per_bit"] == 5.0
    assert metrics["E_access_total_pJ_per_bit"] == 15.0
    assert metrics["P_access_W"] == 6.0
    assert metrics["P_refresh_W"] == 7.0
    assert metrics["P_memory_background_W"] == 8.0
    assert metrics["P_base_logic_background_W"] == 9.0
    assert metrics["P_memory_total_W"] == 30.0
    assert metrics["P_total_memory_W"] == 30.0


def test_enum_axes_total_point_count_is_34():
    cfg = load_sweep_config(
        REPO_ROOT / "configs" / "legacy" / "sweeps" /
        "memory_internal_v0.yaml")
    points = list(_enum_axes(cfg))
    assert len(points) == 34
    cases_seen = {p[0] for p in points}
    assert cases_seen == {
        "conventional_hbm", "orthogonal_si", "orthogonal_m3d_igzo",
    }


@pytest.mark.parametrize("case_path", [
    "configs/cases/conventional_hbm_2x1.yaml",
    "configs/cases/orthogonal_si.yaml",
])
def test_rd_act_sweep_results_are_monotonic(tmp_path, case_path):
    """Exercise the real analytical power path for all eight Si points."""
    from om3dthermal.power.config import load_case_config
    from om3dthermal.sweep import _system_metrics

    values = [0.015625, 0.03125, 0.0625, 0.10,
              0.125, 0.25, 0.50, 1.00]
    canonical = load_case_config(REPO_ROOT / case_path)
    metrics = []
    for value in values:
        point, _ = apply_override(
            canonical, REPO_ROOT,
            alias="activated_row_data_utilization", value=value,
            override_dir=tmp_path)
        metrics.append(_system_metrics(point, REPO_ROOT))

    rd_per_act = [float(m["effective_RD_per_ACT"]) for m in metrics]
    energy = [float(m["E_access_total_pJ_per_bit"]) for m in metrics]
    assert all(a < b for a, b in zip(rd_per_act, rd_per_act[1:]))
    assert all(a >= b for a, b in zip(energy, energy[1:]))


@pytest.mark.parametrize("case_path,alias,value", [
    ("configs/cases/conventional_hbm_2x1.yaml", "mat_rows", 512),
    ("configs/cases/orthogonal_si.yaml", "mat_cols", 512),
    ("configs/cases/orthogonal_m3d_igzo.yaml", "m3d_subarray_rows", 512),
])
def test_nominal_override_reproduces_canonical_system_metrics(
        tmp_path, case_path, alias, value):
    """Nominal OFAT overrides reproduce canonical capacity, energy, power."""
    from om3dthermal.power.config import load_case_config
    from om3dthermal.sweep import _system_metrics

    canonical = load_case_config(REPO_ROOT / case_path)
    point, _ = apply_override(
        canonical, REPO_ROOT, alias=alias, value=value,
        override_dir=tmp_path)
    expected = _system_metrics(canonical, REPO_ROOT)
    actual = _system_metrics(point, REPO_ROOT)
    for key in (
        "system_capacity_GiB",
        "E_access_total_pJ_per_bit",
        "P_total_memory_W",
    ):
        assert actual[key] == pytest.approx(expected[key], rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Canonical-case immutability across override application
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_name,path", [
    ("conventional_hbm", "configs/cases/conventional_hbm_2x1.yaml"),
    ("orthogonal_si", "configs/cases/orthogonal_si.yaml"),
    ("orthogonal_m3d_igzo", "configs/cases/orthogonal_m3d_igzo.yaml"),
])
def test_canonical_case_object_unchanged_after_override(
        tmp_path, case_name, path):
    """Loading a case, applying an override, and re-reading the YAML must
    yield the exact same byte-for-byte canonical config."""
    from om3dthermal.power.config import load_case_config
    canonical = load_case_config(REPO_ROOT / path)

    # Snapshot the canonical model's dump before any override.
    before = canonical.model_dump(mode="json")
    rd_before = (
        canonical.workload.row_policy.activated_row_data_utilization
        if canonical.workload.row_policy is not None else None)

    # Pick a non-nominal value
    if case_name == "orthogonal_m3d_igzo":
        alias = "m3d_subarray_rows"
        value = 1024
    elif canonical.workload.row_policy is not None:
        alias = "activated_row_data_utilization"
        value = 0.5
    else:
        pytest.skip("case has no applicable alias")

    new_case, _ = apply_override(
        canonical, REPO_ROOT, alias=alias, value=value,
        override_dir=tmp_path)

    # Canonical must be the same object, with the same internal state.
    after = canonical.model_dump(mode="json")
    assert before == after
    if rd_before is not None:
        assert (canonical.workload.row_policy.activated_row_data_utilization
                == rd_before)
    # And the new case must actually be different.
    assert new_case is not canonical
    assert new_case.model_dump(mode="json") != after


# ---------------------------------------------------------------------------
# 3. Parameter applicability
# ---------------------------------------------------------------------------

def test_activated_row_data_utilization_does_not_apply_to_m3d(tmp_path):
    from om3dthermal.power.config import load_case_config
    case = load_case_config(
        REPO_ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml")
    with pytest.raises(ValueError, match="does not apply"):
        apply_override(
            case, REPO_ROOT, alias="activated_row_data_utilization",
            value=0.5, override_dir=tmp_path)


def test_m3d_subarray_does_not_apply_to_hbm(tmp_path):
    from om3dthermal.power.config import load_case_config
    case = load_case_config(
        REPO_ROOT / "configs" / "cases" / "conventional_hbm_2x1.yaml")
    with pytest.raises(ValueError, match="does not apply"):
        apply_override(
            case, REPO_ROOT, alias="m3d_subarray_rows",
            value=512, override_dir=tmp_path)


# ---------------------------------------------------------------------------
# 4. DreamRAM MAT override does not modify third_party
# ---------------------------------------------------------------------------

def test_mat_override_writes_only_into_sweep_dir(tmp_path):
    """The pinned DreamRAM config under third_party/ must be byte-identical
    before and after a MAT override. The override JSON must live under
    the sweep's point_configs/ directory."""
    from om3dthermal.power.config import load_case_config
    case = load_case_config(
        REPO_ROOT / "configs" / "cases" / "conventional_hbm_2x1.yaml")
    pinned_path = (REPO_ROOT / case.memory.dreamram.memory_config).resolve()
    pinned_before = pinned_path.read_bytes()

    new_case, prov = apply_override(
        case, REPO_ROOT, alias="mat_rows", value=1024,
        override_dir=tmp_path)

    pinned_after = pinned_path.read_bytes()
    assert pinned_before == pinned_after, (
        "MAT override must not modify third_party/DreamRAM/")

    override_path = Path(prov["override_json_path"])
    assert override_path.exists()
    assert "third_party" not in override_path.parts
    assert override_path.parent.name == "point_configs"

    data = json.loads(override_path.read_text())
    assert data["memconfig"]["mat"]["wordlines"] == 1024
    # The non-overridden axis must be preserved.
    assert data["memconfig"]["mat"]["bitlines"] == 512


def test_mat_cols_override_does_not_touch_mat_rows(tmp_path):
    from om3dthermal.power.config import load_case_config
    case = load_case_config(
        REPO_ROOT / "configs" / "cases" / "orthogonal_si.yaml")
    new_case, prov = apply_override(
        case, REPO_ROOT, alias="mat_cols", value=1024,
        override_dir=tmp_path)
    data = json.loads(Path(prov["override_json_path"]).read_text())
    assert data["memconfig"]["mat"]["bitlines"] == 1024
    assert data["memconfig"]["mat"]["wordlines"] == 512


# ---------------------------------------------------------------------------
# 5. M3D subarray override
# ---------------------------------------------------------------------------

def test_m3d_subarray_rows_override_recomputes_subarray(tmp_path):
    from om3dthermal.power.config import load_case_config
    case = load_case_config(
        REPO_ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml")
    new_case, prov = apply_override(
        case, REPO_ROOT, alias="m3d_subarray_rows", value=1024,
        override_dir=tmp_path)
    assert (new_case.architecture.m3d_subarray.subarray.n_rows
            == 1024)
    assert prov["old_value"] == 512
    assert prov["new_value"] == 1024


# ---------------------------------------------------------------------------
# 6. Nominal flagging
# ---------------------------------------------------------------------------

def test_nominal_value_for_alias_per_case():
    from om3dthermal.power.config import load_case_config
    for case_name, path, alias, expected in [
        ("conventional_hbm", "configs/cases/conventional_hbm_2x1.yaml",
         "activated_row_data_utilization", 0.10),
        ("orthogonal_si", "configs/cases/orthogonal_si.yaml",
         "activated_row_data_utilization", 0.10),
        ("conventional_hbm", "configs/cases/conventional_hbm_2x1.yaml",
         "mat_rows", 512),
        ("conventional_hbm", "configs/cases/conventional_hbm_2x1.yaml",
         "mat_cols", 512),
        ("orthogonal_m3d_igzo",
         "configs/cases/orthogonal_m3d_igzo.yaml",
         "m3d_subarray_rows", 512),
        ("orthogonal_m3d_igzo",
         "configs/cases/orthogonal_m3d_igzo.yaml",
         "m3d_subarray_cols", 512),
    ]:
        canonical = load_case_config(REPO_ROOT / path)
        v = _nominal_value_for_alias(alias, canonical, REPO_ROOT)
        assert v == expected, (
            f"{case_name} {alias} expected {expected}, got {v}")


# ---------------------------------------------------------------------------
# 7. Output schema (lightweight, without running the full pipeline)
# ---------------------------------------------------------------------------

def test_sweep_alias_table_contains_documented_fields():
    """Every alias must declare the four fields we report in the
    parameter-mapping table."""
    for name, info in ALLOWED_ALIASES.items():
        assert "applies_to" in info, name
        assert "physical_field" in info, name
        assert "type" in info, name
        assert "description" in info, name


# ---------------------------------------------------------------------------
# 8. Failure handling (config-validate path; thermal path is exercised
#    in the smoke runs)
# ---------------------------------------------------------------------------

def test_invalid_value_for_m3d_subarray_is_rejected(tmp_path):
    from om3dthermal.power.config import load_case_config
    case = load_case_config(
        REPO_ROOT / "configs" / "cases" / "orthogonal_m3d_igzo.yaml")
    with pytest.raises(ValueError):
        apply_override(
            case, REPO_ROOT, alias="m3d_subarray_cols", value=0,
            override_dir=tmp_path)


def test_invalid_value_for_row_policy_is_rejected(tmp_path):
    from om3dthermal.power.config import load_case_config
    case = load_case_config(
        REPO_ROOT / "configs" / "cases" / "conventional_hbm_2x1.yaml")
    with pytest.raises(ValueError):
        apply_override(
            case, REPO_ROOT, alias="activated_row_data_utilization",
            value=1.5, override_dir=tmp_path)
