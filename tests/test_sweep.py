"""Targeted unit tests for the memory sweep benchmark framework.

The tests are deliberately small: each one pins down a single contract that
the rest of the benchmark pipeline depends on. They do not exercise the
full thermal solver; that work happens in the smoke runs in
``.smoke*.py`` at the repository root.

Run via:
    cd /tmp/om3d_project
    PYTHONPATH=src python3 -m pytest /tmp/om3d_project/tests/test_sweep.py -q
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

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
        REPO_ROOT / "configs" / "sweeps" / "memory_internal_v0.yaml")
    assert cfg.name == "memory_internal_v0"
    assert cfg.mode == "ofat"
    assert cfg.thermal is True
    assert set(cfg.cases.keys()) == {
        "conventional_hbm", "orthogonal_si", "orthogonal_m3d_igzo",
    }
    names = {s.name for s in cfg.sweeps}
    assert names == {
        "rd_per_act", "mat_rows", "mat_cols",
        "m3d_subarray_rows", "m3d_subarray_cols",
    }


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


def test_enum_axes_total_point_count_is_34():
    cfg = load_sweep_config(
        REPO_ROOT / "configs" / "sweeps" / "memory_internal_v0.yaml")
    points = list(_enum_axes(cfg))
    assert len(points) == 34
    cases_seen = {p[0] for p in points}
    assert cases_seen == {
        "conventional_hbm", "orthogonal_si", "orthogonal_m3d_igzo",
    }


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
