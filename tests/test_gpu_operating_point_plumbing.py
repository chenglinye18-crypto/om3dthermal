"""Audit tests for explicit GPU operating-point propagation."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from om3dthermal.adapters import resolve_architecture_spec
from om3dthermal.experiment.config import load_architecture_spec
from om3dthermal.platform import (
    resolve_gpu_compute_power,
    resolve_gpu_decode_power,
)
from om3dthermal.power import (
    load_case_config,
    resolve_case_geometry,
    resolve_system_power,
)


ROOT = Path(__file__).parents[1]
CASE_PATH = ROOT / "configs/cases/conventional_hbm_2x1.yaml"
ARCHITECTURE_PATH = ROOT / "configs/architecture/conventional_hbm_2x1.yaml"


def test_power_and_architecture_signatures_require_resolved_point() -> None:
    for function in (resolve_system_power, resolve_architecture_spec):
        parameter = inspect.signature(function).parameters["gpu_operating_point"]
        assert parameter.default is inspect.Parameter.empty


def test_decode_operating_point_propagates_unchanged() -> None:
    case = load_case_config(CASE_PATH)
    geometry = resolve_case_geometry(case)
    point = resolve_gpu_decode_power(
        static_power_W=74.0,
        e_decode_J_per_bit=7.645e-12,
        bandwidth_demand_bytes_per_s=2.4e12,
        peak_bandwidth_bytes_per_s=4.8e12,
    )
    system = resolve_system_power(
        case, project_root=ROOT, geometry=geometry,
        gpu_operating_point=point)
    spec = load_architecture_spec(ARCHITECTURE_PATH, project_root=ROOT)
    resolved = resolve_architecture_spec(
        spec, project_root=ROOT, gpu_operating_point=point)

    assert system.gpu_power_W == pytest.approx(point.gpu_power_W)
    assert resolved.system_power.gpu_power_W == pytest.approx(point.gpu_power_W)


def test_compute_operating_point_propagates_unchanged() -> None:
    point = resolve_gpu_compute_power(
        static_power_W=74.0,
        compute_energy_dynamic_J_per_FLOP=5.0e-13,
        compute_demand_FLOP_per_s=4.0e14,
        effective_compute_ceiling_FLOP_per_s=8.0e14,
    )
    spec = load_architecture_spec(ARCHITECTURE_PATH, project_root=ROOT)
    resolved = resolve_architecture_spec(
        spec, project_root=ROOT, gpu_operating_point=point)

    assert resolved.system_power.gpu_power_W == pytest.approx(point.gpu_power_W)


def test_system_power_has_no_hidden_platform_or_gpu_resolution() -> None:
    source = inspect.getsource(resolve_system_power)

    assert "load_platform_spec_file" not in source
    assert "resolve_gpu_decode_power" not in source
