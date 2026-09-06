"""Boundary and regime tests for the canonical GPU compute power model."""

from pathlib import Path

import pytest

from om3dthermal.evaluator.llm_decode_gpu_energy import (
    evaluate_gpu_decode_energy,
)
from om3dthermal.experiment.config import load_platform_spec
from om3dthermal.platform import PlatformSpec, resolve_gpu_compute_power

from test_llm_decode_gpu_energy import _energy, _performance


ROOT = Path(__file__).parents[1]
PLATFORM = ROOT / "configs/platform/gpu_package_h200_reference.yaml"
STATIC_POWER_W = 74.0
VENDOR_PEAK_FLOPS_PER_S = 989.5e12
E_DYNAMIC_MIN = 4.557857503789793e-13
E_DYNAMIC_MAX = 6.326427488630622e-13


def _resolve(demand: float, coefficient: float = E_DYNAMIC_MIN):
    return resolve_gpu_compute_power(
        static_power_W=STATIC_POWER_W,
        compute_energy_dynamic_J_per_FLOP=coefficient,
        compute_demand_FLOP_per_s=demand,
        effective_compute_ceiling_FLOP_per_s=VENDOR_PEAK_FLOPS_PER_S,
    )


def test_zero_compute_demand_returns_shared_static_power() -> None:
    result = _resolve(0.0)
    assert result.compute_actual_flops_per_s == 0.0
    assert result.compute_utilization == 0.0
    assert result.compute_saturated is False
    assert result.gpu_dynamic_compute_power_W == 0.0
    assert result.gpu_power_W == STATIC_POWER_W


def test_half_compute_point() -> None:
    result = _resolve(0.5 * VENDOR_PEAK_FLOPS_PER_S)
    assert result.compute_actual_flops_per_s == 0.5 * VENDOR_PEAK_FLOPS_PER_S
    assert result.compute_utilization == pytest.approx(0.5)
    assert result.compute_saturated is False
    assert result.gpu_power_W == pytest.approx(
        STATIC_POWER_W + E_DYNAMIC_MIN * 0.5 * VENDOR_PEAK_FLOPS_PER_S)


def test_exact_compute_boundary_is_not_strictly_saturated() -> None:
    result = _resolve(VENDOR_PEAK_FLOPS_PER_S)
    assert result.compute_actual_flops_per_s == VENDOR_PEAK_FLOPS_PER_S
    assert result.compute_utilization == 1.0
    assert result.compute_saturated is False
    assert result.gpu_power_W == pytest.approx(525.0)


def test_above_compute_boundary_is_capped() -> None:
    result = _resolve(1.2 * VENDOR_PEAK_FLOPS_PER_S)
    assert result.compute_demand_flops_per_s == pytest.approx(
        1.2 * VENDOR_PEAK_FLOPS_PER_S)
    assert result.compute_actual_flops_per_s == VENDOR_PEAK_FLOPS_PER_S
    assert result.compute_saturated is True
    assert result.gpu_power_W == pytest.approx(525.0)


def test_compute_power_is_continuous_at_boundary() -> None:
    left = _resolve(VENDOR_PEAK_FLOPS_PER_S * (1.0 - 1e-9))
    boundary = _resolve(VENDOR_PEAK_FLOPS_PER_S)
    assert left.gpu_power_W == pytest.approx(
        boundary.gpu_power_W, rel=1e-9)


def test_compute_power_is_monotonic_then_constant() -> None:
    demands = [
        0.0,
        0.25 * VENDOR_PEAK_FLOPS_PER_S,
        0.5 * VENDOR_PEAK_FLOPS_PER_S,
        VENDOR_PEAK_FLOPS_PER_S,
        1.2 * VENDOR_PEAK_FLOPS_PER_S,
        2.0 * VENDOR_PEAK_FLOPS_PER_S,
    ]
    powers = [_resolve(demand).gpu_power_W for demand in demands]
    assert powers[0] < powers[1] < powers[2] < powers[3]
    assert powers[3:] == pytest.approx([525.0, 525.0, 525.0])


def test_dynamic_coefficient_endpoints_close_to_total_power_range() -> None:
    assert _resolve(VENDOR_PEAK_FLOPS_PER_S, E_DYNAMIC_MIN).gpu_power_W == (
        pytest.approx(525.0))
    assert _resolve(VENDOR_PEAK_FLOPS_PER_S, E_DYNAMIC_MAX).gpu_power_W == (
        pytest.approx(700.0))
    assert 525.0 / 989.5e12 * 1e12 == pytest.approx(0.5305709954522486)
    assert 700.0 / 989.5e12 * 1e12 == pytest.approx(0.7074279939363315)


def test_platform_separates_vendor_peak_from_effective_scenario_ceiling() -> None:
    platform = load_platform_spec(PLATFORM, project_root=ROOT)
    decode = platform.gpu_decode_power
    compute = platform.gpu_compute_power
    assert decode.static_power_W == compute.static_power_W == 74.0
    assert compute.peak_compute_BF16_dense_flops_per_s == 989.5e12
    assert compute.e_compute_dynamic_J_per_FLOP_min == pytest.approx(
        (525.0 - 74.0) / 989.5e12)
    assert compute.e_compute_dynamic_J_per_FLOP_max == pytest.approx(
        (700.0 - 74.0) / 989.5e12)


def test_platform_rejects_different_static_power_between_regimes() -> None:
    platform = load_platform_spec(PLATFORM, project_root=ROOT)
    data = platform.model_dump()
    compute = data["gpu_compute_power"]
    compute["static_power_W"] = 75.0
    compute["e_compute_dynamic_J_per_FLOP_min"] = (
        (525.0 - 75.0) / VENDOR_PEAK_FLOPS_PER_S)
    compute["e_compute_dynamic_J_per_FLOP_max"] = (
        (700.0 - 75.0) / VENDOR_PEAK_FLOPS_PER_S)
    with pytest.raises(ValueError, match="must share static power"):
        PlatformSpec.model_validate(data)


def test_compute_regime_uses_explicit_effective_ceiling_and_coefficient() -> None:
    platform = load_platform_spec(PLATFORM, project_root=ROOT)
    performance = _performance(
        token_time_s=1e-12, bottleneck="COMPUTE").model_copy(update={
            "effective_compute_flops_per_second": 100e12,
            "compute_throughput_required_to_match_memory_flops_per_second": (
                120e12),
        })
    result = evaluate_gpu_decode_energy(
        performance,
        _energy(),
        platform.gpu_decode_power,
        platform.gpu_compute_power,
        compute_energy_dynamic_J_per_FLOP=E_DYNAMIC_MIN,
    )
    assert result.gpu_power_regime == "COMPUTE"
    assert result.compute_demand_flops_per_s == 120e12
    assert result.compute_actual_flops_per_s == 100e12
    assert result.compute_saturated is True
    assert result.gpu_decode_power_W == pytest.approx(
        74.0 + E_DYNAMIC_MIN * 100e12)
    assert result.bandwidth_actual_bytes_per_s is None


def test_compute_regime_has_no_implicit_nominal() -> None:
    platform = load_platform_spec(PLATFORM, project_root=ROOT)
    performance = _performance(
        token_time_s=1e-12, bottleneck="COMPUTE").model_copy(update={
            "effective_compute_flops_per_second": 100e12,
            "compute_throughput_required_to_match_memory_flops_per_second": (
                120e12),
        })
    with pytest.raises(ValueError, match="no nominal is configured"):
        evaluate_gpu_decode_energy(
            performance,
            _energy(),
            platform.gpu_decode_power,
            platform.gpu_compute_power,
        )


def test_effective_compute_ceiling_cannot_exceed_vendor_peak() -> None:
    platform = load_platform_spec(PLATFORM, project_root=ROOT)
    performance = _performance().model_copy(update={
        "effective_compute_flops_per_second": 1.01 * VENDOR_PEAK_FLOPS_PER_S,
    })
    with pytest.raises(ValueError, match="cannot exceed vendor peak"):
        evaluate_gpu_decode_energy(
            performance,
            _energy(),
            platform.gpu_decode_power,
            platform.gpu_compute_power,
        )


def test_balanced_regime_is_explicitly_unresolved() -> None:
    platform = load_platform_spec(PLATFORM, project_root=ROOT)
    performance = _performance(bottleneck="BALANCED")
    with pytest.raises(ValueError, match="balanced GPU power is unresolved"):
        evaluate_gpu_decode_energy(
            performance,
            _energy(),
            platform.gpu_decode_power,
            platform.gpu_compute_power,
        )
