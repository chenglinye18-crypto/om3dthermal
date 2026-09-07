"""Shared local-memory-to-GPU transfer-rate closure tests."""

from __future__ import annotations

import pytest

from om3dthermal.platform import resolve_local_memory_gpu_transfer


@pytest.mark.parametrize(
    ("demand", "memory", "gpu", "actual", "bottleneck"),
    (
        (2.0e12, 5.3e12, 4.8e12, 2.0e12, "DEMAND"),
        (6.0e12, 3.0e12, 4.8e12, 3.0e12, "MEMORY"),
        (4.9e12, 5.3e12, 4.8e12, 4.8e12, "GPU"),
    ),
)
def test_transfer_bottleneck_closure(
    demand: float,
    memory: float,
    gpu: float,
    actual: float,
    bottleneck: str,
) -> None:
    point = resolve_local_memory_gpu_transfer(
        bandwidth_demand_bytes_per_s=demand,
        memory_capability_bytes_per_s=memory,
        gpu_peak_bandwidth_bytes_per_s=gpu,
    )
    assert point.bandwidth_actual_bytes_per_s == actual
    assert point.bottleneck == bottleneck


def test_transfer_tie_reports_all_limiters_with_deterministic_identity() -> None:
    point = resolve_local_memory_gpu_transfer(
        bandwidth_demand_bytes_per_s=5.0e12,
        memory_capability_bytes_per_s=4.8e12,
        gpu_peak_bandwidth_bytes_per_s=4.8e12,
    )
    assert point.bandwidth_actual_bytes_per_s == 4.8e12
    assert point.demand_limited is False
    assert point.memory_limited is True
    assert point.gpu_limited is True
    assert point.bottleneck == "MEMORY"
