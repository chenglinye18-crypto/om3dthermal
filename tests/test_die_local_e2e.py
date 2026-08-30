"""Targeted invariants for strict GPU-mediated dense placement accounting."""
from __future__ import annotations

import pytest

from scripts.evaluate_die_local_placement import ROOT, _architecture
from om3dthermal.experiment import load_experiment_spec, load_workload_spec
from om3dthermal.placement import (evaluate_die_local_placement,
    independent_domain_semantics, minimum_active_domains)
from om3dthermal.workload import build_m3d_workload_page_demand


@pytest.fixture(scope="module")
def inputs():
    layout, bandwidth = _architecture()
    workload = load_workload_spec(ROOT / "configs/workload/llama31_8b_decode_b1_s131072.yaml", project_root=ROOT).decode
    demand = build_m3d_workload_page_demand(workload, layout)
    gpu = load_experiment_spec(ROOT / "configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml", project_root=ROOT).scenario.effective_compute_flops_per_second
    return layout, bandwidth, workload, demand, gpu


def test_no_direct_die_to_die_and_fixed_external_bytes(inputs):
    layout, bandwidth, workload, demand, gpu = inputs
    d = minimum_active_domains(demand, independent_domain_semantics(layout))
    result = evaluate_die_local_placement(workload, demand, layout, bandwidth,
        active_domains=d, communication_model="FUSED_DIE_LOCAL", gpu_compute_flops_per_s=gpu)
    assert result.traffic.direct_die_to_die_bytes == 0
    assert result.traffic.local_weight_read_bytes == pytest.approx(demand.total_weight_read_bytes_per_decode_step)
    assert result.traffic.local_kv_read_bytes == pytest.approx(demand.total_kv_read_bytes_per_decode_step)
    assert result.traffic.external_interface_bytes > 0
    assert result.no_tier_no_overlap.external_interface_time_ms == pytest.approx(result.fast_no_overlap.external_interface_time_ms)


def test_domain_capacity_latency_and_dmax_global_closure(inputs):
    layout, bandwidth, workload, demand, gpu = inputs
    domains = independent_domain_semantics(layout)
    dmin = minimum_active_domains(demand, domains)
    with pytest.raises(ValueError, match="D_MIN_FIT"):
        evaluate_die_local_placement(workload, demand, layout, bandwidth,
            active_domains=dmin - 1, communication_model="FUSED_DIE_LOCAL", gpu_compute_flops_per_s=gpu)
    result = evaluate_die_local_placement(workload, demand, layout, bandwidth,
        active_domains=domains.independent_memory_domain_count,
        communication_model="CONSERVATIVE_GPU_BOUNDARY", gpu_compute_flops_per_s=gpu)
    assert max(result.fast_pack.domain_page_counts) - min(result.fast_pack.domain_page_counts) <= 1
    assert result.fast_pack.weighted_average_access_latency_ns == pytest.approx(result.fast_pack.global_fast_pack_latency_ns, rel=1e-12)
    assert result.fast_no_overlap.physical_latency_ns <= result.no_tier_no_overlap.physical_latency_ns
    assert result.traffic.direct_die_to_die_bytes == 0


def test_external_traffic_increases_with_domains_and_service_is_latency_only(inputs):
    layout, bandwidth, workload, demand, gpu = inputs
    domains = independent_domain_semantics(layout); dmin = minimum_active_domains(demand, domains)
    low = evaluate_die_local_placement(workload, demand, layout, bandwidth,
        active_domains=dmin, communication_model="FUSED_DIE_LOCAL", gpu_compute_flops_per_s=gpu)
    high = evaluate_die_local_placement(workload, demand, layout, bandwidth,
        active_domains=domains.independent_memory_domain_count, communication_model="FUSED_DIE_LOCAL", gpu_compute_flops_per_s=gpu)
    assert high.traffic.external_interface_bytes > low.traffic.external_interface_bytes
    assert low.placement_speedup == pytest.approx(low.no_tier_no_overlap.physical_latency_ns / low.fast_no_overlap.physical_latency_ns)
    assert high.fast_no_overlap.internal_bandwidth_bytes_per_s > 0
