"""A-final path and operator-locality invariants."""
from __future__ import annotations
import pytest
from dataclasses import replace
from scripts.evaluate_die_local_placement import ROOT, _architecture
from om3dthermal.experiment import load_experiment_spec, load_workload_spec
from om3dthermal.placement import evaluate_nmp_locality_case, independent_physical_die_count
from om3dthermal.power import calculate_memory_power, calculate_physical_access_latency, load_case_config, resolve_case_geometry
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.workload import build_m3d_workload_page_demand

@pytest.fixture(scope='module')
def inputs():
    layout,bw=_architecture(); c=load_case_config(ROOT/'configs/cases/orthogonal_m3d_igzo.yaml'); g=resolve_case_geometry(c); power=calculate_memory_power(c,project_root=ROOT,geometry=g); top=calculate_m3d_subarray(c.architecture.m3d_subarray,g.m3d); feol=calculate_feol_route(c.architecture.feol_route,top)
    phy=calculate_physical_access_latency(c.architecture.physical_access_latency,feol_route=feol,miv_length_per_layer_um=power.diagnostics['miv_length_per_layer_um'],miv_delay_per_layer_ns=power.diagnostics['miv_delay_per_layer_ns'],miv_status=power.diagnostics['miv_latency_status'],miv_parameter_status=power.diagnostics['miv_resistance_parameter_status'],miv_provenance=power.diagnostics['miv_resistance_provenance'])
    w=load_workload_spec(ROOT/'configs/workload/llama31_8b_decode_b1_s131072.yaml',project_root=ROOT).decode; d=build_m3d_workload_page_demand(w,layout); gpu=load_experiment_spec(ROOT/'configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml',project_root=ROOT).scenario.effective_compute_flops_per_second
    return layout,bw,phy,w,d,gpu

def test_path_semantics_and_locality(inputs):
    l,b,p,w,d,g=inputs
    non=evaluate_nmp_locality_case(w,d,l,p,b,case='NON_NMP_GPU',nmp_aggregate_tflops=None,gpu_compute_flops_per_s=g)
    naive=evaluate_nmp_locality_case(w,d,l,p,b,case='NMP_NAIVE',nmp_aggregate_tflops=64,gpu_compute_flops_per_s=g)
    local=evaluate_nmp_locality_case(w,d,l,p,b,case='NMP_LOCALITY_AWARE_PLACEMENT',nmp_aggregate_tflops=64,gpu_compute_flops_per_s=g)
    assert independent_physical_die_count(l)==98
    assert non.placement.long_feol_edge_included and non.traffic.weight_bulk_external_bytes>0 and non.traffic.kv_bulk_external_bytes>0
    assert not naive.placement.long_feol_edge_included and naive.traffic.weight_bulk_external_bytes==0
    assert naive.traffic.direct_die_to_die_bytes==local.traffic.direct_die_to_die_bytes==0
    assert local.placement.mean_operator_die_span<=naive.placement.mean_operator_die_span
    assert local.traffic.external_interface_bytes<=naive.traffic.external_interface_bytes
    assert max(local.placement.die_used_bytes)<=l.capacity_per_slab_bytes

def test_topology_local_groups_are_decoupled_from_coils(inputs):
    l,b,p,w,d,g=inputs
    assert b.local_service_groups_per_die == l.clusters_per_slab // b.clusters_per_service == 70
    assert b.total_local_service_groups == 98 * 70
    assert b.read_payload_bytes_per_service == 32
    current=evaluate_nmp_locality_case(w,d,l,p,b,case='NMP_LOCALITY_AWARE_PLACEMENT',nmp_aggregate_tflops=64,gpu_compute_flops_per_s=g)
    # Altering external-resource metadata and its aggregate link rate cannot
    # alter local NMP service time; it only changes external boundary time.
    altered=replace(b, coil_links_per_die=25, external_coil_links_per_die=25,
        coil_bandwidth_bytes_per_s=b.coil_bandwidth_bytes_per_s/2)
    changed=evaluate_nmp_locality_case(w,d,l,p,altered,case='NMP_LOCALITY_AWARE_PLACEMENT',nmp_aggregate_tflops=64,gpu_compute_flops_per_s=g)
    assert changed.timing.local_memory_ms == pytest.approx(current.timing.local_memory_ms)
    assert changed.timing.external_ms == pytest.approx(2 * current.timing.external_ms)
    old_50_lane_bw=(l.slab_count*b.parallel_service_units_per_slab*b.read_payload_bytes_per_service/(b.service_cycle_scale*current.placement.local_access_latency_ns*1e-9))
    new_bw=current.traffic.local_memory_bytes/(current.timing.local_memory_ms*1e-3)
    assert new_bw > old_50_lane_bw

@pytest.mark.parametrize('batch',[1,8,16])
def test_flops_scale_and_more_nmp_compute_never_hurts(inputs,batch):
    l,b,p,w,d,g=inputs; wb=w.model_copy(update={'batch_size':batch}); db=build_m3d_workload_page_demand(wb,l)
    low=evaluate_nmp_locality_case(wb,db,l,p,b,case='NMP_LOCALITY_AWARE_PLACEMENT',nmp_aggregate_tflops=32,gpu_compute_flops_per_s=g)
    high=evaluate_nmp_locality_case(wb,db,l,p,b,case='NMP_LOCALITY_AWARE_PLACEMENT',nmp_aggregate_tflops=128,gpu_compute_flops_per_s=g)
    assert high.timing.tokens_per_s>=low.timing.tokens_per_s
    assert high.traffic.local_weight_read_bytes==pytest.approx(db.total_weight_read_bytes_per_decode_step)
