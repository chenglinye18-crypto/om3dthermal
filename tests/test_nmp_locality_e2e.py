"""A-final path and operator-locality invariants."""
from __future__ import annotations
import pytest
from dataclasses import replace
from scripts.evaluate_die_local_placement import ROOT, _architecture
from om3dthermal.experiment import load_experiment_spec, load_workload_spec
from om3dthermal.placement import evaluate_nmp_locality_case, independent_physical_die_count
from om3dthermal.power.memory_bandwidth import resolve_effective_bandwidth
from om3dthermal.power import calculate_memory_power, calculate_physical_access_latency, load_case_config, resolve_case_geometry
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.workload import build_m3d_workload_page_demand

@pytest.fixture(scope='module')
def inputs():
    layout,bw=_architecture(); c=load_case_config(ROOT/'configs/cases/orthogonal_m3d_igzo.yaml'); g=resolve_case_geometry(c); power=calculate_memory_power(c, read_bandwidth_gbps=c.workload.read_bandwidth_gbps,project_root=ROOT,geometry=g); top=calculate_m3d_subarray(c.architecture.m3d_subarray,g.m3d); feol=calculate_feol_route(c.architecture,top)
    phy=calculate_physical_access_latency(c.architecture.physical_access_latency,feol_route=feol,miv_length_per_layer_um=power.diagnostics['miv_length_per_layer_um'],miv_delay_per_layer_ns=power.diagnostics['miv_delay_per_layer_ns'],miv_status=power.diagnostics['miv_latency_status'],miv_parameter_status=power.diagnostics['miv_resistance_parameter_status'],miv_provenance=power.diagnostics['miv_resistance_provenance'])
    w=load_workload_spec(ROOT/'configs/workload/llama31_8b_decode_b1_s131072.yaml',project_root=ROOT).decode; d=build_m3d_workload_page_demand(w,layout); gpu=load_experiment_spec(ROOT/'configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml',project_root=ROOT).scenario.effective_compute_flops_per_second
    return layout,bw,phy,w,d,gpu

def test_path_semantics_and_locality(inputs):
    l,b,p,w,d,g=inputs
    non=evaluate_nmp_locality_case(w,d,l,p,b,case="NON_NMP_GPU",gpu_compute_flops_per_s=g)
    local=evaluate_nmp_locality_case(w,d,l,p,b,case="NMP_LOCALITY_AWARE_PLACEMENT",gpu_compute_flops_per_s=g)
    assert non.traffic.weight_bulk_external_bytes > 0 and non.traffic.kv_bulk_external_bytes > 0
    assert local.traffic.weight_bulk_external_bytes == local.traffic.kv_bulk_external_bytes == 0
    assert local.traffic.direct_die_to_die_bytes == 0
    assert not local.placement.long_feol_edge_included
    assert local.timing.external_bandwidth_bytes_per_s == non.timing.external_bandwidth_bytes_per_s == 2.4e12
    changed=evaluate_nmp_locality_case(w,d,l,p,replace(b,coil_bandwidth_bytes_per_s=1.2e12),case="NMP_LOCALITY_AWARE_PLACEMENT",gpu_compute_flops_per_s=g)
    assert changed.timing.local_memory_ms == local.timing.local_memory_ms
    assert changed.timing.external_ms == pytest.approx(2*local.timing.external_ms)


@pytest.mark.parametrize("batch,context,kv_bits",[(1,131072,16),(2,17,8),(1,0,16)])
def test_workload_ledger_formulas(inputs,batch,context,kv_bits):
    from om3dthermal.workload.dense_decode_ledger import build_dense_decode_placement_units, boundary_bytes_per_die
    from om3dthermal.workload.llm_decode import evaluate_llm_decode
    l,b,p,w,d,g=inputs
    w=w.model_copy(update=dict(batch_size=batch,context_length=context,kv_bits=kv_bits))
    metrics=evaluate_llm_decode(w); units=build_dense_decode_placement_units(w)
    qk=[u for u in units if u.operator_type=="ATTENTION_QK"]
    av=[u for u in units if u.operator_type=="ATTENTION_AV"]
    expected=2*batch*w.n_heads_q*context*w.d_head*w.n_layers
    assert sum(u.local_flops for u in qk) == sum(u.local_flops for u in av) == expected
    assert sum(u.local_flops*(batch if u.placement_scope=="SHARED_BATCH" else 1) for u in units) == batch*metrics.flops_per_token
    assert sum(u.weight_bytes for u in units) == metrics.weight_footprint_bytes
    assert sum(u.kv_bytes for u in units) == batch*metrics.kv_read_bytes_per_token
    assert sum(u.kv_write_bytes for u in units) == batch*metrics.kv_write_bytes_per_token
    # Two active AV dies, each returns the entire batch hidden partial vector.
    ownership=tuple((0,1) for u in units)
    boundary=sum(boundary_bytes_per_die(units,ownership,2))
    assert boundary-sum(2*(2*u.activation_input_bytes+u.partial_output_bytes) for u in units if u.placement_scope=="SHARED_BATCH") == 2*batch*w.n_heads_q*context*2*w.n_layers+2*batch*w.d_model*4*w.n_layers


def test_demand_cap_uses_existing_resolver(inputs,monkeypatch):
    import om3dthermal.power.nmp_die_activity as module
    from om3dthermal.platform import resolve_local_memory_gpu_transfer
    calls=[]
    def record(**kwargs):
        calls.append(kwargs)
        return resolve_local_memory_gpu_transfer(**kwargs)
    monkeypatch.setattr(module,"resolve_local_memory_gpu_transfer",record)
    l,b,p,w,d,g=inputs
    result=evaluate_nmp_locality_case(w,d,l,p,b,case="NMP_LOCALITY_AWARE_PLACEMENT",gpu_compute_flops_per_s=g,external_bandwidth_cap_bytes_per_s=1e12)
    assert calls and calls[0]["bandwidth_demand_bytes_per_s"]==1e12
    assert result.timing.external_bandwidth_bytes_per_s == 1e12
    assert result.timing.external_ms == pytest.approx(result.traffic.external_interface_bytes/1e12*1e3)
