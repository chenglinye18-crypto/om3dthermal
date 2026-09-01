"""Canonical A-final NMP/locality placement sweep."""
from __future__ import annotations
import argparse, json
from pathlib import Path
try:
    from evaluate_die_local_placement import ROOT, _architecture
except ModuleNotFoundError:  # imported as scripts.evaluate_nmp_locality_placement
    from scripts.evaluate_die_local_placement import ROOT, _architecture
from om3dthermal.experiment import load_experiment_spec, load_workload_spec
from om3dthermal.placement import evaluate_nmp_locality_case
from om3dthermal.placement.nmp_load_balance import (build_locality_only_placement,
    build_performance_balanced_placement,remaining_external_bytes_for_ownership)
from om3dthermal.power import calculate_memory_power, calculate_physical_access_latency, load_case_config, resolve_case_geometry
from om3dthermal.power.feol_route import calculate_feol_route
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.power.nmp_die_activity import canonical_nmp_hardware, evaluate_nmp_die_activity
from om3dthermal.workload import build_m3d_workload_page_demand

def run(output_dir: Path):
    layout, bandwidth=_architecture(); case=load_case_config(ROOT/"configs/cases/orthogonal_m3d_igzo.yaml"); geo=resolve_case_geometry(case); power=calculate_memory_power(case,project_root=ROOT,geometry=geo)
    topology=calculate_m3d_subarray(case.architecture.m3d_subarray,geo.m3d); feol=calculate_feol_route(case.architecture.feol_route,topology)
    physical=calculate_physical_access_latency(case.architecture.physical_access_latency,feol_route=feol,miv_length_per_layer_um=power.diagnostics['miv_length_per_layer_um'],miv_delay_per_layer_ns=power.diagnostics['miv_delay_per_layer_ns'],miv_status=power.diagnostics['miv_latency_status'],miv_parameter_status=power.diagnostics['miv_resistance_parameter_status'],miv_provenance=power.diagnostics['miv_resistance_provenance'])
    base=load_workload_spec(ROOT/"configs/workload/llama31_8b_decode_b1_s131072.yaml",project_root=ROOT).decode
    gpu=load_experiment_spec(ROOT/"configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml",project_root=ROOT).scenario.effective_compute_flops_per_second
    rows=[]
    for n in (1,8,16):
        w=base.model_copy(update={'batch_size':n}); d=build_m3d_workload_page_demand(w,layout)
        baseline=evaluate_nmp_locality_case(w,d,layout,physical,bandwidth,case='NON_NMP_GPU',nmp_aggregate_tflops=None,gpu_compute_flops_per_s=gpu)
        points=[]
        for p in (32.,64.,128.):
            naive=evaluate_nmp_locality_case(w,d,layout,physical,bandwidth,case='NMP_NAIVE',nmp_aggregate_tflops=p,gpu_compute_flops_per_s=gpu)
            local=evaluate_nmp_locality_case(w,d,layout,physical,bandwidth,case='NMP_LOCALITY_AWARE_PLACEMENT',nmp_aggregate_tflops=p,gpu_compute_flops_per_s=gpu)
            points.append({'nmp_aggregate_tflops':p,'naive':naive.as_dict(),'locality_aware':local.as_dict(),
                'nmp_gain':naive.timing.tokens_per_s/baseline.timing.tokens_per_s,
                'placement_incremental_gain':local.timing.tokens_per_s/naive.timing.tokens_per_s,
                'combined_A_gain':local.timing.tokens_per_s/baseline.timing.tokens_per_s,
                'serial_stage_bound': {
                    'nmp_gain': naive.timing.tokens_per_s_serial / baseline.timing.tokens_per_s_serial,
                    'placement_incremental_gain': local.timing.tokens_per_s_serial / naive.timing.tokens_per_s_serial,
                    'combined_A_gain': local.timing.tokens_per_s_serial / baseline.timing.tokens_per_s_serial,
                }})
        hardware=canonical_nmp_hardware(layout.slab_count)
        canonical_naive=evaluate_nmp_locality_case(w,d,layout,physical,bandwidth,case='NMP_NAIVE',nmp_aggregate_tflops=hardware.aggregate_peak_flops/1e12,gpu_compute_flops_per_s=gpu)
        canonical=evaluate_nmp_locality_case(w,d,layout,physical,bandwidth,case='NMP_LOCALITY_AWARE_PLACEMENT',nmp_aggregate_tflops=hardware.aggregate_peak_flops/1e12,gpu_compute_flops_per_s=gpu)
        hardware_bw_die=(bandwidth.local_service_groups_per_die*bandwidth.read_payload_bytes_per_service/(bandwidth.service_cycle_scale*canonical.placement.local_access_latency_ns*1e-9))
        locality_placement=build_locality_only_placement(w,d,layout,
            bandwidth_per_die_bytes_per_s=hardware_bw_die,compute_per_die_flops_per_s=hardware.peak_flops_per_die)
        external_bytes=remaining_external_bytes_for_ownership(locality_placement.unit_loads,locality_placement.ownership)
        external_ms=external_bytes/bandwidth.coil_bandwidth_bytes_per_s*1e3
        activity=evaluate_nmp_die_activity(w,d,layout,bandwidth,local_access_latency_ns=canonical.placement.local_access_latency_ns,external_boundary_time_ms=external_ms,ownership=locality_placement.ownership)
        diagnostic_tps=n/(activity.decode_step_interval_ms*1e-3)
        balanced_placement=build_performance_balanced_placement(w,d,layout,
            bandwidth_per_die_bytes_per_s=hardware_bw_die,
            compute_per_die_flops_per_s=hardware.peak_flops_per_die)
        balanced_activity=evaluate_nmp_die_activity(w,d,layout,bandwidth,
            local_access_latency_ns=canonical.placement.local_access_latency_ns,
            external_boundary_time_ms=external_ms,
            ownership=balanced_placement.ownership)
        balanced_tps=n/(balanced_activity.decode_step_interval_ms*1e-3)
        ideal_local_ms=(d.total_read_bytes_per_decode_step+d.kv_write_bytes_per_decode_step)/(hardware_bw_die*layout.slab_count)*1e3
        ideal_compute_ms=canonical.timing.nmp_compute_ms
        ideal_step_ms=max(ideal_local_ms,ideal_compute_ms)+external_ms
        ideal_tps=n/(ideal_step_ms*1e-3)
        rows.append({'requests':n,'working_set_bytes':d.allocated_page_bytes,'logical_working_set_bytes':d.logical_working_set_bytes,'non_nmp_gpu':baseline.as_dict(),'points':points,
            'canonical_nmp_hardware':hardware.__dict__,'canonical_die_activity':activity.as_dict(),
            'IDEAL_AGGREGATE_UPPER_BOUND': {
                'timing_semantics': 'IDEAL_AGGREGATE_BALANCED_UPPER_BOUND',
                'non_nmp_step_ms': baseline.timing.total_step_ms,
                'nmp_local_memory_ms': ideal_local_ms,
                'nmp_compute_ms': canonical.timing.nmp_compute_ms,
                'nmp_remaining_external_ms': external_ms,
                'remaining_external_bytes': external_bytes,
                'nmp_step_ms': ideal_step_ms,
                'nmp_gain': canonical_naive.timing.tokens_per_s/baseline.timing.tokens_per_s,
                'placement_incremental_gain': canonical.timing.tokens_per_s/canonical_naive.timing.tokens_per_s,
                'combined_A_gain': ideal_tps/baseline.timing.tokens_per_s,
            },
            'LOCALITY_ONLY_BASELINE': {
                'timing_semantics': 'REALIZED_DIE_LEVEL_LOCALITY_ONLY',
                'combined_A_gain': diagnostic_tps/baseline.timing.tokens_per_s,
                'placement': locality_placement.as_dict(),
                'remaining_external_bytes': external_bytes,
                'activity': activity.as_dict(),
            },
            'A_FINAL_CANONICAL_GAIN': {
                'timing_semantics': 'REALIZED_DIE_LEVEL_CANONICAL__PERFORMANCE_BALANCED',
                'combined_A_gain': balanced_tps/baseline.timing.tokens_per_s,
                'nmp_step_ms': balanced_activity.decode_step_interval_ms,
                'placement': balanced_placement.as_dict(),
                'activity': balanced_activity.as_dict(),
                'remaining_external_bytes': external_bytes,
            }})
    payload={'model':'A_FINAL_NMP_LOCALITY_AWARE_PLACEMENT','physical_die_count':layout.slab_count,'die_semantics':'ARCHITECTURE_DEFINED_ONE_SLAB_PER_PHYSICAL_DIE','DIRECT_DIE_TO_DIE_COMMUNICATION':'FORBIDDEN','canonical_overlap':'CONSERVATIVE_NO_OVERLAP','rows':rows}
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/'nmp_locality_placement.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); return payload
def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'runs/nmp_locality_placement'); x=run(p.parse_args().output_dir)
    for r in x['rows']:
        for q in r['points']: print(f"N={r['requests']} P={q['nmp_aggregate_tflops']:.0f} NMP={q['nmp_gain']:.3f}x placement={q['placement_incremental_gain']:.3f}x combined={q['combined_A_gain']:.3f}x")
if __name__=='__main__': main()
