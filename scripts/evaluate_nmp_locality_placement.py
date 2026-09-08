"""Canonical B=1 dense NMP attention E2E audit (no sweep)."""
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
from om3dthermal.power.nmp_die_power import build_nmp_die_power_map
from om3dthermal.workload import build_m3d_workload_page_demand

def run(output_dir: Path, batch_size: int = 1):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if batch_size > 1:
        from om3dthermal.serving import evaluate_nmp_decode_batch
        base=load_workload_spec(
            ROOT/"configs/workload/llama31_8b_decode_b1_s131072.yaml",
            project_root=ROOT).decode
        workload=base.model_copy(update={"batch_size":batch_size})
        result=evaluate_nmp_decode_batch(workload,project_root=ROOT)
        payload={"summary":result.model_dump(mode="json"),
                 "workload":workload.model_dump(mode="json")}
        output_dir.mkdir(parents=True,exist_ok=True)
        (output_dir/"nmp_locality_placement.json").write_text(
            json.dumps(payload,indent=2),encoding="utf-8")
        return payload
    layout, bandwidth=_architecture(); case=load_case_config(ROOT/"configs/cases/orthogonal_m3d_igzo.yaml"); geo=resolve_case_geometry(case); power=calculate_memory_power(case, read_bandwidth_gbps=case.workload.read_bandwidth_gbps,project_root=ROOT,geometry=geo)
    topology=calculate_m3d_subarray(case.architecture.m3d_subarray,geo.m3d); feol=calculate_feol_route(case.architecture,topology)
    physical=calculate_physical_access_latency(case.architecture.physical_access_latency,feol_route=feol,miv_length_per_layer_um=power.diagnostics['miv_length_per_layer_um'],miv_delay_per_layer_ns=power.diagnostics['miv_delay_per_layer_ns'],miv_status=power.diagnostics['miv_latency_status'],miv_parameter_status=power.diagnostics['miv_resistance_parameter_status'],miv_provenance=power.diagnostics['miv_resistance_provenance'])
    base=load_workload_spec(ROOT/"configs/workload/llama31_8b_decode_b1_s131072.yaml",project_root=ROOT).decode
    experiment=load_experiment_spec(ROOT/"configs/experiment/m3d_igzo_llama31_8b_decode_conditional_v0.yaml",project_root=ROOT)
    gpu=experiment.scenario.effective_compute_flops_per_second
    from om3dthermal.platform import load_platform_spec_file
    platform=load_platform_spec_file(ROOT/"configs/platform/gpu_package_h200_reference.yaml")
    gpu_power=platform.gpu_decode_power
    w=base.model_copy(update={"batch_size":batch_size}); d=build_m3d_workload_page_demand(w,layout)
    baseline=evaluate_nmp_locality_case(w,d,layout,physical,bandwidth,
        case="NON_NMP_GPU",gpu_compute_flops_per_s=gpu)
    hardware=canonical_nmp_hardware(layout.slab_count)
    canonical=evaluate_nmp_locality_case(w,d,layout,physical,bandwidth,
        case="NMP_LOCALITY_AWARE_PLACEMENT",gpu_compute_flops_per_s=gpu)
    bw_die=bandwidth.local_service_groups_per_slab*bandwidth.read_payload_bytes_per_service/(bandwidth.service_cycle_scale*canonical.placement.local_access_latency_ns*1e-9)
    placement=build_performance_balanced_placement(w,d,layout,
        bandwidth_per_die_bytes_per_s=bw_die,compute_per_die_flops_per_s=hardware.peak_flops_per_die)
    activity=evaluate_nmp_die_activity(w,d,layout,bandwidth,
        local_access_latency_ns=canonical.placement.local_access_latency_ns,
        bandwidth_demand_bytes_per_s=bandwidth.coil_bandwidth_bytes_per_s,ownership=placement.ownership)
    power_map=build_nmp_die_power_map(case,power,topology,feol,activity,placement)
    interval=activity.decode_step_interval_ms*1e-3
    nmp_energy=(power_map.aggregate_total_W*interval+activity.softmax_dynamic_energy_j
                +activity.gpu_remaining_dynamic_energy_j+activity.gpu_static_energy_j)
    common_small_bytes=sum(x["gpu_local_total_bytes"] for x in activity.small_ops
                           if x["operator"] not in ("AV_REDUCTION","TOKEN_EMBED_LOOKUP"))
    common_small_ms=common_small_bytes/2.4e12*1e3
    baseline_complete_ms=baseline.timing.total_step_ms
    baseline_s=baseline_complete_ms*1e-3
    # Same accounting boundary: memory dynamic + refresh + GPU dynamic + static.
    primitive=power_map.primitives
    baseline_write=primitive.igzo_weighted_write_pj_per_bit+power.E_vertical_pj_bit+power.E_feol_route_pj_bit+power.E_base_route_pj_bit+power.E_interface_pj_bit
    active_weight=sum(x.weight_read_bytes for x in placement.unit_loads)
    baseline_memory_j=8*((active_weight+d.total_kv_read_bytes_per_decode_step)*power.E_access_total_pj_bit+d.kv_write_bytes_per_decode_step*baseline_write)*1e-12
    baseline_small_j=8*common_small_bytes*gpu_power.e_decode_J_per_bit
    baseline_energy=(baseline_memory_j+(power.P_refresh_W or 0)*baseline_s
        +8*baseline.traffic.external_interface_bytes*gpu_power.e_decode_J_per_bit
        +baseline_small_j+gpu_power.static_power_W*baseline_s)
    loads=placement.unit_loads
    stage_rows=[x for x in activity.stages if "memory_ms" in x]
    transfer_ms=lambda name:sum(x["time_ms"] for x in activity.stages if x["operator"]==name)
    spans_by_operator={name:sorted({x["execution_die_span"] for x in stage_rows if x["operator"]==name})
                       for name in sorted({x["operator"] for x in stage_rows})}
    local_dynamic_j=(power_map.aggregate_memory_read_dynamic_W+power_map.aggregate_memory_write_dynamic_W)*interval
    mac_j=power_map.aggregate_mac_dynamic_W*interval
    residual_interface_j=power_map.aggregate_residual_external_W*interval
    refresh_j=power_map.aggregate_refresh_W*interval
    small_by_operator={}
    for row in activity.small_ops:
        if row["operator"]=="TOKEN_EMBED_LOOKUP": continue
        entry=small_by_operator.setdefault(row["operator"],{"bytes":0.0,"count":0})
        entry["bytes"]+=row["gpu_local_total_bytes"]; entry["count"]+=row["execution_count"]
    for entry in small_by_operator.values():
        entry["MB_per_token"]=entry["bytes"]/w.batch_size/1e6
        entry["ms_per_token"]=entry["bytes"]/2.4e12*1e3/w.batch_size
        entry["J_per_token"]=8*entry["bytes"]*gpu_power.e_decode_J_per_bit/w.batch_size
    rms_all_bytes=small_by_operator["RMSNORM"]["bytes"]+small_by_operator["FINAL_RMSNORM"]["bytes"]
    embedding_transfer_ms=transfer_ms("EMBEDDING_TRANSFER")
    embedding_local_j=8*activity.embedding_local_read_bytes*primitive.local_read_total_pj_per_bit*1e-12
    embedding_transfer_j=8*sum(x["bytes"] for x in activity.handoffs
        if x["producer"]=="TOKEN_EMBED_LOOKUP")*(primitive.long_feol_pj_per_bit+primitive.interface_pj_per_bit)*1e-12
    summary={
        "qk_flops_per_token":sum(x.nmp_flops for x in loads if x.unit.operator_type=="ATTENTION_QK")/w.batch_size,
        "av_flops_per_token":sum(x.nmp_flops for x in loads if x.unit.operator_type=="ATTENTION_AV")/w.batch_size,
        "weight_nmp_flops_per_token":sum(x.nmp_flops for x in loads if x.unit.weight_bytes)/w.batch_size,
        "total_nmp_flops_per_token":sum(x.nmp_flops for x in loads)/w.batch_size,
        "local_weight_bytes_per_token":sum(x.weight_read_bytes for x in loads)/w.batch_size,
        "active_weight_GB_per_token":sum(x.weight_read_bytes for x in loads)/w.batch_size/1e9,
        "resident_weight_GB":sum(x.unit.weight_bytes for x in loads)/1e9,
        "local_kv_bytes_per_token":sum(x.kv_read_bytes+x.kv_write_bytes for x in loads)/w.batch_size,
        "kv_GB_per_token":sum(x.kv_read_bytes+x.kv_write_bytes for x in loads)/w.batch_size/1e9,
        "score_MB_per_token":activity.score_bytes/w.batch_size/1e6,
        "probability_MB_per_token":activity.probability_bytes/w.batch_size/1e6,
        "partial_MB_per_token":activity.partial_bytes/w.batch_size/1e6,
        "attention_boundary_MB_per_token":activity.attention_boundary_bytes/w.batch_size/1e6,
        "boundary_MB_per_token":activity.residual_boundary_bytes/w.batch_size/1e6,
        "effective_boundary_BW_bytes_per_s":activity.transfer["bandwidth_actual_bytes_per_s"],
        "boundary_ms_per_token":activity.boundary_time_ms/w.batch_size,
        "softmax_ms_per_token":activity.softmax_time_ms/w.batch_size,
        "score_transfer_ms_per_token":transfer_ms("SCORE_TRANSFER")/w.batch_size,
        "probability_transfer_ms_per_token":transfer_ms("PROBABILITY_TRANSFER")/w.batch_size,
        "av_partial_transfer_ms_per_token":transfer_ms("PARTIAL_TRANSFER")/w.batch_size,
        "nmp_compute_ms_per_token":sum(x["compute_ms"] for x in activity.stages if "compute_ms" in x)/w.batch_size,
        "nmp_local_memory_ms_per_token":sum(x["memory_ms"] for x in activity.stages if "memory_ms" in x)/w.batch_size,
        "operator_execution_die_spans":spans_by_operator,
        "mean_exec_die_span":activity.mean_exec_die_span,
        "median_exec_die_span":activity.median_exec_die_span,
        "max_exec_die_span":activity.max_exec_die_span,
        "aggregate_local_bandwidth_bytes_per_s":activity.aggregate_local_bandwidth_bytes_per_s,
        "realized_effective_local_bandwidth_bytes_per_s":activity.realized_effective_local_bandwidth_bytes_per_s,
        "decode_ms_per_token":activity.decode_step_interval_ms/w.batch_size,
        "tokens_per_s":w.batch_size/interval,"J_per_token":nmp_energy/w.batch_size,"tokens_per_J":w.batch_size/nmp_energy,
        "nmp_local_memory_dynamic_J_per_token":local_dynamic_j/w.batch_size,
        "mac_J_per_token":mac_j/w.batch_size,
        "residual_interface_J_per_token":residual_interface_j/w.batch_size,
        "softmax_dynamic_J_per_token":activity.softmax_dynamic_energy_j/w.batch_size,
        "gpu_remaining_small_ops":small_by_operator,
        "rmsnorm_including_final_count":small_by_operator["RMSNORM"]["count"]+small_by_operator["FINAL_RMSNORM"]["count"],
        "rmsnorm_including_final_MB_per_token":rms_all_bytes/w.batch_size/1e6,
        "gpu_remaining_local_MB_per_token":activity.gpu_remaining_local_bytes/w.batch_size/1e6,
        "gpu_remaining_ms_per_token":activity.gpu_remaining_time_ms/w.batch_size,
        "gpu_remaining_dynamic_J_per_token":activity.gpu_remaining_dynamic_energy_j/w.batch_size,
        "embedding_read_bytes_per_token":activity.embedding_local_read_bytes/w.batch_size,
        "embedding_latency_ms_per_token":(activity.embedding_local_time_ms+embedding_transfer_ms)/w.batch_size,
        "embedding_dynamic_J_per_token":(embedding_local_j+embedding_transfer_j)/w.batch_size,
        "gpu_static_J_per_token":activity.gpu_static_energy_j/w.batch_size,
        "refresh_J_per_token":refresh_j/w.batch_size,
        "baseline_decode_ms_per_token":baseline_complete_ms/w.batch_size,
        "baseline_tokens_per_s":w.batch_size/(baseline_complete_ms*1e-3),
        "baseline_gpu_remaining_ms_per_token":common_small_ms/w.batch_size,
        "baseline_gpu_remaining_local_MB_per_token":common_small_bytes/w.batch_size/1e6,
        "baseline_gpu_remaining_dynamic_J_per_token":baseline_small_j/w.batch_size,
        "baseline_J_per_token":baseline_energy/w.batch_size,"baseline_tokens_per_J":w.batch_size/baseline_energy,
        "speedup":baseline_complete_ms/activity.decode_step_interval_ms,
        "energy_efficiency_gain":baseline_energy/nmp_energy,
    }
    layer0=[x for x in stage_rows if x["layer"]==0]
    attention_owner_counts={str(layer):{
        "QK_by_request":{str(request):len(owners) for request,owners in row["qk_request_owners"].items()},
        "AV_by_request":{str(request):len(owners) for request,owners in row["av_request_owners"].items()}}
        for layer,row in activity.attention_layers.items()}
    payload=dict(summary=summary,activity=activity.as_dict(),placement=placement.as_dict(),
        power_map=power_map.as_dict(),baseline=baseline.as_dict(),workload=w.model_dump(),
        diagnostics={"layer0_stage_example":layer0,"attention_owner_counts_by_layer":attention_owner_counts,
            "realized_bandwidth_definition":"total active local bytes divided by summed memory-dominated stage latency",
            "decode_interval_definition":"complete token: embedding row lookup through greedy argmax",
            "attention_scaling":"QK/sqrt(d_head) included in GPU Softmax preprocessing; no separate throughput model",
            "ledger_locations":{"handoffs":"activity.handoffs","small_ops":"activity.small_ops"}})
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"nmp_locality_placement.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,default=ROOT/"runs/nmp_attention_nominal")
    p.add_argument("--batch-size",type=int,default=1)
    args=p.parse_args()
    print(json.dumps(run(args.output_dir,args.batch_size)["summary"],indent=2))

if __name__ == "__main__":
    main()
