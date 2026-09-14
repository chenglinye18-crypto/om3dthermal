"""B=1/8 formal architecture semantics, separate from frozen physical execution.

GPU execution uses the conventional workload roofline and the existing M3D
hierarchical streaming service closure. NMP checkpoints remain physical operator
schedules. Neither this module nor the selector constructs a tile placement.
"""
from copy import deepcopy
from functools import lru_cache
import math

from .workload_matrix import inputs, setup, timing
from .mixed_phase_e2e import resolve_m3d_architecture_backend
from om3dthermal.workload import evaluate_llm_decode, evaluate_cached_prefix_incremental_prefill
from om3dthermal.power.feol_energy import FEOLEnergyModel
from om3dthermal.architecture.feol_floorplan import resolve_feol_floorplan


ARCHITECTURES = ("HBM_GPU", "M3D_GPU", "M3D_MAC_NMP")
SELECTOR_STATUS = "PRE_DISPATCH_NOT_IN_INFERENCE_CRITICAL_PATH"
GPU_PLACEMENT = "NOT_APPLICABLE_GPU_EXECUTION"


def require_primary(batch):
    if batch not in (1, 8):
        raise ValueError("Only B=1/8; B32 is frozen and outside this rebaseline")


@lru_cache(maxsize=1)
def gpu_memory_closure(root):
    """Full-cycle closure: MAT/MIV/FEOL occur once in the lane service cycle.

    This is the existing FIRST_ORDER_PARALLEL_MEMORY_SERVICE_MODEL, using
    its full-capacity average cycle. 50 external service lanes/slab, NOT 70
    NMP-local groups, and no eightfold layer parallelism. FEOL route latency
    is already in this cycle, so there is no second independent startup.
    This aggregate GPU memory model does not inherit NMP row/tile placement.
    """
    architecture = resolve_m3d_architecture_backend(root)
    _, _, platform, _, thermal = setup(root)
    c = architecture.bandwidth
    internal = c.internal_bandwidth_average_bytes_per_s
    boundary = c.coil_bandwidth_bytes_per_s
    gpu = platform.gpu_decode_power.peak_memory_bandwidth_bytes_per_s
    return dict(thermal_Bps=thermal, internal_Bps=internal, boundary_Bps=boundary,
                GPU_Bps=gpu, effective_Bps=min(thermal, internal, boundary, gpu),
                startup_s=0.0, service_cycle_ns=c.average_service_cycle_ns,
                service_cycle_scale=c.service_cycle_scale,
                parallel_service_lanes=c.total_parallel_service_units,
                payload_bytes_per_service=c.read_payload_bytes_per_service,
                model=c.internal_service_model,
                startup_status="MAT_MIV_FEOL_INCLUDED_ONCE_IN_SERVICE_CYCLE",
                boundary_status="EXISTING_AGGREGATE_GPU_MEMORY_SERVICE_LANES",
                source="power/memory_bandwidth.py:derive_architecture_bandwidth")


def gpu_roofline(compute_s, memory_bytes, closure):
    memory_s = closure['startup_s'] + memory_bytes / closure['effective_Bps']
    return max(compute_s, memory_s)


def candidate_timing(name, wid, batch, root):
    require_primary(batch)
    w, cw, spec = inputs(name, wid, batch, root)
    _, _, platform, _, _ = setup(root)
    c = gpu_memory_closure(root)
    pre = evaluate_cached_prefix_incremental_prefill(
        spec.prefill_input(batch_size=batch, prompt_length=cw.prompt),
        cached_history_tokens=cw.history)
    p = platform.gpu_prefill_compute
    pre_compute = ((pre.linear_flops + pre.lm_head_flops)/(p.large_gemm_effective_tflops*1e12)
                   + pre.attention_flops/(p.causal_attention_effective_tflops*1e12))
    steps = []
    for context in cw.contexts:
        m = evaluate_llm_decode(w.model_copy(update={'context_length': context}))
        read, write = m.read_bytes_per_token*batch, m.write_bytes_per_token*batch
        flops = m.flops_per_token*batch
        compute = flops/platform.gpu_compute_power.peak_compute_BF16_dense_flops_per_s
        steps.append(dict(context=context, read_bytes=read, write_bytes=write,
                          weight_bytes=m.weight_read_bytes_per_token*batch,
                          KV_read_bytes=m.kv_read_bytes_per_token*batch,
                          KV_write_bytes=m.kv_write_bytes_per_token*batch,
                          flops=flops, compute_s=compute,
                          latency_s=gpu_roofline(compute, read+write, c)))
    pre_s = gpu_roofline(pre_compute, pre.total_memory_bytes, c)
    return dict(ledger=pre.model_dump(), prefill_compute_s=pre_compute, closure=c,
                steps=steps, timing=timing(pre_s, [s['latency_s'] for s in steps], batch*cw.generated))


def corrected_gpu_result(old, candidate, root):
    """Change only timing-dependent energy; valid physical event ledger is reused."""
    require_primary(old['summary']['batch_size'])
    result = deepcopy(old)
    s, e = result['summary'], result['energy']
    events = result['events']
    steps = candidate['steps']
    # The HBM semantic ledger counts active weights and KV read/append. The
    # old physical GPU bulk events already conserve exactly these bytes.
    for key, expected in [('array_read_bits', sum(x['read_bytes'] for x in steps)*8),
                          ('array_write_bits', sum(x['write_bytes'] for x in steps)*8),
                          ('interface_bits', sum(x['read_bytes']+x['write_bytes'] for x in steps)*8)]:
        if events[key] != expected:
            raise ValueError(f"GPU semantic traffic mismatch: {key}")
    if result['prefill_ledger'] != candidate['ledger']:
        raise ValueError("Prefill ledger mismatch")
    if any(events[k] for k in ('mac_operations', 'router_bit_traversals', 'noc_link_bit_um',
                              'sa_to_tile_bit_um', 'root_to_tile_bit_um',
                              'fp32_reduction_adds', 'sram_read32_accesses', 'sram_write32_accesses')):
        raise ValueError("GPU execution contains NMP activity")
    s.update(candidate['timing'], system='M3D_GPU', selected_executor='GPU',
             placement_policy=GPU_PLACEMENT, CPA_moves=0, CPA_optimizer_runtime=0,
             optimizer_runtime_s=0, accepted_moves=0, moved_fraction=0, resident_moved_fraction=0,
             effective_bandwidth_TBps=candidate['closure']['effective_Bps']/1e12,
             NMP_MAC_status='INACTIVE', NMP_Fabric_status='INACTIVE', NMP_NoC_status='INACTIVE',
             NMP_reduction_status='INACTIVE', NMP_background_status='INACTIVE')
    energy_model = FEOLEnergyModel(resolve_feol_floorplan(root), setup(root)[2])
    dec = energy_model.account(events, s['decode_s'], phase='decode', policy='NO_NMP')
    # No other existing Prefill event or coefficient changes.
    e['prefill_gpu_static_J'] = s['prefill_s']*setup(root)[2].gpu_decode_power.static_power_W
    e['prefill_J'] = sum(v for k,v in e.items() if k.startswith('prefill_') and k.endswith('_J') and k not in ('prefill_J','prefill_tokens_per_J'))
    e.update({'decode_'+k:v for k,v in dec['components'].items()})
    generated = s['decode_generated_tokens']
    total = e['prefill_J']+dec['total_J']
    e.update(decode_J=dec['total_J'], decode_J_per_token=dec['total_J']/generated,
             decode_tokens_per_J=generated/dec['total_J'], E2E_J=total,
             E2E_J_per_token=total/generated, E2E_tokens_per_J=generated/total,
             average_decode_power_W=dec['average_power_W'])
    result['traffic']['external_realized_TBps'] = candidate['closure']['effective_Bps']/1e12
    result['component_sums'] = dict(
        ARRAY=sum(x['read_bytes']+x['write_bytes'] for x in steps)/candidate['closure']['internal_Bps'],
        EXTERNAL_BOUNDARY=sum(x['read_bytes']+x['write_bytes'] for x in steps)/min(
            candidate['closure'][k] for k in ('thermal_Bps','boundary_Bps','GPU_Bps')),
        GPU_COMPUTE=sum(x['compute_s'] for x in steps), LOCAL_FABRIC=0., MAC=0., INTER_REGION_NOC=0.)
    result['cpa_audit'] = []
    result['legacy_fingerprint'] = result.pop('fingerprint')
    return result


def adaptive_result(gpu, nmp_cpa):
    """Pre-dispatch modeled minimum Decode latency; exact selected result copy."""
    a, b = gpu['summary'], nmp_cpa['summary']
    require_primary(a['batch_size'])
    if any(a[k] != b[k] for k in ('model','workload_id','batch_size','H','P','G')):
        raise ValueError("Candidate operating points differ")
    if a['status'] != 'EVALUATED' or b['status'] != 'EVALUATED':
        raise ValueError("Both candidates must pass the existing capacity gate")
    if b.get('placement_policy') != 'CRITICAL_PATH_AWARE':
        raise ValueError("NMP candidate must use CPA; Uniform is only an ablation")
    if a['prefill_s'] != b['prefill_s']:
        raise ValueError("Both executors must have identical GPU Prefill latency")
    if not all(math.isfinite(s['decode_s']) and s['decode_s'] > 0 for s in (a,b)):
        raise ValueError("Candidate Decode costs must be finite and positive")
    executor = 'GPU' if a['decode_s'] <= b['decode_s'] else 'NMP_CPA'
    result = deepcopy(gpu if executor == 'GPU' else nmp_cpa)
    s = result['summary']
    s.update(system='M3D_MAC_NMP', selected_executor=executor,
             selector_objective='MINIMUM_DECODE_LATENCY', selector_overhead_status=SELECTOR_STATUS,
             selector_latency_s=0., selector_energy_J=0.)
    if executor == 'GPU':
        s.update(placement_policy=GPU_PLACEMENT, CPA_moves=0, CPA_optimizer_runtime=0)
        result['cpa_audit'] = []
    else:
        s.update(placement_policy='CRITICAL_PATH_AWARE', CPA_moves=s['accepted_moves'],
                 CPA_optimizer_runtime=s['optimizer_runtime_s'])
    return result
