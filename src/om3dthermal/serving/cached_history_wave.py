"""H+P+G run-to-completion waves; the legacy fixed-context API is untouched."""
from copy import deepcopy
import math
import numpy as np

from om3dthermal.workload import evaluate_cached_prefix_incremental_prefill, evaluate_llm_decode
from .workspace import evaluate_prefill_workspace, evaluate_decode_workspace
from .workload_matrix import grace_residency


def high_water(spec, history, prompt, generated, batch, workspace):
    pre = evaluate_prefill_workspace(spec, batch_size=batch, context_length=prompt, config=workspace).peak_bytes
    dec = evaluate_decode_workspace(spec, batch_size=batch, context_length=history+prompt+generated-1, config=workspace).peak_bytes
    scratch = math.ceil(max(pre, dec)/32)*32
    kv = 2*spec.n_layers*spec.n_heads_kv*(spec.d_model//spec.n_heads_q)*(spec.kv_bits//8)
    return spec.n_param*(spec.weight_bits//8)+batch*(history+prompt+generated)*kv+scratch, scratch


def resident_limit(spec, cw, workspace, capacity):
    kv = 2*spec.n_layers*spec.n_heads_kv*(spec.d_model//spec.n_heads_q)*(spec.kv_bits//8)
    upper = max(0, int((capacity-spec.n_param*(spec.weight_bits//8))//((cw.history+cw.prompt+cw.generated)*kv)))
    return next((b for b in range(upper, 0, -1) if high_water(spec, cw.history, cw.prompt, cw.generated, b, workspace)[0] <= capacity), 0)


def latency_metrics(requests, generated):
    values = dict(TTFT=[r['queue_delay']+r['admission_delay']+r['prefill_latency']+r['first_decode_step'] for r in requests],
                  TPOT=[r['decode_duration']/generated for r in requests],
                  completion_latency=[r['completion_time'] for r in requests])
    result = {}
    for name, series in values.items():
        result['mean_'+name] = float(np.mean(series))
        result['P95_'+name] = float(np.percentile(series, 95, method='linear'))
        if name != 'TPOT': result['max_'+name] = max(series)
    return result


def hbm_service(spec, cw, batch, workspace, platform, hbm, *, offload):
    """Canonical incremental ledger, growing decode, and optimistic max overlap."""
    w = spec.decode_input(batch_size=batch, context_length=cw.history+cw.prompt+cw.generated)
    peak, scratch = high_water(spec, cw.history, cw.prompt, cw.generated, batch, workspace)
    if not offload and peak > hbm.capacity_bytes:
        raise ValueError('Local HBM service exceeds high-water capacity')
    pre = evaluate_cached_prefix_incremental_prefill(spec.prefill_input(batch_size=batch, prompt_length=cw.prompt), cached_history_tokens=cw.history)
    objects = grace_residency(w, cw, pre, hbm.capacity_bytes, scratch) if offload else []
    remote = lambda o: o.size-o.local
    gp, pc, host = platform.gpu_decode_power, platform.gpu_compute_power, platform.host_offload
    pcomp = platform.gpu_prefill_compute
    pre_compute = (pre.linear_flops+pre.lm_head_flops)/(pcomp.large_gemm_effective_tflops*1e12)+pre.attention_flops/(pcomp.causal_attention_effective_tflops*1e12)
    energy, traffic, steps = {}, dict(HBM_read_bytes=0., HBM_write_bytes=0., host_read_bytes=0., host_write_bytes=0.), []

    def phase(name, read, write, host_read, host_write, compute, dynamic):
        local_read, local_write = read-host_read, write-host_write
        if min(local_read, local_write) < 0: raise ValueError('Negative local traffic')
        duration = max(compute, (local_read+local_write)/hbm.sustained_bandwidth_bytes_per_s,
                       (host_read+host_write)/host.direct_effective_bandwidth_bytes_per_second)
        components = dict(gpu_dynamic=dynamic, gpu_static=duration*gp.static_power_W,
                          hbm_read=local_read*8*hbm.read_energy_pJ_per_bit*1e-12,
                          hbm_write=local_write*8*1.9955e-12,
                          host_memory=(host_read+host_write)*8*host.memory_dynamic_J_per_bit,
                          c2c=(host_read+host_write)*8*host.link_dynamic_J_per_bit)
        for k, v in components.items(): energy[name+'_'+k+'_J'] = energy.get(name+'_'+k+'_J', 0)+v
        for k, v in zip(traffic, (local_read, local_write, host_read, host_write)): traffic[k] += v
        return duration

    pre_s = phase('prefill', pre.total_read_bytes, pre.total_write_bytes,
        sum(remote(o)*o.prefill_reads for o in objects), sum(remote(o) for o in objects if o.write_phase=='prefill'),
        pre_compute, pre.total_flops*(pc.e_compute_dynamic_J_per_FLOP_min+pc.e_compute_dynamic_J_per_FLOP_max)/2)
    for j, context in enumerate(cw.contexts):
        d = evaluate_llm_decode(w.model_copy(update={'context_length': context}))
        read, write = d.read_bytes_per_token*batch, d.write_bytes_per_token*batch
        hr = sum(remote(o) for o in objects if ('weights' in o.name and o.decode_reads) or ('KV' in o.name and o.birth<j))
        hw = sum(remote(o) for o in objects if o.write_phase=='decode' and o.birth==j)
        duration = phase('decode', read, write, hr, hw, d.flops_per_token*batch/pc.peak_compute_BF16_dense_flops_per_s,
                         (read+write)*8*gp.e_decode_J_per_bit)
        steps.append(dict(context=context, latency_s=duration))
    return dict(prefill_s=pre_s, decode_s=sum(s['latency_s'] for s in steps), steps=steps,
                energy=energy, traffic=traffic, prefill_ledger=pre.model_dump(), high_water_bytes=peak)


def evaluate_hbm_policy(spec, cw, batch, workspace, platform, hbm, *, policy):
    if policy not in ('HBM_HOST_OFFLOAD', 'HBM_RESIDENT_WAVE'): raise ValueError(policy)
    limit = resident_limit(spec, cw, workspace, hbm.capacity_bytes)
    if policy == 'HBM_RESIDENT_WAVE' and not limit:
        return dict(policy=policy, status='CAPACITY_INFEASIBLE', safe_resident_batch=0, wave_sizes=[], num_waves=0)
    size = batch if policy == 'HBM_HOST_OFFLOAD' else min(limit, batch)
    sizes = [min(size, batch-i) for i in range(0, batch, size)]
    requests, waves, energy = [], [], {}
    traffic = dict(HBM_read_bytes=0., HBM_write_bytes=0., host_read_bytes=0., host_write_bytes=0., admission_bytes=0.)
    elapsed = decode = 0.
    kv = 2*spec.n_layers*spec.n_heads_kv*(spec.d_model//spec.n_heads_q)*(spec.kv_bits//8)
    for i, b in enumerate(sizes):
        r = hbm_service(spec, cw, b, workspace, platform, hbm, offload=policy=='HBM_HOST_OFFLOAD')
        admission = 0 if i==0 else b*cw.history*kv
        host = platform.host_offload
        # Serialized transfer pipeline is bounded by both C2C and HBM writes.
        admission_s = max(admission/host.direct_effective_bandwidth_bytes_per_second,
                          admission/hbm.sustained_bandwidth_bytes_per_s)
        admission_energy = dict(admission_host_memory_J=admission*8*host.memory_dynamic_J_per_bit,
            admission_c2c_J=admission*8*host.link_dynamic_J_per_bit, admission_hbm_write_J=admission*8*1.9955e-12,
            admission_gpu_static_J=admission_s*platform.gpu_decode_power.static_power_W)
        for k, v in (r['energy']|admission_energy).items(): energy[k] = energy.get(k, 0)+v
        for k, v in r['traffic'].items(): traffic[k] += v
        traffic['admission_bytes'] += admission
        traffic['host_read_bytes'] += admission
        traffic['HBM_write_bytes'] += admission
        end = elapsed+admission_s+r['prefill_s']+r['decode_s']
        for _ in range(b):
            requests.append(dict(request_id=len(requests), wave=i, queue_delay=elapsed, admission_delay=admission_s,
                prefill_latency=r['prefill_s'], first_decode_step=r['steps'][0]['latency_s'],
                decode_duration=r['decode_s'], completion_time=end))
        waves.append(dict(wave=i, batch=b, admission_bytes=admission, admission_s=admission_s,
                          admission_energy_J=sum(admission_energy.values()), queue_delay=elapsed, **r))
        elapsed = end
        decode += r['decode_s']
    total = sum(energy.values())
    return dict(policy=policy, status='EVALUATED', safe_resident_batch=limit, wave_sizes=sizes, num_waves=len(sizes),
        E2E_s=elapsed, decode_s=decode, tokens_per_s=batch*cw.generated/elapsed, Decode_tokens_per_s=batch*cw.generated/decode,
        E2E_J=total, J_per_token=total/(batch*cw.generated), tokens_per_J=batch*cw.generated/total,
        Tmax_C=85., thermal_feasible=True, energy=energy, traffic=traffic, requests=requests, waves=waves,
        **latency_metrics(requests, cw.generated))


def select_hbm_best(offload, wave):
    chosen = wave if wave['status']=='EVALUATED' and wave['tokens_per_s']>offload['tokens_per_s'] else offload
    result = deepcopy(chosen)
    result['selected_HBM_policy'] = chosen['policy']
    result['policy'] = 'HBM_BEST'
    return result
