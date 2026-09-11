"""B1 placement comparison compiled from the canonical MAC-NMP stage ledger."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from om3dthermal.platform import load_platform_spec_file, resolve_gpu_prefill_compute_energy_calibration
from om3dthermal.workload import LLMDecodeInput, LLMPrefillInput, evaluate_cached_prefix_incremental_prefill
from om3dthermal.workload.dense_decode_ledger import build_dense_decode_placement_units
from .nmp_decode import evaluate_nmp_decode_batch


class ExecutionPolicy(StrEnum):
    NO_NMP = "NO_NMP"
    ATTENTION_NMP = "ATTENTION_NMP"
    MAC_NMP = "MAC_NMP"


ATTENTION = frozenset(("ATTENTION_QK", "ATTENTION_AV"))
MAC_OPERATORS = ATTENTION | {"Q", "K", "V", "O", "FFN_GATE", "FFN_UP", "FFN_DOWN", "LM_HEAD", "TOKEN_EMBED_LOOKUP"}


def on_nmp(operator: str, policy: ExecutionPolicy) -> bool:
    return operator in (MAC_OPERATORS if policy == ExecutionPolicy.MAC_NMP else
                        ATTENTION if policy == ExecutionPolicy.ATTENTION_NMP else ())


@dataclass(frozen=True)
class CachedWorkload:
    history: int = 125000
    prompt: int = 1000
    generated: int = 1000

    def __post_init__(self):
        if min(self.history, self.prompt, self.generated) <= 0:
            raise ValueError("cached workload lengths must be positive")
        if self.history+self.prompt+self.generated > 131072:
            raise ValueError("workload exceeds Llama 3.1 context limit")

    @property
    def contexts(self):
        return range(self.history + self.prompt, self.history + self.prompt + self.generated)


def llama31_models():
    """Meta llama-models/models/sku_list.py; 405B bf16-mp8 (not mp16).

    Full untied embedding + output + linear matrices + RMSNorm parameter count.
    FFN widths derive from Meta's multiplier/multiple_of rounding rule.
    """
    result = {}
    for size, h, layers, heads, ff in ((8, 4096, 32, 32, 14336),
                                     (70, 8192, 80, 64, 28672),
                                     (405, 16384, 126, 128, 53248)):
        params = layers * (2*h*h + 2*h*1024 + 3*h*ff + 2*h) + 2*h*128256 + h
        result[f"Llama-3.1-{size}B"] = LLMDecodeInput(
            n_param=params, n_layers=layers, n_heads_q=heads, n_heads_kv=8,
            d_model=h, d_ff=ff, vocab_size=128256, batch_size=1,
            context_length=127000, weight_bits=16, kv_bits=16,
            weight_activity_model="dimension_derived_active_operators",
            weight_reuse_model="tile_reuse", kv_read_model="full_reread")
    return result


class DecodePolicyModel:
    """Reuse one resident ownership and stage schedule for all policies/steps.

    Compile at the maximum resident context, then evaluate exact integer KV
    shards per step. No interpolation, context averaging, or per-step re-placement.
    """

    def __init__(self, workload: LLMDecodeInput, *, project_root: Path):
        if workload.batch_size != 1 or workload.weight_bits != 16 or workload.kv_bits != 16:
            raise ValueError("comparison requires B1 and uniform 16-bit storage")
        self.workload = workload
        self.canonical = evaluate_nmp_decode_batch(workload, project_root=project_root)
        if self.canonical.evaluation_status != "EVALUATED":
            raise ValueError("resident capacity failed; host offload is not permitted")
        trace = self.canonical.execution_trace
        self.architecture = trace.architecture
        self.activity = trace.activity
        self.primitives = trace.power_map.primitives
        self.platform = load_platform_spec_file(project_root / "configs/platform/gpu_package_h200_reference.yaml")
        self.units = {(u.layer_id, u.operator_type): u for u in build_dense_decode_placement_units(workload)}
        self.owners = {(u.layer_id, u.operator_type): owners for u, owners in
                       zip(build_dense_decode_placement_units(workload), trace.placement.ownership, strict=True)}
        self.bandwidth = self.activity.transfer["bandwidth_actual_bytes_per_s"]
        self.gpu_bw = self.platform.gpu_decode_power.peak_memory_bandwidth_bytes_per_s
        self.compute = self.platform.gpu_compute_power.peak_compute_BF16_dense_flops_per_s

    def _attention_stage(self, u, context):
        # The first shard has ceil(atoms/span) vectors and ceil(KV_heads/span)
        # appended vectors, hence is the canonical maximum for both resources.
        span = len(self.owners[u.layer_id, u.operator_type])
        w = self.workload
        atoms = context*w.n_heads_kv
        max_atoms = (atoms+span-1)//span
        writes = ((w.n_heads_kv+span-1)//span)*w.d_head*2
        memory_s = (max_atoms*w.d_head*2+writes)/self.activity.local_bandwidth_per_die_bytes_per_s
        flops_per_atom = 2*w.n_heads_q*w.d_head/w.n_heads_kv
        compute_s = max_atoms*flops_per_atom/self.activity.hardware.peak_flops_per_die
        return memory_s, compute_s

    def step(self, context: int, policy: ExecutionPolicy, *, include_stages=False):
        policy = ExecutionPolicy(policy)
        w = self.workload
        if not 0 < context <= w.context_length:
            raise ValueError("context outside reserved resident horizon")
        traffic = dict.fromkeys(("weight", "historical_K", "historical_V", "Q", "score_T",
                                "softmax_S", "attention_O", "FFN_activation", "other",
                                "KV_append", "local_read", "local_write"), 0.0)
        stages = []
        gpu_bytes = nmp_flops = peak_util = 0.0
        boundary_keys = tuple(k for k in traffic if not k.startswith("local_"))
        for original in self.activity.stages:
            s = dict(original)
            op, layer, kind = s["operator"], s["layer"], s["kind"]
            seconds = s["time_ms"]*1e-3
            if kind == "NMP_STAGE":
                u = self.units[layer, op]
                kv = context*w.n_heads_kv*w.d_head*2 if op in ATTENTION else 0
                flops = 2*w.n_heads_q*context*w.d_head if op in ATTENTION else u.local_flops
                reads = u.active_weight_read_bytes+kv
                writes = u.kv_write_bytes
                traffic["local_read"] += reads
                traffic["local_write"] += writes
                if on_nmp(op, policy):
                    if op in ATTENTION:
                        local_memory_s, local_compute_s = self._attention_stage(u, context)
                        seconds = max(local_memory_s, local_compute_s)
                        s.update(memory_ms=local_memory_s*1000, compute_ms=local_compute_s*1000,
                                 stage_ms=seconds*1000)
                    nmp_flops += flops
                    if seconds:
                        peak_util = max(peak_util, flops/seconds/self.activity.hardware.aggregate_peak_flops)
                else:
                    traffic["weight"] += u.active_weight_read_bytes
                    if op in ATTENTION:
                        traffic["historical_K" if op == "ATTENTION_QK" else "historical_V"] += kv
                    # KV append transfers are already explicit handoffs below.
                    transfer_s = reads/self.bandwidth
                    local_memory_s = (self._attention_stage(u, context)[0] if op in ATTENTION
                                      else original["memory_ms"]*1e-3)
                    seconds = max(flops/self.compute, transfer_s, local_memory_s)
                    local_activation = u.activation_input_bytes + u.partial_output_bytes
                    if op == "ATTENTION_AV":
                        local_activation = context*w.n_heads_q*2+w.d_model*2
                    elif op == "ATTENTION_QK":
                        local_activation = w.d_model*2+context*w.n_heads_q*2
                    gpu_bytes += reads + local_activation
                    seconds = max(seconds, (reads+local_activation)/self.gpu_bw)
                    s["kind"] = "GPU_STAGE"
                    s.update(compute_ms=flops/self.compute*1000,
                             memory_ms=max(transfer_s, local_memory_s, (reads+local_activation)/self.gpu_bw)*1000,
                             stage_ms=seconds*1000)
            elif kind == "BOUNDARY_HANDOFF":
                producer, consumer = s["producer"], s["consumer"]
                source_local = on_nmp(producer, policy)
                dest_local = on_nmp(consumer, policy) or consumer == "KV_WRITE"
                if source_local == dest_local:
                    continue
                count = (context*w.n_heads_q*2 if op in ("SCORE_TRANSFER", "PROBABILITY_TRANSFER") else s["bytes"])
                if consumer == "KV_WRITE":
                    category = "KV_append"
                elif consumer == "ATTENTION_QK":
                    category = "Q"
                elif producer == "ATTENTION_QK":
                    category = "score_T"
                elif consumer == "ATTENTION_AV":
                    category = "softmax_S"
                elif producer in ("ATTENTION_AV", "AV_REDUCTION"):
                    category = "attention_O"
                elif "FFN" in producer or "FFN" in consumer or producer == "SWIGLU" or consumer == "SWIGLU":
                    category = "FFN_activation"
                else:
                    category = "other"
                traffic[category] += count
                seconds = count/self.bandwidth
                s["bytes"] = count
            else:
                if op == "AV_REDUCTION" and policy == ExecutionPolicy.NO_NMP:
                    continue
                count = context*w.n_heads_q*4 if kind == "GPU_SOFTMAX" else s.get("gpu_local_bytes", 0)
                s["gpu_local_bytes"] = count
                gpu_bytes += count
                seconds = count/self.gpu_bw
            s["time_ms"] = seconds*1000
            stages.append(s)
        seconds = sum(s["time_ms"] for s in stages)*1e-3
        boundary = sum(traffic[k] for k in boundary_keys)
        p = self.primitives
        memory_j = 8e-12*(traffic["local_read"]*p.local_read_total_pj_per_bit
                          + traffic["local_write"]*p.local_write_total_pj_per_bit)
        boundary_j = boundary*8e-12*(p.long_feol_pj_per_bit+p.interface_pj_per_bit)
        nmp_j = nmp_flops/2*p.mac_energy_pj_per_mac*1e-12
        gpu_dynamic_j = gpu_bytes*8*self.platform.gpu_decode_power.e_decode_J_per_bit
        gpu_static_j = seconds*self.platform.gpu_decode_power.static_power_W
        energy = dict(gpu_J=gpu_dynamic_j+gpu_static_j, memory_J=memory_j,
                      boundary_J=boundary_j, nmp_J=nmp_j, memory_static_refresh_J=0.0)
        result = dict(context=context, latency_s=seconds, **energy, total_J=sum(energy.values()),
                      gpu_dynamic_J=gpu_dynamic_j, gpu_static_J=gpu_static_j,
                      boundary_bytes=boundary, traffic_bytes=traffic,
                      nmp_flops=nmp_flops, peak_nmp_utilization=peak_util,
                      average_nmp_utilization=nmp_flops/seconds/self.activity.hardware.aggregate_peak_flops)
        if include_stages:
            result["stages"] = stages
        return result

    def prefill(self, workload: CachedWorkload):
        if workload.history+workload.prompt+workload.generated > self.workload.context_length:
            raise ValueError("workload exceeds reserved resident horizon")
        fields = LLMPrefillInput.model_fields
        inp = LLMPrefillInput(**{k: getattr(self.workload, k) for k in fields if k != "prompt_length"},
                              prompt_length=workload.prompt)
        m = evaluate_cached_prefix_incremental_prefill(inp, cached_history_tokens=workload.history)
        c = resolve_gpu_prefill_compute_energy_calibration(self.platform.gpu_compute_power,
                                                         self.platform.gpu_prefill_compute)
        compute_s = ((m.linear_flops+m.lm_head_flops)/(c.large_gemm_effective_tflops*1e12)
                     + m.attention_flops/(c.causal_attention_effective_tflops*1e12))
        seconds = max(compute_s, m.total_memory_bytes/self.bandwidth)
        p = self.primitives
        memory_j = 8e-12*(m.total_read_bytes*p.local_read_total_pj_per_bit
                          + m.total_write_bytes*p.local_write_total_pj_per_bit)
        boundary_j = m.total_memory_bytes*8e-12*(p.long_feol_pj_per_bit+p.interface_pj_per_bit)
        energies = {}
        for bound in ("min", "max"):
            dynamic = ((m.linear_flops+m.lm_head_flops)*getattr(c, f"nominal_gemm_dynamic_J_per_FLOP_{bound}")
                       + m.attention_flops*getattr(c, f"nominal_attention_dynamic_J_per_FLOP_{bound}"))
            energies[f"gpu_J_{bound}"] = dynamic+c.static_power_W*seconds
            energies[f"total_J_{bound}"] = energies[f"gpu_J_{bound}"]+memory_j+boundary_j
        return dict(latency_s=seconds, **energies, memory_J=memory_j, boundary_J=boundary_j,
                    nmp_J=0.0, memory_static_refresh_J=0.0, ledger=m.model_dump())
