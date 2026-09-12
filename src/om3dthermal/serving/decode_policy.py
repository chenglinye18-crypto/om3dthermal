"""Performance-only placement policies on the physical FEOL execution model."""
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import numpy as np
from om3dthermal.platform import load_platform_spec_file
from om3dthermal.workload import LLMDecodeInput, LLMPrefillInput, evaluate_cached_prefix_incremental_prefill
from om3dthermal.workload.dense_decode_ledger import build_dense_decode_placement_units, build_dense_decode_small_ops
from om3dthermal.architecture.feol_floorplan import resolve_feol_floorplan
from om3dthermal.placement.nmp_load_balance import PhysicalResidentPlacement
from om3dthermal.power.nmp_die_activity import PhysicalStageModel

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
    """One operator schedule, one physical resident placement, three executors."""
    def __init__(self, workload, *, project_root: Path, record_energy=False):
        self.record_energy = record_energy
        self.workload = workload
        self.floorplan = resolve_feol_floorplan(project_root)
        self.platform = load_platform_spec_file(project_root/"configs/platform/gpu_package_h200_reference.yaml")
        self.placement = PhysicalResidentPlacement(workload, self.floorplan)
        self.physical = PhysicalStageModel(self.floorplan, self.platform, workload, record_events=record_energy)
        self.static = {}
        self.dynamic = {}
        self.context = None
        units = build_dense_decode_placement_units(workload)
        owners = [tuple(sorted(set(self.placement.operators[u.layer_id, u.operator_type].lanes % self.floorplan.layout.slab_count))) for u in units]
        self.small = build_dense_decode_small_ops(workload, units, owners, self.floorplan.layout.slab_count)
        self.small_bytes = {}
        for x in self.small:
            key = (x.operator, x.layer_id)
            self.small_bytes[key] = self.small_bytes.get(key, 0)+x.gpu_local_total_bytes

    def _operator(self, op, layer, context, policy, *, write=False):
        entry = self.placement.operators[layer, op]
        nmp = on_nmp(op, policy) and not write
        if op in ATTENTION:
            atoms = context*self.workload.n_heads_kv
            key = (op, layer, nmp, write)
            if key not in self.dynamic:
                self.dynamic[key] = self.physical.evaluate(entry, atoms+self.workload.n_heads_kv if write else atoms,
                    nmp=nmp, begin=atoms if write else 0, write=write)
            return self.dynamic[key]
        key = (op, layer, nmp)
        if key not in self.static:
            self.static[key] = self.physical.evaluate(entry, 1 if op == "TOKEN_EMBED_LOOKUP" else entry.atom_count, nmp=nmp)
        return self.static[key]

    def _gpu_small(self, op, layer, context):
        w = self.workload
        if op == "SOFTMAX":
            count = w.n_heads_q*context*4
        else:
            count = self.small_bytes.get((op, layer), 0)
        seconds = count/self.physical.gpu_bw
        return dict(operator=op, layer=layer, executor="GPU", latency_s=seconds,
                    bottleneck="GPU_COMPUTE", components={"GPU_COMPUTE": seconds})

    def step(self, context, policy, *, include_stages=False):
        if not 0 < context < self.workload.context_length:
            raise ValueError("context plus append exceeds reserved resident KV")
        policy = ExecutionPolicy(policy)
        if self.context != context:
            self.dynamic.clear()
            self.context = context
        w = self.workload
        stages = [self._operator("TOKEN_EMBED_LOOKUP", -1, context, policy)]
        for l in range(w.n_layers):
            # RMSNorm has two variants; select one half per explicit norm stage.
            norm = self._gpu_small("RMSNORM", l, context)
            norm = dict(norm, latency_s=norm["latency_s"]/2,
                        components={"GPU_COMPUTE": norm["latency_s"]/2})
            stages.append(norm)
            stages.extend(self._operator(op, l, context, policy) for op in ("Q", "K", "V"))
            stages.append(self._gpu_small("ROPE", l, context))
            stages.extend(self._operator(op, l, context, policy, write=True) for op in ("ATTENTION_QK", "ATTENTION_AV"))
            stages.append(self._operator("ATTENTION_QK", l, context, policy))
            stages.append(self._gpu_small("SOFTMAX", l, context))
            stages.append(self._operator("ATTENTION_AV", l, context, policy))
            if policy != ExecutionPolicy.NO_NMP:
                stages.append(self._gpu_small("AV_REDUCTION", l, context))
            stages.append(self._operator("O", l, context, policy))
            residual = self._gpu_small("RESIDUAL_ADD", l, context)
            residual = dict(residual, latency_s=residual["latency_s"]/2,
                            components={"GPU_COMPUTE": residual["latency_s"]/2})
            stages.extend((residual, norm))
            stages.extend(self._operator(op, l, context, policy) for op in ("FFN_GATE", "FFN_UP"))
            stages.append(self._gpu_small("SWIGLU", l, context))
            stages.append(self._operator("FFN_DOWN", l, context, policy))
            stages.append(residual)
        stages.append(self._gpu_small("FINAL_RMSNORM", w.n_layers, context))
        stages.append(self._operator("LM_HEAD", w.n_layers, context, policy))
        stages.append(self._gpu_small("SAMPLING", w.n_layers, context))
        seconds = sum(s["latency_s"] for s in stages)
        physical = [s for s in stages if "local_array_bytes" in s]
        nmp_stages = [s for s in physical if s["executor"] == "NMP"]
        traffic = dict.fromkeys(("weight", "historical_K", "historical_V", "Q", "score_T", "softmax_S", "attention_O",
                                "FFN_activation", "other", "KV_append", "local_read", "local_write"), 0.0)
        for s in physical:
            op = s["operator"]
            if op == "KV_APPEND":
                traffic["KV_append"] += s["boundary_bytes"]
                traffic["local_write"] += s["local_array_bytes"]
                continue
            traffic["local_read"] += s["local_array_bytes"]
            if s["executor"] == "GPU":
                key = "historical_K" if op == "ATTENTION_QK" else "historical_V" if op == "ATTENTION_AV" else "weight"
                traffic[key] += s["boundary_bytes"]
            elif op == "ATTENTION_QK":
                traffic["Q"] += s["input_boundary_bytes"]; traffic["score_T"] += s["output_boundary_bytes"]
            elif op == "ATTENTION_AV":
                traffic["softmax_S"] += s["input_boundary_bytes"]; traffic["attention_O"] += s["output_boundary_bytes"]
            else:
                traffic["FFN_activation" if op.startswith("FFN") else "other"] += s["boundary_bytes"]
        def total(key):
            return sum(s.get(key, 0) for s in physical)
        bottlenecks = {}
        for s in stages:
            b = bottlenecks.setdefault(s["bottleneck"], dict(count=0, time_s=0.0))
            b["count"] += 1; b["time_s"] += s["latency_s"]
        transfers = [t for s in physical for t in s["external_transfers"] if t["total_boundary_bytes"] > 0]
        component_sums = {k:sum(s["components"].get(k,0) for s in stages) for k in
                          ("ARRAY","LOCAL_FABRIC","MAC","INTER_REGION_NOC","EXTERNAL_BOUNDARY","GPU_COMPUTE")}
        groups = [s["active_groups"] for s in physical]
        row = dict(context=context, latency_s=seconds, traffic_bytes=traffic,
                   boundary_bytes=total("boundary_bytes"), external_service_s=total("external_service_s"),
                   local_array_bytes=total("local_array_bytes"), array_service_s=total("array_service_s"),
                   active_group_counts=groups, active_region_counts=[s["active_regions"] for s in physical],
                   active_tile_counts=[s["active_mac_tiles"] for s in physical],
                   fabric_region_peak_Bps=max((s["fabric_region_peak_Bps"] for s in physical), default=0),
                   fabric_region_average_Bps=(sum(s["fabric_region_average_Bps"]*s["latency_s"] for s in nmp_stages)/seconds),
                   noc_bytes=total("noc_bytes"), noc_s=total("noc_s"),
                   noc_max_link_utilization=float(sum(s["noc_link_busy_s"] for s in physical).max())/seconds,
                   noc_link_busy_s=sum(s["noc_link_busy_s"] for s in physical).ravel().tolist(),
                   nmp_flops=total("nmp_flops"),
                   NMP_peak_utilization=max((s["nmp_peak_utilization"] for s in physical), default=0),
                   NMP_average_utilization=total("nmp_flops")/seconds/(self.floorplan.layout.slab_count*32*self.floorplan.tile_flops),
                   component_sums=component_sums,
                   active_external_port_counts=[t["active_port_count"] for t in transfers],
                   max_external_port_utilization=max((t["max_port_utilization"] for t in transfers),default=0),
                   external_limit_counts={reason:sum(t["limiting_reason"]==reason for t in transfers) for reason in
                                          ("GLOBAL_THERMAL_CAP","PORT_SERIALIZATION","ROUTE_STARTUP")},
                   bottlenecks=bottlenecks)
        if self.record_energy:
            from om3dthermal.power.feol_energy import sum_events
            row["energy_events"] = sum_events(s["energy_events"] for s in physical)
        if include_stages:
            row["stages"] = stages
        return row

    def prefill(self, workload):
        w = self.workload
        inp = LLMPrefillInput(**{k: getattr(w, k) for k in LLMPrefillInput.model_fields if k != "prompt_length"}, prompt_length=workload.prompt)
        m = evaluate_cached_prefix_incremental_prefill(inp, cached_history_tokens=workload.history)
        p = self.platform.gpu_prefill_compute
        compute_s = ((m.linear_flops+m.lm_head_flops)/(p.large_gemm_effective_tflops*1e12)
                     + m.attention_flops/(p.causal_attention_effective_tflops*1e12))
        # Existing fused/tiled incremental Prefill ledger, GPU-only roofline.
        # Real resident groups service every weight and historical KV read.
        array_s = external_s = 0.0
        bulk_events = []
        for (layer, op), entry in self.placement.operators.items():
            if op in ("OTHER_WEIGHT", "TOKEN_EMBED_LOOKUP"):
                continue
            atoms = workload.history*w.n_heads_kv if op in ATTENTION else entry.atom_count
            point = self.physical.evaluate(entry, atoms, nmp=False)
            array_s += point["array_service_s"]
            external_s += point["external_service_s"]
            if self.record_energy: bulk_events.append(point["energy_events"])
        nonbulk = m.total_memory_bytes-m.active_weight_read_bytes-m.historical_cached_kv_read_bytes
        external_s += nonbulk/self.floorplan.external_Bps
        result = dict(latency_s=max(compute_s, array_s, external_s), compute_s=compute_s,
                    array_service_s=array_s, external_service_s=external_s,
                    external_bandwidth_cap_TBps=self.floorplan.external_Bps/1e12, ledger=m.model_dump())

        if self.record_energy:
            from om3dthermal.power.feol_energy import prefill_events
            result["energy_events"] = prefill_events(self,workload,result["ledger"],bulk_events)
        return result
