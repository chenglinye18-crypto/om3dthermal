"""Dense decode operator ledger; all counts are per aggregate decode step.

Resident capacity and active decode traffic are deliberately separate.
``OTHER_WEIGHT`` closes the declared parameter footprint, but is not an
invented operator and therefore has no read traffic or FLOPs.
"""
from dataclasses import dataclass
from typing import Literal

from .llm_decode import LLMDecodeInput

SCORE_BITS = 16
PROBABILITY_BITS = 16
PARTIAL_ACCUMULATION_BITS = 32
ACTIVATION_BYTES = 2
TOKEN_BYTES = 4


@dataclass(frozen=True)
class DenseDecodePlacementUnit:
    unit_id: str
    layer_id: int
    operator_type: str
    weight_bytes: float
    kv_bytes: float
    local_flops: int  # per request for shared weights
    activation_input_bytes: float
    partial_output_bytes: float
    request_id: int | None
    placement_scope: Literal["SHARED_BATCH", "REQUEST_LOCAL"]
    kv_write_bytes: float = 0
    score_bytes: float = 0
    probability_bytes: float = 0
    active_weight_read_bytes: float = 0
    shard_mode: Literal["ROW_PARALLEL", "KV_ATOMIC", "LOCAL_LOOKUP", "RESIDENT_ONLY"] = "RESIDENT_ONLY"
    parallel_dimension: str | None = None
    max_useful_parallelism: int = 1
    atomic_locality_bytes: float = 0
    atomic_count: int = 0
    output_rows: int = 0
    traffic_provenance: str = "DIMENSION_DERIVED_ACTIVE_OPERATOR"


@dataclass(frozen=True)
class DenseDecodeSmallOp:
    operator: str
    variant: str
    layer_id: int
    request_id: int
    gpu_local_read_bytes: float
    gpu_local_write_bytes: float
    gpu_local_total_bytes: float
    nmp_to_gpu_bytes: float
    gpu_to_nmp_bytes: float
    execution_count: int
    provenance: str


@dataclass(frozen=True)
class DenseDecodeHandoff:
    producer: str
    consumer: str
    direction: Literal["NMP_TO_GPU", "GPU_TO_NMP"]
    bytes: float
    layer_id: int
    request_id: int | None
    reason: str
    per_die_bytes: tuple[float, ...]


def build_dense_decode_placement_units(w: LLMDecodeInput) -> tuple[DenseDecodePlacementUnit, ...]:
    h, kv, b = w.d_model, w.n_heads_kv * w.d_head, w.weight_bits / 8
    specs = (("Q", h*h, h, h), ("K", h*kv, h, kv), ("V", h*kv, h, kv),
             ("O", h*h, h, h), ("FFN_GATE", h*w.d_ff, h, w.d_ff),
             ("FFN_UP", h*w.d_ff, h, w.d_ff), ("FFN_DOWN", w.d_ff*h, w.d_ff, h))
    units = []
    for request in range(w.batch_size):
        units.append(DenseDecodePlacementUnit(
            f"TOKEN_EMBED_LOOKUP.request.{request}", -1, "TOKEN_EMBED_LOOKUP",
            0, 0, 0, 0, h*ACTIVATION_BYTES, request, "REQUEST_LOCAL",
            active_weight_read_bytes=h*b, shard_mode="LOCAL_LOOKUP",
            parallel_dimension="embedding_row", max_useful_parallelism=1,
            atomic_count=1,
            traffic_provenance="ONE_EMBEDDING_ROW_ACTIVE_READ__MATRIX_RESIDENT_IN_OTHER_WEIGHT"))
    for layer in range(w.n_layers):
        for name, params, inp, out in specs:
            weight_bytes=params*b
            units.append(DenseDecodePlacementUnit(
                f"layer.{layer}.{name}", layer, name, params*b, 0,
                2*params, w.batch_size*inp*2, w.batch_size*out*2, None, "SHARED_BATCH",
                active_weight_read_bytes=weight_bytes, shard_mode="ROW_PARALLEL",
                parallel_dimension="output_rows", max_useful_parallelism=out,
                atomic_count=out, output_rows=out))
        for request in range(w.batch_size):
            for name in ("ATTENTION_QK", "ATTENTION_AV"):
                elements = w.n_heads_q*w.context_length
                atomic_count=w.context_length*w.n_heads_kv
                atomic_bytes=w.d_head*w.kv_bits/8
                units.append(DenseDecodePlacementUnit(
                    f"layer.{layer}.{name}.request.{request}", layer, name, 0,
                    w.context_length*kv*w.kv_bits/8,
                    2*w.n_heads_q*w.context_length*w.d_head, 0,
                    h*PARTIAL_ACCUMULATION_BITS/8 if name == "ATTENTION_AV" else 0,
                    request, "REQUEST_LOCAL", kv*w.kv_bits/8,
                    elements*SCORE_BITS/8 if name == "ATTENTION_QK" else 0,
                    elements*PROBABILITY_BITS/8 if name == "ATTENTION_AV" else 0,
                    shard_mode="KV_ATOMIC", parallel_dimension="token_kv_head",
                    max_useful_parallelism=max(1,atomic_count),
                    atomic_locality_bytes=atomic_bytes, atomic_count=atomic_count))
    params = h*w.vocab_size
    units.append(DenseDecodePlacementUnit("LM_HEAD", w.n_layers, "LM_HEAD",
        params*b, 0, 2*params, w.batch_size*h*2, w.batch_size*w.vocab_size*2, None, "SHARED_BATCH",
        active_weight_read_bytes=params*b, shard_mode="ROW_PARALLEL",
        parallel_dimension="output_rows", max_useful_parallelism=w.vocab_size,
        atomic_count=w.vocab_size, output_rows=w.vocab_size))
    residual = w.n_param*b-sum(u.weight_bytes for u in units)
    if residual < 0:
        raise ValueError("named operators exceed declared parameter footprint")
    if residual:
        units.append(DenseDecodePlacementUnit("OTHER_WEIGHT", w.n_layers,
            "OTHER_WEIGHT", residual, 0, 0, 0, 0, None, "SHARED_BATCH",
            active_weight_read_bytes=0, shard_mode="RESIDENT_ONLY",
            traffic_provenance="RESIDENT_FOOTPRINT_RESIDUAL_ONLY__NOT_ACTIVE_FULL_READ_TRAFFIC"))
    order = {name: i for i, name in enumerate(("TOKEN_EMBED_LOOKUP", "Q", "K", "V", "ATTENTION_QK",
        "ATTENTION_AV", "O", "FFN_GATE", "FFN_UP", "FFN_DOWN", "LM_HEAD", "OTHER_WEIGHT"))}
    return tuple(sorted(units, key=lambda u: (u.layer_id, order[u.operator_type], u.request_id or 0)))


def active_weight_read_bytes(units) -> float:
    """Dimension-derived active operator weights for one aggregate step."""
    return sum(unit.active_weight_read_bytes for unit in units)


def unit_shard_fractions(unit: DenseDecodePlacementUnit, span: int) -> tuple[float, ...]:
    """Return exact row/vector fractions for a deterministic integer shard."""
    if span <= 0:
        raise ValueError("span must be positive")
    if unit.shard_mode in ("ROW_PARALLEL", "KV_ATOMIC", "LOCAL_LOOKUP") and unit.atomic_count:
        quotient,remainder=divmod(unit.atomic_count,span)
        return tuple((quotient+(index<remainder))/unit.atomic_count
                     for index in range(span))
    return (1/span,)*span


def attention_boundary_by_layer(units, ownership):
    """Placement-resolved attention ledger, resolving AV per request."""
    layers = {}
    for unit, owners in zip(units, ownership, strict=True):
        if unit.operator_type not in ("ATTENTION_QK", "ATTENTION_AV"):
            continue
        row = layers.setdefault(unit.layer_id, dict(score_bytes=0.0,
            probability_bytes=0.0, partial_vector_bytes=0.0, av_owners=set(),
            qk_request_owners={}, av_request_owners={}, partial_bytes=0.0))
        row["score_bytes"] += unit.score_bytes
        row["probability_bytes"] += unit.probability_bytes
        if unit.operator_type == "ATTENTION_QK":
            row["qk_request_owners"][unit.request_id] = tuple(sorted(owners))
        if unit.operator_type == "ATTENTION_AV":
            row["av_owners"].update(owners)
            row["partial_vector_bytes"] += unit.partial_output_bytes
            row["av_request_owners"][unit.request_id] = tuple(sorted(owners))
            row["partial_bytes"] += len(owners)*unit.partial_output_bytes
    for row in layers.values():
        row["av_owners"] = tuple(sorted(row["av_owners"]))
    return layers


def _unit_map(units, ownership):
    return {(unit.layer_id,unit.operator_type,unit.request_id):(unit,owners)
            for unit,owners in zip(units,ownership,strict=True)}


def _per_die_partition(unit, owners, total_bytes, die_count):
    result=[0.0]*die_count
    for die,fraction in zip(owners,unit_shard_fractions(unit,len(owners)),strict=True):
        result[die]+=total_bytes*fraction
    return tuple(result)


def _per_die_broadcast(owners, bytes_per_owner, die_count):
    result=[0.0]*die_count
    for die in owners:
        result[die]+=bytes_per_owner
    return tuple(result)


def build_dense_decode_handoffs(units, ownership, die_count):
    """Resolve every physical NMP/GPU producer-consumer transfer once."""
    by=_unit_map(units,ownership); handoffs=[]
    batch=1+max((u.request_id for u in units if u.request_id is not None),default=0)
    layers=1+max(u.layer_id for u in units if u.operator_type=="Q")
    def add(producer,consumer,direction,bytes_,layer,request,reason,per_die):
        handoffs.append(DenseDecodeHandoff(producer,consumer,direction,bytes_,layer,
            request,reason,per_die))
    for request in range(batch):
        unit,owners=by[-1,"TOKEN_EMBED_LOOKUP",request]
        add("TOKEN_EMBED_LOOKUP","RMSNORM_PRE_ATTENTION","NMP_TO_GPU",
            unit.partial_output_bytes,-1,request,"complete-token embedding handoff",
            _per_die_partition(unit,owners,unit.partial_output_bytes,die_count))
    for layer in range(layers):
        q,qowners=by[layer,"Q",None]; k,kowners=by[layer,"K",None]
        v,vowners=by[layer,"V",None]; o,oowners=by[layer,"O",None]
        gate,gateowners=by[layer,"FFN_GATE",None]; up,upowners=by[layer,"FFN_UP",None]
        down,downowners=by[layer,"FFN_DOWN",None]
        for unit,owners in ((q,qowners),(k,kowners),(v,vowners)):
            add("RMSNORM_PRE_ATTENTION",unit.operator_type,"GPU_TO_NMP",
                unit.activation_input_bytes*len(owners),layer,None,
                "normalized hidden broadcast to row-parallel projection",
                _per_die_broadcast(owners,unit.activation_input_bytes,die_count))
        for unit,owners in ((q,qowners),(k,kowners),(v,vowners)):
            add(unit.operator_type,"ROPE" if unit.operator_type in ("Q","K") else "KV_VECTOR_REPACK",
                "NMP_TO_GPU",unit.partial_output_bytes,layer,None,
                "partition gather for GPU RoPE" if unit.operator_type in ("Q","K")
                else "conservative GPU-mediated whole-head KV locality repack; not RoPE",
                _per_die_partition(unit,owners,unit.partial_output_bytes,die_count))
        for request in range(batch):
            qk,qkowners=by[layer,"ATTENTION_QK",request]
            av,avowners=by[layer,"ATTENTION_AV",request]
            qbytes=q.partial_output_bytes/batch; kbytes=k.partial_output_bytes/batch
            vbytes=v.partial_output_bytes/batch
            # Every QK owner needs Q; new K/V vectors are partitioned, not replicated.
            add("ROPE_Q","ATTENTION_QK","GPU_TO_NMP",qbytes*len(qkowners),
                layer,request,"Q broadcast to KV-sharded QK owners",
                _per_die_broadcast(qkowners,qbytes,die_count))
            add("ROPE_K","KV_WRITE","GPU_TO_NMP",kbytes,layer,request,
                "whole K-head vectors routed to cache owners",
                _per_die_partition(qk,qkowners,kbytes,die_count))
            add("KV_VECTOR_REPACK","KV_WRITE","GPU_TO_NMP",vbytes,layer,request,
                "whole V-head vectors routed to cache owners; not RoPE traffic",
                _per_die_partition(av,avowners,vbytes,die_count))
            add("ATTENTION_QK","GPU_SOFTMAX","NMP_TO_GPU",qk.score_bytes,layer,request,
                "partitioned score gather",
                _per_die_partition(qk,qkowners,qk.score_bytes,die_count))
            add("GPU_SOFTMAX","ATTENTION_AV","GPU_TO_NMP",av.probability_bytes,layer,request,
                "probability partition routing",
                _per_die_partition(av,avowners,av.probability_bytes,die_count))
            partial=av.partial_output_bytes*len(avowners)
            add("ATTENTION_AV","AV_REDUCTION","NMP_TO_GPU",partial,layer,request,
                "one FP32 hidden partial per AV owner",
                _per_die_broadcast(avowners,av.partial_output_bytes,die_count))
        add("AV_REDUCTION","O","GPU_TO_NMP",o.activation_input_bytes*len(oowners),layer,None,
            "reduced FP16 attention output broadcast to O owners",
            _per_die_broadcast(oowners,o.activation_input_bytes,die_count))
        add("O","RESIDUAL_ADD_ATTENTION","NMP_TO_GPU",o.partial_output_bytes,layer,None,
            "row-partitioned O output gather",
            _per_die_partition(o,oowners,o.partial_output_bytes,die_count))
        for unit,owners in ((gate,gateowners),(up,upowners)):
            add("RMSNORM_PRE_FFN",unit.operator_type,"GPU_TO_NMP",
                unit.activation_input_bytes*len(owners),layer,None,
                "normalized hidden broadcast to FFN projection",
                _per_die_broadcast(owners,unit.activation_input_bytes,die_count))
            add(unit.operator_type,"SWIGLU","NMP_TO_GPU",unit.partial_output_bytes,layer,None,
                "row-partitioned FFN activation gather",
                _per_die_partition(unit,owners,unit.partial_output_bytes,die_count))
        add("SWIGLU","FFN_DOWN","GPU_TO_NMP",down.activation_input_bytes*len(downowners),
            layer,None,"gated activation broadcast to down projection",
            _per_die_broadcast(downowners,down.activation_input_bytes,die_count))
        add("FFN_DOWN","RESIDUAL_ADD_FFN","NMP_TO_GPU",down.partial_output_bytes,layer,None,
            "row-partitioned FFN output gather",
            _per_die_partition(down,downowners,down.partial_output_bytes,die_count))
    lm,lmowners=by[layers,"LM_HEAD",None]
    add("FINAL_RMSNORM","LM_HEAD","GPU_TO_NMP",lm.activation_input_bytes*len(lmowners),
        layers,None,"final normalized hidden broadcast to LM head owners",
        _per_die_broadcast(lmowners,lm.activation_input_bytes,die_count))
    add("LM_HEAD","SAMPLING","NMP_TO_GPU",lm.partial_output_bytes,layers,None,
        "row-partitioned logits gather",
        _per_die_partition(lm,lmowners,lm.partial_output_bytes,die_count))
    return tuple(handoffs)


def build_dense_decode_small_ops(w: LLMDecodeInput, units, ownership, die_count):
    """Single source of truth for non-matrix decode operations."""
    by=_unit_map(units,ownership); result=[]
    hidden=w.d_model*ACTIVATION_BYTES; ff=w.d_ff*ACTIVATION_BYTES
    def add(operator,variant,layer,request,read,write,n2g=0.0,g2n=0.0,provenance="GPU_MEMORY_BOUND_FIRST_ORDER"):
        result.append(DenseDecodeSmallOp(operator,variant,layer,request,read,write,
            read+write,n2g,g2n,1,provenance))
    for request in range(w.batch_size):
        embed,owners=by[-1,"TOKEN_EMBED_LOOKUP",request]
        add("TOKEN_EMBED_LOOKUP","M3D_ROW_LOOKUP",-1,request,0,0,
            n2g=embed.partial_output_bytes,
            provenance="ONE_M3D_EMBEDDING_ROW__NO_GPU_LOCAL_PROCESSING")
        for layer in range(w.n_layers):
            q,qowners=by[layer,"Q",None]; k,kowners=by[layer,"K",None]
            v,vowners=by[layer,"V",None]; qk,qkowners=by[layer,"ATTENTION_QK",request]
            av,avowners=by[layer,"ATTENTION_AV",request]; o,oowners=by[layer,"O",None]
            gate,gateowners=by[layer,"FFN_GATE",None]; up,upowners=by[layer,"FFN_UP",None]
            down,downowners=by[layer,"FFN_DOWN",None]
            qbytes=w.d_model*ACTIVATION_BYTES; kbytes=w.n_heads_kv*w.d_head*ACTIVATION_BYTES
            add("RMSNORM","PRE_ATTENTION",layer,request,2*hidden,hidden,
                g2n=hidden*(len(qowners)+len(kowners)+len(vowners)))
            add("ROPE","Q_K",layer,request,qbytes+kbytes,qbytes+kbytes,
                n2g=qbytes+kbytes,g2n=qbytes*len(qkowners)+kbytes)
            add("AV_REDUCTION","FP32_PARTIAL_TO_FP16",layer,request,
                len(avowners)*w.d_model*PARTIAL_ACCUMULATION_BITS/8,hidden,
                n2g=len(avowners)*w.d_model*PARTIAL_ACCUMULATION_BITS/8,
                g2n=hidden*len(oowners))
            add("RESIDUAL_ADD","ATTENTION",layer,request,2*hidden,hidden,n2g=hidden)
            add("RMSNORM","PRE_FFN",layer,request,2*hidden,hidden,
                g2n=hidden*(len(gateowners)+len(upowners)))
            add("SWIGLU","SILU_GATE_TIMES_UP",layer,request,2*ff,ff,
                n2g=2*ff,g2n=ff*len(downowners))
            add("RESIDUAL_ADD","FFN",layer,request,2*hidden,hidden,n2g=hidden)
        lm,lmowners=by[w.n_layers,"LM_HEAD",None]
        add("FINAL_RMSNORM","FINAL",w.n_layers,request,2*hidden,hidden,
            g2n=hidden*len(lmowners))
        logits=w.vocab_size*ACTIVATION_BYTES
        add("SAMPLING","GREEDY_ARGMAX",w.n_layers,request,logits,TOKEN_BYTES,
            n2g=logits,provenance="GPU_MEMORY_BOUND_GREEDY_ARGMAX")
    return tuple(result)


def boundary_bytes_per_die(units, ownership, die_count):
    """Per-die attribution of the explicit handoff ledger."""
    result=[0.0]*die_count
    for handoff in build_dense_decode_handoffs(units,ownership,die_count):
        for die,bytes_ in enumerate(handoff.per_die_bytes):
            result[die]+=bytes_
    return tuple(result)
