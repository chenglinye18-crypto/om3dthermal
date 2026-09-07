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
    shard_mode: Literal["ROW_PARALLEL", "KV_ATOMIC", "RESIDENT_ONLY"] = "RESIDENT_ONLY"
    parallel_dimension: str | None = None
    max_useful_parallelism: int = 1
    atomic_locality_bytes: float = 0
    atomic_count: int = 0
    output_rows: int = 0
    traffic_provenance: str = "DIMENSION_DERIVED_ACTIVE_OPERATOR"


def build_dense_decode_placement_units(w: LLMDecodeInput) -> tuple[DenseDecodePlacementUnit, ...]:
    h, kv, b = w.d_model, w.n_heads_kv * w.d_head, w.weight_bits / 8
    specs = (("Q", h*h, h, h), ("K", h*kv, h, kv), ("V", h*kv, h, kv),
             ("O", h*h, h, h), ("FFN_GATE", h*w.d_ff, h, w.d_ff),
             ("FFN_UP", h*w.d_ff, h, w.d_ff), ("FFN_DOWN", w.d_ff*h, w.d_ff, h))
    units = []
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
    order = {name: i for i, name in enumerate(("Q", "K", "V", "ATTENTION_QK",
        "ATTENTION_AV", "O", "FFN_GATE", "FFN_UP", "FFN_DOWN", "LM_HEAD", "OTHER_WEIGHT"))}
    return tuple(sorted(units, key=lambda u: (u.layer_id, order[u.operator_type], u.request_id or 0)))


def active_weight_read_bytes(units) -> float:
    """Dimension-derived active operator weights for one aggregate step."""
    return sum(unit.active_weight_read_bytes for unit in units)


def unit_shard_fractions(unit: DenseDecodePlacementUnit, span: int) -> tuple[float, ...]:
    """Return exact row/vector fractions for a deterministic integer shard."""
    if span <= 0:
        raise ValueError("span must be positive")
    if unit.shard_mode in ("ROW_PARALLEL", "KV_ATOMIC") and unit.atomic_count:
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


def boundary_bytes_per_die(units, ownership, die_count):
    """Each active AV die returns a complete batch of hidden partial vectors.

    Owner unions are resolved separately for every layer. No reduction network
    or score/probability replication is assumed.
    """
    result = [0.0]*die_count
    for unit, owners in zip(units, ownership, strict=True):
        fractions=unit_shard_fractions(unit,len(owners))
        for die,fraction in zip(owners,fractions,strict=True):
            result[die] += (unit.score_bytes+unit.probability_bytes)*fraction
            if unit.shard_mode == "ROW_PARALLEL":
                # Input is broadcast; output rows are partitioned and gathered once.
                result[die] += unit.activation_input_bytes
                result[die] += unit.partial_output_bytes*fraction
        if unit.operator_type == "ATTENTION_AV":
            # One full FP32 hidden partial per (layer, request, owner).
            for die in owners:
                result[die] += unit.partial_output_bytes
    return tuple(result)
