"""Dense decode operator ledger; all counts are per aggregate decode step.

MODELING_CHOICE: FP16 scores/probabilities, FP32 AV partials, full resident
weight reads. OTHER_WEIGHT closes the declared footprint without inventing MACs.
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


def build_dense_decode_placement_units(w: LLMDecodeInput) -> tuple[DenseDecodePlacementUnit, ...]:
    h, kv, b = w.d_model, w.n_heads_kv * w.d_head, w.weight_bits / 8
    specs = (("Q", h*h, h, h), ("K", h*kv, h, kv), ("V", h*kv, h, kv),
             ("O", h*h, h, h), ("FFN_GATE", h*w.d_ff, h, w.d_ff),
             ("FFN_UP", h*w.d_ff, h, w.d_ff), ("FFN_DOWN", w.d_ff*h, w.d_ff, h))
    units = []
    for layer in range(w.n_layers):
        for name, params, inp, out in specs:
            units.append(DenseDecodePlacementUnit(
                f"layer.{layer}.{name}", layer, name, params*b, 0,
                2*params, w.batch_size*inp*2, w.batch_size*out*2, None, "SHARED_BATCH"))
        for request in range(w.batch_size):
            for name in ("ATTENTION_QK", "ATTENTION_AV"):
                elements = w.n_heads_q*w.context_length
                units.append(DenseDecodePlacementUnit(
                    f"layer.{layer}.{name}.request.{request}", layer, name, 0,
                    w.context_length*kv*w.kv_bits/8,
                    2*w.n_heads_q*w.context_length*w.d_head, 0,
                    h*PARTIAL_ACCUMULATION_BITS/8 if name == "ATTENTION_AV" else 0,
                    request, "REQUEST_LOCAL", kv*w.kv_bits/8,
                    elements*SCORE_BITS/8 if name == "ATTENTION_QK" else 0,
                    elements*PROBABILITY_BITS/8 if name == "ATTENTION_AV" else 0))
    params = h*w.vocab_size
    units.append(DenseDecodePlacementUnit("LM_HEAD", w.n_layers, "LM_HEAD",
        params*b, 0, 2*params, w.batch_size*h*2, w.batch_size*w.vocab_size*2, None, "SHARED_BATCH"))
    residual = w.n_param*b-sum(u.weight_bytes for u in units)
    if residual < 0:
        raise ValueError("named operators exceed declared parameter footprint")
    if residual:
        units.append(DenseDecodePlacementUnit("OTHER_WEIGHT", w.n_layers,
            "OTHER_WEIGHT", residual, 0, 0, 0, 0, None, "SHARED_BATCH"))
    order = {name: i for i, name in enumerate(("Q", "K", "V", "ATTENTION_QK",
        "ATTENTION_AV", "O", "FFN_GATE", "FFN_UP", "FFN_DOWN", "LM_HEAD", "OTHER_WEIGHT"))}
    return tuple(sorted(units, key=lambda u: (u.layer_id, order[u.operator_type], u.request_id or 0)))


def attention_boundary_by_layer(units, ownership):
    """Placement-resolved attention ledger, with an explicit AV owner union."""
    layers = {}
    for unit, owners in zip(units, ownership, strict=True):
        if unit.operator_type not in ("ATTENTION_QK", "ATTENTION_AV"):
            continue
        row = layers.setdefault(unit.layer_id, dict(score_bytes=0.0,
            probability_bytes=0.0, partial_vector_bytes=0.0, av_owners=set()))
        row["score_bytes"] += unit.score_bytes
        row["probability_bytes"] += unit.probability_bytes
        if unit.operator_type == "ATTENTION_AV":
            row["av_owners"].update(owners)
            row["partial_vector_bytes"] += unit.partial_output_bytes
    for row in layers.values():
        row["av_owners"] = tuple(sorted(row["av_owners"]))
        row["partial_bytes"] = len(row["av_owners"])*row["partial_vector_bytes"]
    return layers


def boundary_bytes_per_die(units, ownership, die_count):
    """Each active AV die returns a complete batch of hidden partial vectors.

    Owner unions are resolved separately for every layer. No reduction network
    or score/probability replication is assumed.
    """
    result = [0.0]*die_count
    for unit, owners in zip(units, ownership, strict=True):
        for die in owners:
            result[die] += (unit.score_bytes+unit.probability_bytes)/len(owners)
            if unit.placement_scope == "SHARED_BATCH":
                # Preserve the existing conservative GPU combine/forward boundary
                # for weight operators; no implicit die-to-die network.
                result[die] += 2*unit.activation_input_bytes+unit.partial_output_bytes
    for row in attention_boundary_by_layer(units, ownership).values():
        for die in row["av_owners"]:
            result[die] += row["partial_vector_bytes"]
    return tuple(result)
