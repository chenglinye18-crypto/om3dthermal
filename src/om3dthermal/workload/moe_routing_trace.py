"""Decode-only routing profiles from real Mixtral router logits.

The production entry point lazily imports PyTorch/Transformers and records
top-k indices from the model's own per-layer ``router_logits``.  Pure-Python
aggregation remains independently testable, but test tensors carry an explicit
``TEST_ONLY_SYNTHETIC_ROUTER_TENSORS`` status and are never formal measurements.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Literal, Sequence

from .moe_decode import MoEDecodeInput, evaluate_moe_decode
from .moe_m3d import build_moe_resident_objects


CANONICAL_MODEL_ID = "mistralai/Mixtral-8x7B-v0.1"
REAL_DATASET_ID = "REAL_SHAREGPT_CONVERSATIONS"
FORMAL_MEASUREMENT_STATUS = "MEASURED_REAL_ROUTING"
TEST_MEASUREMENT_STATUS = "TEST_ONLY_SYNTHETIC_ROUTER_TENSORS"
SKEW_CV_THRESHOLD = 0.10


MeasurementStatus = Literal[
    "MEASURED_REAL_ROUTING",
    "TEST_ONLY_SYNTHETIC_ROUTER_TENSORS",
]
SkewVerdict = Literal["SKEW_PRESENT", "NEAR_UNIFORM"]


@dataclass(frozen=True)
class ShareGPTPrompt:
    sample_id: str
    prompt: str


@dataclass(frozen=True)
class PromptRoutingSelections:
    """Actual selected expert IDs indexed by token, layer, then top-k."""

    sample_id: str
    selected_experts: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class GlobalRoutingSkew:
    min_p: float
    p10: float
    p25: float
    median_p: float
    mean_p: float
    p75: float
    p90: float
    max_p: float
    std_p: float
    max_to_median_ratio: float | None
    coefficient_of_variation: float | None
    skew_cv_threshold: float
    skew_verdict: SkewVerdict


@dataclass(frozen=True)
class LayerRoutingSkew:
    layer: int
    min_expert_p: float
    max_expert_p: float
    max_to_min_ratio: float | None
    entropy_bits: float


@dataclass(frozen=True)
class RoutingConcentration:
    top_10_percent_selection_share: float
    top_25_percent_selection_share: float
    top_50_percent_selection_share: float


@dataclass(frozen=True)
class SplitRoutingStability:
    split_a_prompt_ids: tuple[str, ...]
    split_b_prompt_ids: tuple[str, ...]
    split_a_decode_tokens: int
    split_b_decode_tokens: int
    pearson_correlation: float | None
    spearman_rank_correlation: float | None
    top_10_percent_expert_overlap: float | None
    top_25_percent_expert_overlap: float | None


@dataclass(frozen=True)
class MixtralRoutingProfile:
    model_id: str
    dataset_id: str
    measurement_status: MeasurementStatus
    routing_scope: Literal["DECODE_ROUTING_ONLY"]
    prefill_routing_included: bool
    seed: int
    num_prompts: int
    num_decode_tokens: int
    num_layers: int
    num_experts: int
    top_k: int
    total_selection_events: int
    sample_ids: tuple[str, ...]
    selection_counts: tuple[tuple[int, ...], ...]
    selection_probability: tuple[tuple[float, ...], ...]
    normalized_share: tuple[tuple[float, ...], ...]
    global_skew: GlobalRoutingSkew
    per_layer_skew: tuple[LayerRoutingSkew, ...]
    concentration: RoutingConcentration
    split_stability: SplitRoutingStability
    probability_semantics: str
    normalized_share_semantics: str
    router_capture_semantics: str
    event_closure_status: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExpertObjectReadDemand:
    object_id: str
    layer: int
    expert: int
    selection_probability: float
    read_demand_bytes_per_token: float


@dataclass(frozen=True)
class ExpertRoutingDemand:
    model_id: str
    measurement_status: MeasurementStatus
    expert_footprint_bytes: float
    expert_objects: tuple[ExpertObjectReadDemand, ...]
    total_expert_read_bytes_per_token: float
    expected_active_expert_read_bytes_per_token: float
    closure_error_bytes: float
    traffic_closure_status: str
    shared_nonexpert_traffic_modified: bool
    kv_traffic_modified: bool


def load_sharegpt_prompts(
    dataset_path: str | Path,
    *,
    num_prompts: int,
    seed: int,
) -> tuple[ShareGPTPrompt, ...]:
    """Load deterministic first-human-turn prompts from real ShareGPT JSON."""
    path = Path(dataset_path)
    if num_prompts <= 0:
        raise ValueError("num_prompts must be positive")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    records = payload.get("conversations") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("ShareGPT dataset root must be a list or conversations list")

    prompts: list[ShareGPTPrompt] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        sample_id = str(
            record.get("id", record.get("conversation_id", index)))
        if sample_id in seen_ids:
            raise ValueError(f"duplicate ShareGPT sample ID: {sample_id}")
        turns = record.get("conversations", record.get("messages"))
        prompt = _first_human_turn(turns)
        if prompt is None:
            continue
        seen_ids.add(sample_id)
        prompts.append(ShareGPTPrompt(sample_id=sample_id, prompt=prompt))
    if len(prompts) < num_prompts:
        raise ValueError(
            f"ShareGPT dataset has {len(prompts)} usable prompts, "
            f"fewer than requested {num_prompts}")
    ordered = sorted(prompts, key=lambda item: item.sample_id)
    rng = random.Random(seed)
    selected_indices = rng.sample(range(len(ordered)), num_prompts)
    return tuple(ordered[index] for index in selected_indices)


def build_routing_profile(
    prompt_selections: Sequence[PromptRoutingSelections],
    *,
    model_id: str,
    dataset_id: str,
    seed: int,
    num_layers: int,
    num_experts: int,
    top_k: int,
    measurement_status: MeasurementStatus,
) -> MixtralRoutingProfile:
    """Aggregate exact selected indices without inventing popularity."""
    _validate_source_identity(model_id, dataset_id, measurement_status)
    if num_layers <= 0 or num_experts <= 0 or top_k <= 0:
        raise ValueError("routing dimensions must be positive")
    if top_k > num_experts:
        raise ValueError("top_k cannot exceed num_experts")
    if not prompt_selections:
        raise ValueError("at least one prompt routing trace is required")
    ids = tuple(item.sample_id for item in prompt_selections)
    if any(not sample_id for sample_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("prompt routing sample IDs must be non-empty and unique")

    counts, token_count = _aggregate_counts(
        prompt_selections,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
    )
    if token_count <= 0:
        raise ValueError("routing profile requires at least one decode token")
    probabilities = tuple(
        tuple(value / token_count for value in row) for row in counts)
    shares = tuple(
        tuple(value / (token_count * top_k) for value in row)
        for row in counts
    )
    expected_events = token_count * num_layers * top_k
    actual_events = sum(sum(row) for row in counts)
    if actual_events != expected_events:
        raise ValueError("routing event count does not close")
    for layer, (count_row, p_row, q_row) in enumerate(
        zip(counts, probabilities, shares)
    ):
        if sum(count_row) != token_count * top_k:
            raise ValueError(f"routing count does not close for layer {layer}")
        if not math.isclose(sum(p_row), top_k, abs_tol=1e-12):
            raise ValueError(f"selection probability does not close for layer {layer}")
        if not math.isclose(sum(q_row), 1.0, abs_tol=1e-12):
            raise ValueError(f"normalized share does not close for layer {layer}")

    global_skew = _global_skew(probabilities)
    per_layer = tuple(
        _layer_skew(layer, probabilities[layer], shares[layer])
        for layer in range(num_layers)
    )
    concentration = _concentration(counts)
    stability = _split_stability(
        prompt_selections,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
    )
    return MixtralRoutingProfile(
        model_id=model_id,
        dataset_id=dataset_id,
        measurement_status=measurement_status,
        routing_scope="DECODE_ROUTING_ONLY",
        prefill_routing_included=False,
        seed=seed,
        num_prompts=len(prompt_selections),
        num_decode_tokens=token_count,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        total_selection_events=actual_events,
        sample_ids=ids,
        selection_counts=counts,
        selection_probability=probabilities,
        normalized_share=shares,
        global_skew=global_skew,
        per_layer_skew=per_layer,
        concentration=concentration,
        split_stability=stability,
        probability_semantics=(
            "P_SELECT_EQUALS_COUNT_DIVIDED_BY_DECODE_TOKENS_SUMS_TO_TOP_K"),
        normalized_share_semantics=(
            "Q_SHARE_EQUALS_COUNT_DIVIDED_BY_LAYER_SELECTION_EVENTS_SUMS_TO_ONE"),
        router_capture_semantics=(
            "TOP_K_FROM_MODEL_EXPOSED_ROUTER_LOGITS_USING_OFFICIAL_SOFTMAX_"
            "TOPK_SEMANTICS"),
        event_closure_status="EXACT_DECODE_TOKEN_LAYER_TOP_K_CLOSURE",
    )


def build_expert_object_read_demand(
    profile: MixtralRoutingProfile,
    workload: MoEDecodeInput,
) -> ExpertRoutingDemand:
    """Map measured p[layer,expert] to existing expert object identities."""
    if workload.batch_size != 1:
        raise ValueError("routing demand adapter currently requires batch_size=1")
    if profile.model_id != workload.model_id:
        raise ValueError("routing profile model does not match workload")
    if (profile.num_layers, profile.num_experts, profile.top_k) != (
        workload.num_hidden_layers,
        workload.num_local_experts,
        workload.num_experts_per_tok,
    ):
        raise ValueError("routing profile dimensions do not match workload")
    metrics = evaluate_moe_decode(workload)
    expert_bytes = metrics.expert_footprint_bytes
    resident_experts = tuple(
        item for item in build_moe_resident_objects(workload)
        if item.object_id.startswith("expert.")
    )
    if len(resident_experts) != profile.num_layers * profile.num_experts:
        raise ValueError("existing resident expert objects do not close")
    if any(item.size_bytes != expert_bytes for item in resident_experts):
        raise ValueError("resident expert size does not match analytical metrics")
    objects = tuple(
        ExpertObjectReadDemand(
            object_id=resident_experts[layer * profile.num_experts + expert].object_id,
            layer=layer,
            expert=expert,
            selection_probability=profile.selection_probability[layer][expert],
            read_demand_bytes_per_token=(
                profile.selection_probability[layer][expert] * expert_bytes),
        )
        for layer in range(profile.num_layers)
        for expert in range(profile.num_experts)
    )
    actual = sum(item.read_demand_bytes_per_token for item in objects)
    expected = metrics.active_expert_weight_bytes_per_decode_step
    error = actual - expected
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            "expert routing demand does not close to active expert traffic")
    return ExpertRoutingDemand(
        model_id=profile.model_id,
        measurement_status=profile.measurement_status,
        expert_footprint_bytes=expert_bytes,
        expert_objects=objects,
        total_expert_read_bytes_per_token=actual,
        expected_active_expert_read_bytes_per_token=expected,
        closure_error_bytes=error,
        traffic_closure_status="EXACT_ACTIVE_EXPERT_READ_TRAFFIC_CLOSURE",
        shared_nonexpert_traffic_modified=False,
        kv_traffic_modified=False,
    )


def write_routing_artifacts(
    profile: MixtralRoutingProfile,
    output_dir: str | Path,
    *,
    resolved_config: dict[str, object],
    dataset_metadata: dict[str, object],
) -> None:
    """Write the formal machine-readable bundle into a new directory."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "routing_profile.json", profile.as_dict())
    _write_json(output / "resolved_config.json", resolved_config)
    _write_json(output / "dataset_metadata.json", dataset_metadata)
    _write_json(output / "summary.json", {
        "global_skew": asdict(profile.global_skew),
        "concentration": asdict(profile.concentration),
        "split_stability": asdict(profile.split_stability),
        "event_closure_status": profile.event_closure_status,
    })
    with (output / "routing_counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("layer", "expert", "count"))
        for layer in range(profile.num_layers):
            for expert in range(profile.num_experts):
                writer.writerow((
                    layer, expert, profile.selection_counts[layer][expert]))
    with (output / "routing_probability.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("layer", "expert", "p_select", "q_share"))
        for layer in range(profile.num_layers):
            for expert in range(profile.num_experts):
                writer.writerow((
                    layer,
                    expert,
                    profile.selection_probability[layer][expert],
                    profile.normalized_share[layer][expert],
                ))


def run_real_mixtral_routing_profile(
    *,
    model_id: str,
    dataset_path: str | Path,
    dataset_source: str,
    num_prompts: int,
    max_new_tokens: int,
    seed: int,
    output_dir: str | Path,
    revision: str = "main",
) -> MixtralRoutingProfile:
    """Run real base-Mixtral greedy decode and persist formal artifacts."""
    if model_id != CANONICAL_MODEL_ID:
        raise ValueError(
            f"formal routing model must be exactly {CANONICAL_MODEL_ID}")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not dataset_source.strip():
        raise ValueError("dataset_source must identify the ShareGPT provenance")
    if not revision.strip():
        raise ValueError("model revision must be non-empty")
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "REAL_ROUTING_TRACE_BLOCKED: PyTorch and Transformers are required "
            "in the canonical om3dthermal environment"
        ) from error

    dataset = Path(dataset_path)
    prompts = load_sharegpt_prompts(
        dataset, num_prompts=num_prompts, seed=seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    config = model.config
    dimensions = (
        int(config.num_hidden_layers),
        int(config.num_local_experts),
        int(config.num_experts_per_tok),
    )
    if dimensions != (32, 8, 2):
        raise RuntimeError(
            f"loaded Mixtral routing dimensions are {dimensions}, expected (32, 8, 2)")
    embedding_device = model.get_input_embeddings().weight.device
    eos_ids = config.eos_token_id
    eos_set = {int(value) for value in (
        eos_ids if isinstance(eos_ids, list) else [eos_ids]
    ) if value is not None}
    max_prompt_tokens = int(config.max_position_embeddings) - max_new_tokens
    if max_prompt_tokens <= 0:
        raise ValueError("max_new_tokens leaves no room for a prompt")

    traces: list[PromptRoutingSelections] = []
    with torch.inference_mode():
        for prompt in prompts:
            encoded = tokenizer(
                prompt.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_tokens,
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(embedding_device)
            attention_mask = encoded["attention_mask"].to(embedding_device)
            prefill = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_router_logits=False,
                return_dict=True,
            )
            next_token = prefill.logits[:, -1, :].argmax(
                dim=-1, keepdim=True).to(embedding_device)
            past = prefill.past_key_values
            token_layers: list[tuple[tuple[int, ...], ...]] = []
            for _ in range(max_new_tokens):
                attention_mask = torch.cat((
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ), dim=1)
                decoded = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    output_router_logits=True,
                    return_dict=True,
                )
                router_logits = decoded.router_logits
                if router_logits is None or len(router_logits) != dimensions[0]:
                    raise RuntimeError(
                        "Mixtral did not return one router-logit tensor per layer")
                layer_indices: list[tuple[int, ...]] = []
                for layer_logits in router_logits:
                    logits = layer_logits.reshape(-1, dimensions[1])
                    if logits.shape[0] != 1:
                        raise RuntimeError(
                            "decode router logits must contain exactly one token")
                    probabilities = torch.softmax(logits.float(), dim=-1)
                    selected = torch.topk(
                        probabilities, dimensions[2], dim=-1).indices[0]
                    layer_indices.append(tuple(
                        int(value) for value in selected.detach().cpu().tolist()))
                token_layers.append(tuple(layer_indices))
                current_token = int(next_token.item())
                past = decoded.past_key_values
                if current_token in eos_set:
                    break
                next_token = decoded.logits[:, -1, :].argmax(
                    dim=-1, keepdim=True).to(embedding_device)
            traces.append(PromptRoutingSelections(
                sample_id=prompt.sample_id,
                selected_experts=tuple(token_layers),
            ))

    profile = build_routing_profile(
        traces,
        model_id=model_id,
        dataset_id=REAL_DATASET_ID,
        seed=seed,
        num_layers=dimensions[0],
        num_experts=dimensions[1],
        top_k=dimensions[2],
        measurement_status=FORMAL_MEASUREMENT_STATUS,
    )
    resolved_config = {
        "model_id": model_id,
        "dataset_id": REAL_DATASET_ID,
        "dataset_source": dataset_source,
        "dataset_path": str(dataset.resolve()),
        "requested_model_revision": revision,
        "resolved_model_commit": getattr(config, "_commit_hash", None),
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "num_prompts": num_prompts,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "do_sample": False,
        "batch_size": 1,
        "device_map": "auto",
        "dtype": "bfloat16",
        "routing_scope": "DECODE_ROUTING_ONLY",
        "prompt_extraction": "FIRST_HUMAN_TURN",
    }
    dataset_metadata = {
        "dataset_id": REAL_DATASET_ID,
        "dataset_source": dataset_source,
        "dataset_sha256": _sha256(dataset),
        "sample_ids": [prompt.sample_id for prompt in prompts],
        "sample_count": len(prompts),
        "seed": seed,
        "prompt_text_persisted": False,
    }
    write_routing_artifacts(
        profile,
        output_dir,
        resolved_config=resolved_config,
        dataset_metadata=dataset_metadata,
    )
    return profile


def _aggregate_counts(
    prompts: Sequence[PromptRoutingSelections],
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
) -> tuple[tuple[tuple[int, ...], ...], int]:
    counts = [[0 for _ in range(num_experts)] for _ in range(num_layers)]
    tokens = 0
    for prompt in prompts:
        for token in prompt.selected_experts:
            if len(token) != num_layers:
                raise ValueError("routing token layer shape does not close")
            for layer, selected in enumerate(token):
                if len(selected) != top_k or len(set(selected)) != top_k:
                    raise ValueError("each layer must contain distinct top-k experts")
                for expert in selected:
                    if isinstance(expert, bool) or not isinstance(expert, int):
                        raise TypeError("selected expert IDs must be integers")
                    if expert < 0 or expert >= num_experts:
                        raise ValueError("selected expert ID is out of range")
                    counts[layer][expert] += 1
            tokens += 1
    return tuple(tuple(row) for row in counts), tokens


def _global_skew(
    probabilities: tuple[tuple[float, ...], ...],
) -> GlobalRoutingSkew:
    values = tuple(value for row in probabilities for value in row)
    ordered = tuple(sorted(values))
    mean = statistics.fmean(values)
    median = statistics.median(values)
    std = statistics.pstdev(values)
    cv = None if mean == 0 else std / mean
    return GlobalRoutingSkew(
        min_p=ordered[0],
        p10=_percentile(ordered, 0.10),
        p25=_percentile(ordered, 0.25),
        median_p=median,
        mean_p=mean,
        p75=_percentile(ordered, 0.75),
        p90=_percentile(ordered, 0.90),
        max_p=ordered[-1],
        std_p=std,
        max_to_median_ratio=None if median == 0 else ordered[-1] / median,
        coefficient_of_variation=cv,
        skew_cv_threshold=SKEW_CV_THRESHOLD,
        skew_verdict=(
            "SKEW_PRESENT"
            if cv is not None and cv >= SKEW_CV_THRESHOLD
            else "NEAR_UNIFORM"
        ),
    )


def _layer_skew(
    layer: int,
    probabilities: tuple[float, ...],
    shares: tuple[float, ...],
) -> LayerRoutingSkew:
    minimum = min(probabilities)
    maximum = max(probabilities)
    entropy = -sum(value * math.log2(value) for value in shares if value > 0)
    return LayerRoutingSkew(
        layer=layer,
        min_expert_p=minimum,
        max_expert_p=maximum,
        max_to_min_ratio=None if minimum == 0 else maximum / minimum,
        entropy_bits=entropy,
    )


def _concentration(
    counts: tuple[tuple[int, ...], ...],
) -> RoutingConcentration:
    values = tuple(value for row in counts for value in row)
    total = sum(values)
    return RoutingConcentration(
        top_10_percent_selection_share=_top_share(values, total, 0.10),
        top_25_percent_selection_share=_top_share(values, total, 0.25),
        top_50_percent_selection_share=_top_share(values, total, 0.50),
    )


def _split_stability(
    prompts: Sequence[PromptRoutingSelections],
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
) -> SplitRoutingStability:
    midpoint = (len(prompts) + 1) // 2
    split_a = prompts[:midpoint]
    split_b = prompts[midpoint:]
    counts_a, tokens_a = _aggregate_counts(
        split_a, num_layers=num_layers, num_experts=num_experts, top_k=top_k)
    if split_b:
        counts_b, tokens_b = _aggregate_counts(
            split_b,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
        )
        values_a = tuple(value / tokens_a for row in counts_a for value in row)
        values_b = tuple(value / tokens_b for row in counts_b for value in row)
        pearson = _pearson(values_a, values_b)
        spearman = _pearson(_rank(values_a), _rank(values_b))
        overlap_10 = _top_overlap(values_a, values_b, 0.10)
        overlap_25 = _top_overlap(values_a, values_b, 0.25)
    else:
        tokens_b = 0
        pearson = spearman = overlap_10 = overlap_25 = None
    return SplitRoutingStability(
        split_a_prompt_ids=tuple(item.sample_id for item in split_a),
        split_b_prompt_ids=tuple(item.sample_id for item in split_b),
        split_a_decode_tokens=tokens_a,
        split_b_decode_tokens=tokens_b,
        pearson_correlation=pearson,
        spearman_rank_correlation=spearman,
        top_10_percent_expert_overlap=overlap_10,
        top_25_percent_expert_overlap=overlap_25,
    )


def _validate_source_identity(
    model_id: str,
    dataset_id: str,
    measurement_status: MeasurementStatus,
) -> None:
    if measurement_status == FORMAL_MEASUREMENT_STATUS:
        if model_id != CANONICAL_MODEL_ID or dataset_id != REAL_DATASET_ID:
            raise ValueError(
                "MEASURED_REAL_ROUTING requires canonical base Mixtral and "
                "REAL_SHAREGPT_CONVERSATIONS")
    elif measurement_status != TEST_MEASUREMENT_STATUS:
        raise ValueError("unknown routing measurement status")


def _first_human_turn(turns: object) -> str | None:
    if not isinstance(turns, list):
        return None
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", turn.get("role", ""))).lower()
        content = turn.get("value", turn.get("content"))
        if role in {"human", "user"} and isinstance(content, str):
            stripped = content.strip()
            if stripped:
                return stripped
    return None


def _percentile(ordered: tuple[float, ...], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _top_share(values: tuple[int, ...], total: int, fraction: float) -> float:
    count = math.ceil(len(values) * fraction)
    return sum(sorted(values, reverse=True)[:count]) / total


def _pearson(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    centered_left = tuple(value - mean_left for value in left)
    centered_right = tuple(value - mean_right for value in right)
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(centered_left, centered_right)) / denominator


def _rank(values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    return tuple(ranks)


def _top_overlap(
    left: tuple[float, ...],
    right: tuple[float, ...],
    fraction: float,
) -> float:
    count = math.ceil(len(left) * fraction)
    left_top = set(sorted(
        range(len(left)), key=lambda index: (-left[index], index))[:count])
    right_top = set(sorted(
        range(len(right)), key=lambda index: (-right[index], index))[:count])
    return len(left_top & right_top) / count


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile decode-only routing on real base Mixtral + ShareGPT")
    parser.add_argument("--model", default=CANONICAL_MODEL_ID)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument(
        "--dataset-source",
        required=True,
        help="public source URL or immutable dataset provenance identifier",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    profile = run_real_mixtral_routing_profile(
        model_id=args.model,
        dataset_path=args.dataset_path,
        dataset_source=args.dataset_source,
        num_prompts=args.num_prompts,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        output_dir=args.output_dir,
        revision=args.revision,
    )
    print(json.dumps({
        "measurement_status": profile.measurement_status,
        "num_prompts": profile.num_prompts,
        "num_decode_tokens": profile.num_decode_tokens,
        "total_selection_events": profile.total_selection_events,
        "skew_verdict": profile.global_skew.skew_verdict,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
