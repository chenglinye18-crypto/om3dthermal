"""Paper-derived Mixtral expert demand from Fiddler ICLR 2025 Figure 8."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Literal

from .moe_decode import MoEDecodeInput, evaluate_moe_decode
from .moe_m3d import build_moe_resident_objects


@dataclass(frozen=True)
class RelativePopularityStatistics:
    mean: float
    std: float
    minimum: float
    median: float
    maximum: float
    p25: float
    p75: float
    count_lt_0_6: int
    count_gt_0_8: int


@dataclass(frozen=True)
class FiddlerPublishedProfile:
    model_id: str
    num_layers: int
    num_experts: int
    top_k: int
    relative_popularity: tuple[tuple[float, ...], ...]
    selection_probability: tuple[tuple[float, ...], ...]
    relative_statistics: RelativePopularityStatistics
    profile_kind: Literal["FIDDLER_PUBLISHED_ROUTING_PROFILE"]
    source_workload: Literal["SHAREGPT"]
    classification: Literal["DERIVED_FROM_PUBLISHED_ROUTING_PROFILE"]
    measurement_status: Literal["NOT_MEASURED_BY_THIS_WORK"]
    synthetic_status: Literal["NOT_SYNTHETIC"]
    extraction: Literal["DIGITIZED_FROM_OFFICIAL_ARXIV_SOURCE_FIGURE"]
    extraction_gate: Literal["PASS"]
    normalization_semantics: str


@dataclass(frozen=True)
class PublishedExpertObjectDemand:
    object_id: str
    layer: int
    expert: int
    relative_popularity: float
    selection_probability: float
    expected_read_bytes_per_token: float


@dataclass(frozen=True)
class PublishedExpertDemand:
    model_id: str
    expert_footprint_bytes: float
    expert_objects: tuple[PublishedExpertObjectDemand, ...]
    hottest_expert_object: str
    coldest_expert_object: str
    max_to_median_demand_ratio: float
    top_10_percent_selection_share: float
    top_25_percent_selection_share: float
    top_50_percent_selection_share: float
    total_expert_read_bytes_per_token: float
    expected_active_expert_read_bytes_per_token: float
    maximum_per_layer_closure_error_bytes: float
    total_closure_error_bytes: float
    traffic_closure_status: Literal["ACTIVE_EXPERT_READ_TRAFFIC_CLOSURE_PASS"]
    demand_source: Literal["FIDDLER_PUBLISHED_ROUTING_PROFILE"]
    expert_demand_skew_gate: Literal["SKEW_PRESENT", "NEAR_UNIFORM"]
    shared_nonexpert_traffic_modified: bool
    kv_traffic_modified: bool
    placement_included: bool


def load_fiddler_published_profile(
    csv_path: str | Path,
    metadata_path: str | Path,
) -> FiddlerPublishedProfile:
    """Load and validate the fixed 32x8 digitized publication artifact."""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    expected_metadata = {
        "model": "mistralai/Mixtral-8x7B-v0.1",
        "classification": "DERIVED_FROM_PUBLISHED_ROUTING_PROFILE",
        "measurement_status": "NOT_MEASURED_BY_THIS_WORK",
        "synthetic_status": "NOT_SYNTHETIC",
        "extraction": "DIGITIZED_FROM_OFFICIAL_ARXIV_SOURCE_FIGURE",
        "extraction_gate": "PASS",
        "source_workload": "random samples from ShareGPT",
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"Fiddler profile metadata {field} does not match {expected}")

    matrix: list[list[float | None]] = [[None] * 8 for _ in range(32)]
    with Path(csv_path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["layer", "expert", "relative_popularity"]:
            raise ValueError("unexpected Fiddler profile CSV schema")
        for row in reader:
            layer = int(row["layer"])
            expert = int(row["expert"])
            value = float(row["relative_popularity"])
            if not (0 <= layer < 32 and 0 <= expert < 8):
                raise ValueError("Fiddler profile index is out of range")
            if matrix[layer][expert] is not None:
                raise ValueError("duplicate Fiddler profile expert object")
            if not math.isfinite(value) or value < 0:
                raise ValueError("relative popularity must be finite and non-negative")
            matrix[layer][expert] = value
    if any(value is None for row in matrix for value in row):
        raise ValueError("Fiddler profile must contain exactly 32x8 expert objects")
    relative = tuple(tuple(float(value) for value in row) for row in matrix)
    flat = tuple(value for row in relative for value in row)
    stats = _relative_statistics(flat)
    stored_stats = metadata["extracted_statistics"]
    comparisons = {
        "mean": stats.mean,
        "std": stats.std,
        "min": stats.minimum,
        "median": stats.median,
        "max": stats.maximum,
        "p25": stats.p25,
        "p75": stats.p75,
        "count_lt_0_6": stats.count_lt_0_6,
        "count_gt_0_8": stats.count_gt_0_8,
    }
    for field, value in comparisons.items():
        expected = stored_stats[field]
        if isinstance(value, int):
            closes = value == expected
        else:
            closes = math.isclose(value, expected, rel_tol=0, abs_tol=1e-8)
        if not closes:
            raise ValueError(f"Fiddler CSV and metadata do not close for {field}")

    normalized: list[tuple[float, ...]] = []
    for layer, row in enumerate(relative):
        denominator = sum(row)
        if denominator <= 0:
            raise ValueError(f"Fiddler popularity sum is not positive at layer {layer}")
        probabilities = tuple(2.0 * value / denominator for value in row)
        if any(value < 0 or value > 1 for value in probabilities):
            raise ValueError(
                f"normalized Fiddler selection probability invalid at layer {layer}")
        if not math.isclose(sum(probabilities), 2.0, abs_tol=1e-12):
            raise ValueError(
                f"normalized Fiddler selection probabilities do not close at layer {layer}")
        normalized.append(probabilities)
    return FiddlerPublishedProfile(
        model_id=expected_metadata["model"],
        num_layers=32,
        num_experts=8,
        top_k=2,
        relative_popularity=relative,
        selection_probability=tuple(normalized),
        relative_statistics=stats,
        profile_kind="FIDDLER_PUBLISHED_ROUTING_PROFILE",
        source_workload="SHAREGPT",
        classification="DERIVED_FROM_PUBLISHED_ROUTING_PROFILE",
        measurement_status="NOT_MEASURED_BY_THIS_WORK",
        synthetic_status="NOT_SYNTHETIC",
        extraction="DIGITIZED_FROM_OFFICIAL_ARXIV_SOURCE_FIGURE",
        extraction_gate="PASS",
        normalization_semantics=(
            "P_LAYER_EXPERT_EQUALS_TOP_K_TIMES_RELATIVE_POPULARITY_DIVIDED_"
            "BY_LAYER_RELATIVE_POPULARITY_SUM"),
    )


def build_published_expert_demand(
    profile: FiddlerPublishedProfile,
    workload: MoEDecodeInput,
) -> PublishedExpertDemand:
    """Map Fiddler popularity to existing expert objects; no page placement."""
    if workload.batch_size != 1:
        raise ValueError("published expert demand currently requires batch_size=1")
    if profile.model_id != workload.model_id:
        raise ValueError("Fiddler profile model does not match Mixtral workload")
    if (profile.num_layers, profile.num_experts, profile.top_k) != (
        workload.num_hidden_layers,
        workload.num_local_experts,
        workload.num_experts_per_tok,
    ):
        raise ValueError("Fiddler profile dimensions do not match Mixtral workload")

    metrics = evaluate_moe_decode(workload)
    expert_bytes = metrics.expert_footprint_bytes
    resident_experts = tuple(
        item for item in build_moe_resident_objects(workload)
        if item.object_id.startswith("expert.")
    )
    if len(resident_experts) != 256:
        raise ValueError("existing Mixtral resident experts do not close to 256")
    if any(item.size_bytes != expert_bytes for item in resident_experts):
        raise ValueError("resident expert sizes do not match Mixtral metrics")

    objects = tuple(
        PublishedExpertObjectDemand(
            object_id=resident_experts[layer * 8 + expert].object_id,
            layer=layer,
            expert=expert,
            relative_popularity=profile.relative_popularity[layer][expert],
            selection_probability=profile.selection_probability[layer][expert],
            expected_read_bytes_per_token=(
                profile.selection_probability[layer][expert] * expert_bytes),
        )
        for layer in range(32)
        for expert in range(8)
    )
    per_layer_errors = tuple(
        math.fsum(item.expected_read_bytes_per_token for item in objects
                  if item.layer == layer) - 2.0 * expert_bytes
        for layer in range(32)
    )
    actual = math.fsum(item.expected_read_bytes_per_token for item in objects)
    expected = metrics.active_expert_weight_bytes_per_decode_step
    total_error = actual - expected
    if (
        max(abs(value) for value in per_layer_errors) > 1e-6
        or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-6)
    ):
        raise ValueError("published expert traffic does not close")

    ordered = sorted(
        objects,
        key=lambda item: (
            -item.expected_read_bytes_per_token, item.layer, item.expert),
    )
    median_demand = statistics.median(
        item.expected_read_bytes_per_token for item in objects)
    total_probability = sum(item.selection_probability for item in objects)
    flattened_probability = tuple(
        item.selection_probability for item in objects)
    uniform = 2.0 / profile.num_experts
    skew = any(
        not math.isclose(value, uniform, rel_tol=0, abs_tol=1e-12)
        for value in flattened_probability)
    return PublishedExpertDemand(
        model_id=profile.model_id,
        expert_footprint_bytes=expert_bytes,
        expert_objects=objects,
        hottest_expert_object=ordered[0].object_id,
        coldest_expert_object=ordered[-1].object_id,
        max_to_median_demand_ratio=(
            ordered[0].expected_read_bytes_per_token / median_demand),
        top_10_percent_selection_share=_top_share(
            flattened_probability, total_probability, 0.10),
        top_25_percent_selection_share=_top_share(
            flattened_probability, total_probability, 0.25),
        top_50_percent_selection_share=_top_share(
            flattened_probability, total_probability, 0.50),
        total_expert_read_bytes_per_token=actual,
        expected_active_expert_read_bytes_per_token=expected,
        maximum_per_layer_closure_error_bytes=max(
            abs(value) for value in per_layer_errors),
        total_closure_error_bytes=total_error,
        traffic_closure_status="ACTIVE_EXPERT_READ_TRAFFIC_CLOSURE_PASS",
        demand_source="FIDDLER_PUBLISHED_ROUTING_PROFILE",
        expert_demand_skew_gate="SKEW_PRESENT" if skew else "NEAR_UNIFORM",
        shared_nonexpert_traffic_modified=False,
        kv_traffic_modified=False,
        placement_included=False,
    )


def _relative_statistics(values: tuple[float, ...]) -> RelativePopularityStatistics:
    return RelativePopularityStatistics(
        mean=statistics.fmean(values),
        std=statistics.pstdev(values),
        minimum=min(values),
        median=statistics.median(values),
        maximum=max(values),
        p25=_percentile(values, 0.25),
        p75=_percentile(values, 0.75),
        count_lt_0_6=sum(value < 0.6 for value in values),
        count_gt_0_8=sum(value > 0.8 for value in values),
    )


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _top_share(
    values: tuple[float, ...],
    total: float,
    fraction: float,
) -> float:
    count = math.ceil(len(values) * fraction)
    return sum(sorted(values, reverse=True)[:count]) / total
