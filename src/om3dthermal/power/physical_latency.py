"""Pure spatial MAT + MIV + FEOL + interface latency composition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics

from .config import PhysicalAccessLatencyInput
from .feol_route import FEOLRouteResult


@dataclass(frozen=True)
class PhysicalLocationLatency:
    cluster_id: int
    layer_id: int
    cluster_center_x_um: float
    cluster_center_y_um: float
    feol_route_length_um: float
    miv_length_um: float
    mat_latency_ns: float
    miv_latency_ns: float
    feol_latency_ns: float
    interface_latency_ns: float
    total_latency_ns: float


@dataclass(frozen=True)
class PhysicalAccessLatency:
    locations: tuple[PhysicalLocationLatency, ...]
    latency_map_ns: tuple[tuple[float, ...], ...]
    number_of_clusters: int
    number_of_layers: int
    number_of_locations: int
    min_total_latency_ns: float
    median_total_latency_ns: float
    p90_total_latency_ns: float
    max_total_latency_ns: float
    uniform_average_total_latency_ns: float
    latency_spread_ns: float
    far_near_ratio: float
    min_location: tuple[int, int]
    max_location: tuple[int, int]
    mat_classification: str
    mat_status: str
    mat_note: str
    miv_status: str
    miv_parameter_status: str
    miv_provenance: str
    feol_status: str
    feol_provenance: dict[str, dict[str, str]]
    interface_classification: str
    interface_status: str
    interface_included_in_total: bool
    interface_note: str
    physical_latency_semantics: str
    workload_weighted: bool
    serialization_included: bool
    excluded_components: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def calculate_physical_access_latency(
        spec: PhysicalAccessLatencyInput, *, feol_route: FEOLRouteResult,
        miv_length_per_layer_um: tuple[float, ...],
        miv_delay_per_layer_ns: tuple[float, ...],
        miv_status: str, miv_parameter_status: str, miv_provenance: str,
        ) -> PhysicalAccessLatency:
    """Compose an unweighted physical (cluster, layer) latency map."""
    if (not miv_length_per_layer_um
            or len(miv_length_per_layer_um) != len(miv_delay_per_layer_ns)):
        raise ValueError("MIV length and delay arrays must be non-empty and align")
    if any(not math.isfinite(value) or value <= 0.0
           for value in miv_length_per_layer_um):
        raise ValueError("MIV lengths must be finite and positive")
    if any(not math.isfinite(value) or value < 0.0
           for value in miv_delay_per_layer_ns):
        raise ValueError("MIV delays must be finite and non-negative")
    if miv_status != "RESOLVED":
        raise ValueError("physical access latency requires resolved MIV latency")
    if not miv_parameter_status or not miv_provenance:
        raise ValueError("MIV latency parameter status and provenance are required")
    feol_delays = feol_route.feol_delay_per_cluster_ns
    if feol_delays is None:
        raise ValueError("resolved FEOL propagation latency is required")
    cluster_count = feol_route.feol_route_cluster_count
    if not (
            len(feol_delays)
            == len(feol_route.feol_route_length_per_cluster_um)
            == len(feol_route.feol_route_cluster_centers_um)
            == cluster_count):
        raise ValueError("FEOL cluster geometry and delay arrays must align")
    if (feol_route.feol_latency_status is None
            or feol_route.feol_latency_provenance is None):
        raise ValueError("FEOL latency status and provenance are required")

    locations: list[PhysicalLocationLatency] = []
    latency_rows: list[tuple[float, ...]] = []
    for cluster_id, (center, route_length, feol_delay) in enumerate(zip(
            feol_route.feol_route_cluster_centers_um,
            feol_route.feol_route_length_per_cluster_um,
            feol_delays,
            strict=True)):
        row: list[float] = []
        for layer_index, (miv_length, miv_delay) in enumerate(zip(
                miv_length_per_layer_um,
                miv_delay_per_layer_ns,
                strict=True)):
            total = (
                spec.mat_latency_ns
                + miv_delay
                + feol_delay
                + spec.interface_latency_ns)
            row.append(total)
            locations.append(PhysicalLocationLatency(
                cluster_id=cluster_id,
                layer_id=layer_index + 1,
                cluster_center_x_um=center[0],
                cluster_center_y_um=center[1],
                feol_route_length_um=route_length,
                miv_length_um=miv_length,
                mat_latency_ns=spec.mat_latency_ns,
                miv_latency_ns=miv_delay,
                feol_latency_ns=feol_delay,
                interface_latency_ns=spec.interface_latency_ns,
                total_latency_ns=total,
            ))
        latency_rows.append(tuple(row))

    location_tuple = tuple(locations)
    totals = tuple(location.total_latency_ns for location in location_tuple)
    minimum_location = min(location_tuple, key=lambda item: item.total_latency_ns)
    maximum_location = max(location_tuple, key=lambda item: item.total_latency_ns)
    minimum = minimum_location.total_latency_ns
    maximum = maximum_location.total_latency_ns
    return PhysicalAccessLatency(
        locations=location_tuple,
        latency_map_ns=tuple(latency_rows),
        number_of_clusters=cluster_count,
        number_of_layers=len(miv_length_per_layer_um),
        number_of_locations=len(location_tuple),
        min_total_latency_ns=minimum,
        median_total_latency_ns=statistics.median(totals),
        p90_total_latency_ns=_percentile(totals, 0.9),
        max_total_latency_ns=maximum,
        uniform_average_total_latency_ns=sum(totals) / len(totals),
        latency_spread_ns=maximum - minimum,
        far_near_ratio=maximum / minimum,
        min_location=(
            minimum_location.cluster_id, minimum_location.layer_id),
        max_location=(
            maximum_location.cluster_id, maximum_location.layer_id),
        mat_classification=spec.mat_classification,
        mat_status=spec.mat_status,
        mat_note=spec.mat_note,
        miv_status=miv_status,
        miv_parameter_status=miv_parameter_status,
        miv_provenance=miv_provenance,
        feol_status=feol_route.feol_latency_status,
        feol_provenance=feol_route.feol_latency_provenance,
        interface_classification=spec.interface_classification,
        interface_status=spec.interface_status,
        interface_included_in_total=spec.interface_included_in_total,
        interface_note=spec.interface_note,
        physical_latency_semantics="ONE_WAY_PHYSICAL_MEMORY_ACCESS_PATH",
        workload_weighted=False,
        serialization_included=False,
        excluded_components=(
            "serialization", "queueing", "bank_conflict",
            "controller_scheduling", "bandwidth_saturation", "contention",
            "host_device_transfer", "gpu_compute",
        ),
    )
