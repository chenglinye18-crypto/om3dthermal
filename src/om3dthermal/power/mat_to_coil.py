"""Read-only MAT-output to coil-TX-input lateral topology diagnostic.

The canonical FEOL route remains the legacy direct cluster-to-nearest-port
model.  This module reuses its geometry, assignment, and electrical inputs to
construct a first-order hierarchical diagnostic without mutating that model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import statistics

from .feol_route import FEOLRouteResult


@dataclass(frozen=True)
class LatencySummary:
    min_ns: float
    median_ns: float
    p90_ns: float
    max_ns: float
    mean_ns: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class WireRCDelay:
    length_um: float
    wire_resistance_ohm: float
    wire_capacitance_pF: float
    endpoint_load_pF: float
    driver_cap_time_constant_ps: float
    wire_load_time_constant_ps: float
    distributed_wire_time_constant_ps: float
    time_constant_ps: float
    delay_ns: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PortFanIn:
    port_id: int
    fan_in_clusters: int
    collector_x_um: float | None
    collector_y_um: float | None
    shared_trunk_length_um: float | None
    legacy_route_min_um: float | None
    legacy_route_median_um: float | None
    legacy_route_max_um: float | None
    depth_from_edge_min_um: float | None
    depth_from_edge_median_um: float | None
    depth_from_edge_max_um: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MATToCoilPath:
    cluster_id: int
    assigned_port: int
    port_fan_in_clusters: int
    cluster_x_um: float
    cluster_y_um: float
    collector_x_um: float
    collector_y_um: float
    port_x_um: float
    port_y_um: float
    depth_from_edge_um: float
    branch_length_um: float
    trunk_length_um: float
    total_physical_path_um: float
    legacy_direct_length_um: float
    branch_wire_delay_ns: float
    shared_trunk_wire_delay_ns: float
    port_selection_delay_ns: float
    structural_aggregation_delay_ns: float
    total_lateral_delay_ns: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AggregationLoadSensitivity:
    aggregation_load_multiplier: float
    aggregation_load_pF: float
    latency: LatencySummary

    def as_dict(self) -> dict[str, object]:
        return {
            "aggregation_load_multiplier": self.aggregation_load_multiplier,
            "aggregation_load_pF": self.aggregation_load_pF,
            "latency": self.latency.as_dict(),
        }


@dataclass(frozen=True)
class MATToCoilAudit:
    legacy_model_name: str
    hierarchical_model_name: str
    hierarchical_model_status: str
    collector_rule: str
    collector_provenance: str
    segment_driver_assumption: str
    port_count_architecture_status: str
    port_connectivity_status: str
    cluster_count: int
    port_count: int
    active_port_count: int
    port_fan_in_counts: tuple[int, ...]
    port_fan_in_distribution: dict[int, int]
    fan_in_min: int
    fan_in_median: float
    fan_in_mean: float
    fan_in_max: int
    active_fan_in_min: int
    active_fan_in_median: float
    active_fan_in_mean: float
    active_fan_in_max: int
    ports: tuple[PortFanIn, ...]
    paths: tuple[MATToCoilPath, ...]
    legacy_direct_latency: LatencySummary
    hierarchical_latency: LatencySummary
    branch_wire_latency: LatencySummary
    shared_trunk_wire_latency: LatencySummary
    port_selection_latency_ns: float
    port_selection_status: str
    unmodeled_logic_interface_latency: tuple[str, ...]
    aggregation_load_multiplier: float
    aggregation_load_pF: float
    aggregation_load_status: str
    load_sensitivity: tuple[AggregationLoadSensitivity, ...]
    current_feol_data_path_audit: dict[str, str]
    taxonomy: dict[str, dict[str, str]]
    workload_weighted: bool
    contention_included: bool
    serialization_included: bool
    canonical_feol_mutated: bool
    unit_conversion: str

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["legacy_direct_latency"] = self.legacy_direct_latency.as_dict()
        value["hierarchical_latency"] = self.hierarchical_latency.as_dict()
        value["branch_wire_latency"] = self.branch_wire_latency.as_dict()
        value["shared_trunk_wire_latency"] = (
            self.shared_trunk_wire_latency.as_dict())
        value["ports"] = [port.as_dict() for port in self.ports]
        value["paths"] = [path.as_dict() for path in self.paths]
        value["load_sensitivity"] = [row.as_dict()
                                     for row in self.load_sensitivity]
        return value


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: tuple[float, ...]) -> LatencySummary:
    if not values:
        raise ValueError("latency summary requires a non-empty sample")
    return LatencySummary(
        min_ns=min(values),
        median_ns=statistics.median(values),
        p90_ns=_percentile(values, 0.9),
        max_ns=max(values),
        mean_ns=sum(values) / len(values),
    )


def _median_coordinate(values: tuple[float, ...]) -> float:
    return float(statistics.median(values))


def calculate_wire_rc_delay(
        *, length_um: float, resistance_ohm_per_um: float,
        capacitance_fF_per_um: float, driver_resistance_ohm: float,
        endpoint_load_pF: float, rise_target_fraction: float = 0.8,
        ) -> WireRCDelay:
    """Evaluate the repository's first-order distributed-RC equation.

    Unit closure is explicit: fF/um * um * 1e-3 gives pF, and Ohm*pF
    gives ps.  The final rise time is converted from ps to ns by 1e-3.
    """
    positive = {
        "length_um": length_um,
        "resistance_ohm_per_um": resistance_ohm_per_um,
        "capacitance_fF_per_um": capacitance_fF_per_um,
        "driver_resistance_ohm": driver_resistance_ohm,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(endpoint_load_pF) or endpoint_load_pF < 0.0:
        raise ValueError("endpoint_load_pF must be finite and non-negative")
    if not math.isfinite(rise_target_fraction) or not (
            0.0 < rise_target_fraction < 1.0):
        raise ValueError("rise_target_fraction must be between zero and one")

    wire_r = resistance_ohm_per_um * length_um
    wire_c = capacitance_fF_per_um * length_um * 1e-3
    driver = driver_resistance_ohm * (wire_c + endpoint_load_pF)
    wire_load = wire_r * endpoint_load_pF
    distributed = 0.5 * wire_r * wire_c
    tau = driver + wire_load + distributed
    delay = -math.log(1.0 - rise_target_fraction) * tau * 1e-3
    return WireRCDelay(
        length_um=length_um,
        wire_resistance_ohm=wire_r,
        wire_capacitance_pF=wire_c,
        endpoint_load_pF=endpoint_load_pF,
        driver_cap_time_constant_ps=driver,
        wire_load_time_constant_ps=wire_load,
        distributed_wire_time_constant_ps=distributed,
        time_constant_ps=tau,
        delay_ns=delay,
    )


def _collector_for_group(
        cluster_ids: tuple[int, ...], centers: tuple[tuple[float, float], ...],
        ) -> tuple[float, float]:
    """Return the deterministic Manhattan-distance median collector."""
    return (
        _median_coordinate(tuple(centers[index][0] for index in cluster_ids)),
        _median_coordinate(tuple(centers[index][1] for index in cluster_ids)),
    )


def _build_paths(
        feol: FEOLRouteResult, *, aggregation_load_multiplier: float,
        ) -> tuple[tuple[MATToCoilPath, ...], tuple[PortFanIn, ...]]:
    if not math.isfinite(aggregation_load_multiplier) or (
            aggregation_load_multiplier < 0.0):
        raise ValueError("aggregation load multiplier must be non-negative")
    required = (
        feol.feol_resistance_ohm_per_um,
        feol.feol_fixed_driver_resistance_ohm,
        feol.feol_fixed_load_pF,
    )
    if any(value is None for value in required):
        raise ValueError("hierarchical diagnostic requires resolved FEOL RC")
    resistance = float(feol.feol_resistance_ohm_per_um)
    driver = float(feol.feol_fixed_driver_resistance_ohm)
    endpoint_load = float(feol.feol_fixed_load_pF)
    capacitance = feol.feol_wire_capacitance_fF_per_um

    groups = tuple(tuple(
        index for index, assigned in enumerate(
            feol.feol_route_nearest_port_index) if assigned == port_id)
        for port_id in range(feol.feol_io_channel_count))
    collectors: dict[int, tuple[float, float]] = {
        port_id: _collector_for_group(cluster_ids,
                                      feol.feol_route_cluster_centers_um)
        for port_id, cluster_ids in enumerate(groups) if cluster_ids
    }

    paths: list[MATToCoilPath] = []
    ports: list[PortFanIn] = []
    extra_load = aggregation_load_multiplier * endpoint_load
    edge_parallel_y = feol.feol_io_edge.startswith("x_")
    for port_id, cluster_ids in enumerate(groups):
        if not cluster_ids:
            ports.append(PortFanIn(
                port_id=port_id, fan_in_clusters=0,
                collector_x_um=None, collector_y_um=None,
                shared_trunk_length_um=None,
                legacy_route_min_um=None, legacy_route_median_um=None,
                legacy_route_max_um=None, depth_from_edge_min_um=None,
                depth_from_edge_median_um=None, depth_from_edge_max_um=None,
            ))
            continue
        collector = collectors[port_id]
        port = feol.feol_io_channel_coordinates_um[port_id]
        trunk_length = (
            abs(collector[0] - port[0]) + abs(collector[1] - port[1]))
        trunk_rc = calculate_wire_rc_delay(
            length_um=trunk_length,
            resistance_ohm_per_um=resistance,
            capacitance_fF_per_um=capacitance,
            driver_resistance_ohm=driver,
            endpoint_load_pF=endpoint_load,
        )
        legacy_lengths = tuple(
            feol.feol_route_length_per_cluster_um[index]
            for index in cluster_ids)
        depths = tuple(
            (abs(feol.feol_route_cluster_centers_um[index][0] - port[0])
             if edge_parallel_y else
             abs(feol.feol_route_cluster_centers_um[index][1] - port[1]))
            for index in cluster_ids)
        ports.append(PortFanIn(
            port_id=port_id,
            fan_in_clusters=len(cluster_ids),
            collector_x_um=collector[0], collector_y_um=collector[1],
            shared_trunk_length_um=trunk_length,
            legacy_route_min_um=min(legacy_lengths),
            legacy_route_median_um=statistics.median(legacy_lengths),
            legacy_route_max_um=max(legacy_lengths),
            depth_from_edge_min_um=min(depths),
            depth_from_edge_median_um=statistics.median(depths),
            depth_from_edge_max_um=max(depths),
        ))
        for cluster_id in cluster_ids:
            center = feol.feol_route_cluster_centers_um[cluster_id]
            branch_length = (
                abs(center[0] - collector[0])
                + abs(center[1] - collector[1]))
            branch_rc = calculate_wire_rc_delay(
                length_um=branch_length,
                resistance_ohm_per_um=resistance,
                capacitance_fF_per_um=capacitance,
                driver_resistance_ohm=driver,
                endpoint_load_pF=extra_load,
            )
            structural = branch_rc.delay_ns + trunk_rc.delay_ns
            depth = (
                abs(center[0] - port[0]) if edge_parallel_y
                else abs(center[1] - port[1]))
            paths.append(MATToCoilPath(
                cluster_id=cluster_id,
                assigned_port=port_id,
                port_fan_in_clusters=len(cluster_ids),
                cluster_x_um=center[0], cluster_y_um=center[1],
                collector_x_um=collector[0], collector_y_um=collector[1],
                port_x_um=port[0], port_y_um=port[1],
                depth_from_edge_um=depth,
                branch_length_um=branch_length,
                trunk_length_um=trunk_length,
                total_physical_path_um=branch_length + trunk_length,
                legacy_direct_length_um=(
                    feol.feol_route_length_per_cluster_um[cluster_id]),
                branch_wire_delay_ns=branch_rc.delay_ns,
                shared_trunk_wire_delay_ns=trunk_rc.delay_ns,
                port_selection_delay_ns=0.0,
                structural_aggregation_delay_ns=structural,
                total_lateral_delay_ns=structural,
            ))
    return tuple(sorted(paths, key=lambda row: row.cluster_id)), tuple(ports)


def calculate_hierarchical_mat_to_coil(
        feol: FEOLRouteResult, *, aggregation_load_multiplier: float = 0.0,
        sensitivity_multipliers: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0),
        ) -> MATToCoilAudit:
    """Build the side-by-side direct and hierarchical lateral diagnostic."""
    direct = feol.feol_delay_per_cluster_ns
    if direct is None or feol.feol_fixed_load_pF is None:
        raise ValueError("resolved canonical FEOL latency is required")
    paths, ports = _build_paths(
        feol, aggregation_load_multiplier=aggregation_load_multiplier)
    if len(paths) != feol.feol_route_cluster_count:
        raise RuntimeError("every cluster must map to exactly one hierarchy path")
    counts = tuple(port.fan_in_clusters for port in ports)
    active_counts = tuple(value for value in counts if value > 0)
    if sum(counts) != feol.feol_route_cluster_count or not active_counts:
        raise RuntimeError("port fan-in must close to the cluster count")

    sensitivities: list[AggregationLoadSensitivity] = []
    for multiplier in sensitivity_multipliers:
        sensitivity_paths, _ = _build_paths(
            feol, aggregation_load_multiplier=multiplier)
        sensitivities.append(AggregationLoadSensitivity(
            aggregation_load_multiplier=multiplier,
            aggregation_load_pF=multiplier * feol.feol_fixed_load_pF,
            latency=_summary(tuple(
                path.total_lateral_delay_ns for path in sensitivity_paths)),
        ))

    hierarchical_values = tuple(path.total_lateral_delay_ns for path in paths)
    branch_values = tuple(path.branch_wire_delay_ns for path in paths)
    trunk_values = tuple(path.shared_trunk_wire_delay_ns for path in paths)
    return MATToCoilAudit(
        legacy_model_name="LEGACY_DIRECT_NEAREST_PORT_FEOL",
        hierarchical_model_name="HIERARCHICAL_MAT_TO_COIL_FEOL",
        hierarchical_model_status="HIERARCHICAL_TOPOLOGY_LOWER_BOUND",
        collector_rule="COORDINATE_WISE_MEDIAN_OF_ASSIGNED_CLUSTER_CENTERS",
        collector_provenance="GEOMETRY_DERIVED",
        segment_driver_assumption=(
            "CANONICAL_EFFECTIVE_RDRV_AT_BRANCH_SOURCE_AND_COLLECTOR_"
            "TRUNK_SOURCE__COLLECTOR_LOGIC_DELAY_ZERO_LOWER_BOUND"),
        port_count_architecture_status="ARCHITECTURE_FEATURE",
        port_connectivity_status="CONNECTIVITY_ASSUMPTION_NOT_VALIDATED",
        cluster_count=feol.feol_route_cluster_count,
        port_count=feol.feol_io_channel_count,
        active_port_count=len(active_counts),
        port_fan_in_counts=counts,
        port_fan_in_distribution=dict(sorted(Counter(counts).items())),
        fan_in_min=min(counts), fan_in_median=statistics.median(counts),
        fan_in_mean=sum(counts) / len(counts), fan_in_max=max(counts),
        active_fan_in_min=min(active_counts),
        active_fan_in_median=statistics.median(active_counts),
        active_fan_in_mean=sum(active_counts) / len(active_counts),
        active_fan_in_max=max(active_counts),
        ports=ports, paths=paths,
        legacy_direct_latency=_summary(tuple(direct)),
        hierarchical_latency=_summary(hierarchical_values),
        branch_wire_latency=_summary(branch_values),
        shared_trunk_wire_latency=_summary(trunk_values),
        port_selection_latency_ns=0.0,
        port_selection_status="NOT_CALIBRATED_ZERO_NS_LOWER_BOUND",
        unmodeled_logic_interface_latency=(
            "local_mux_logic_delay", "regional_collector_logic_delay",
            "port_arbiter_delay", "serializer_startup", "coil_tx_delay",
            "repeater_delay", "queueing_and_contention"),
        aggregation_load_multiplier=aggregation_load_multiplier,
        aggregation_load_pF=(
            aggregation_load_multiplier * feol.feol_fixed_load_pF),
        aggregation_load_status=(
            "ZERO_EXTRA_CAPACITANCE_TOPOLOGY_ONLY_LOWER_BOUND"
            if aggregation_load_multiplier == 0.0
            else "DIAGNOSTIC_SENSITIVITY_NOT_CANONICAL"),
        load_sensitivity=tuple(sensitivities),
        current_feol_data_path_audit={
            "implicit_path": (
                "CLUSTER_OR_MIV_FEOL_LANDING_TO_ONE_DEDICATED_MANHATTAN_"
                "RC_WIRE_TO_NEAREST_EDGE_PORT"),
            "code_boundary": (
                f"{feol.feol_route_start}_TO_{feol.feol_route_end}"),
            "explicit_mux": "NOT_MODELED",
            "regional_collector": "NOT_MODELED",
            "shared_global_bus": "NOT_MODELED",
            "arbitration": "NOT_MODELED",
            "serialization": "NOT_MODELED",
            "repeater_or_buffer": "NOT_MODELED",
            "port_fan_in_loading": "NOT_MODELED",
            "multiple_clusters_share_port": "ASSIGNMENT_ONLY_NOT_ELECTRICAL",
            "port_connectivity_validation": (
                "CONNECTIVITY_ASSUMPTION_NOT_VALIDATED"),
        },
        taxonomy={
            "L0_MAT_OUTPUT_NODE": {
                "status": "NOT_REQUIRED",
                "note": "boundary; MAT internal timing remains unchanged"},
            "L1_LOCAL_CLUSTER_OUTPUT_LOCAL_MUX": {
                "status": "LUMPED",
                "note": "geometry/energy exist; output latency not calibrated"},
            "L2_CLUSTER_TO_REGIONAL_AGGREGATION": {
                "status": "EXPLICITLY_MODELED",
                "note": "geometry-derived median-collector branch RC"},
            "L3_REGIONAL_GLOBAL_FEOL_TRANSPORT": {
                "status": "EXPLICITLY_MODELED",
                "note": "one shared collector-to-edge-port trunk per active port"},
            "L4_EDGE_PORT_FAN_IN_SELECTION": {
                "status": "LUMPED",
                "note": "fan-in topology explicit; logic delay is 0 ns lower bound"},
            "L5_COIL_TX_INPUT": {
                "status": "NOT_REQUIRED",
                "note": "model boundary; TX latency is not calibrated"},
        },
        workload_weighted=False,
        contention_included=False,
        serialization_included=False,
        canonical_feol_mutated=False,
        unit_conversion=(
            "FF_PER_UM_X_UM_X_1E-3_EQUALS_PF__OHM_X_PF_EQUALS_PS__"
            "PS_X_NEG_LOG_0P2_X_1E-3_EQUALS_NS"),
    )


def calculate_normalized_single_path_delay(
        feol: FEOLRouteResult, *, length_um: float = 11_000.0,
        ) -> WireRCDelay:
    """Evaluate a non-canonical Dream-like single path with M3D R/C."""
    if (feol.feol_resistance_ohm_per_um is None
            or feol.feol_fixed_driver_resistance_ohm is None
            or feol.feol_fixed_load_pF is None):
        raise ValueError("normalized path diagnostic requires resolved FEOL RC")
    return calculate_wire_rc_delay(
        length_um=length_um,
        resistance_ohm_per_um=feol.feol_resistance_ohm_per_um,
        capacitance_fF_per_um=feol.feol_wire_capacitance_fF_per_um,
        driver_resistance_ohm=feol.feol_fixed_driver_resistance_ohm,
        endpoint_load_pF=feol.feol_fixed_load_pF,
    )
