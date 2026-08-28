"""Distributed cluster-to-nearest-edge FEOL routing model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics

from .config import FEOLRouteInput
from .m3d_subarray import M3DSubarrayResult


@dataclass(frozen=True)
class FEOLRouteResult:
    feol_route_type: str
    feol_io_edge: str
    feol_io_channel_count: int
    feol_io_channel_pitch_um: float
    feol_io_channel_distribution: str
    feol_io_channel_coordinates_um: tuple[tuple[float, float], ...]
    feol_route_cluster_count: int
    feol_route_cluster_centers_um: tuple[tuple[float, float], ...]
    feol_route_nearest_port_index: tuple[int, ...]
    feol_route_length_per_cluster_um: tuple[float, ...]
    feol_route_lateral_component_per_cluster_um: tuple[float, ...]
    feol_route_perpendicular_component_per_cluster_um: tuple[float, ...]
    feol_route_min_length_um: float
    feol_route_median_length_um: float
    feol_route_p90_length_um: float
    feol_route_max_length_um: float
    feol_route_average_length_um: float
    feol_route_average_lateral_component_um: float
    feol_route_average_perpendicular_component_um: float
    feol_route_access_assumption: str
    feol_wire_capacitance_fF_per_um: float
    feol_wire_voltage_V: float
    feol_wire_activity_factor: float
    feol_wire_provenance: str
    feol_wire_capacitance_per_cluster_pF: tuple[float, ...]
    feol_average_wire_capacitance_fF: float
    feol_route_energy_pj_per_bit: float
    feol_wire_resistance_per_cluster_ohm: tuple[float, ...] | None
    feol_driver_cap_time_constant_component_per_cluster_ps: (
        tuple[float, ...] | None)
    feol_wire_load_time_constant_component_per_cluster_ps: (
        tuple[float, ...] | None)
    feol_distributed_wire_time_constant_component_per_cluster_ps: (
        tuple[float, ...] | None)
    feol_time_constant_per_cluster_ps: tuple[float, ...] | None
    feol_driver_cap_delay_component_ns: tuple[float, ...] | None
    feol_wire_load_delay_component_ns: tuple[float, ...] | None
    feol_distributed_wire_delay_component_ns: tuple[float, ...] | None
    feol_delay_per_cluster_ns: tuple[float, ...] | None
    feol_min_delay_ns: float | None
    feol_median_delay_ns: float | None
    feol_p90_delay_ns: float | None
    feol_max_delay_ns: float | None
    feol_uniform_average_delay_ns: float | None
    feol_delay_spread_ns: float | None
    feol_far_near_ratio: float | None
    feol_resistance_ohm_per_um: float | None
    feol_capacitance_pF_per_um: float | None
    feol_fixed_driver_resistance_ohm: float | None
    feol_fixed_load_pF: float | None
    feol_rise_target_fraction: float | None
    feol_latency_model_name: str | None
    feol_latency_provenance: dict[str, dict[str, str]] | None
    feol_latency_status: str | None
    feol_latency_unit_conversion: str | None
    feol_latency_workload_weighted: bool
    feol_latency_serialization_included: bool
    feol_route_start: str
    feol_route_end: str
    interface_boundary: str
    feol_route_topology_provenance: str
    feol_io_channel_count_source: str
    feol_serialization_applied: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _cluster_centers(
        topology: M3DSubarrayResult) -> tuple[tuple[float, float], ...]:
    pitch_x = topology.cluster_width_um + topology.cluster_gap_x_um
    pitch_y = topology.cluster_height_um + topology.cluster_gap_y_um
    return tuple(
        (
            ix * pitch_x + topology.cluster_width_um / 2.0,
            iy * pitch_y + topology.cluster_height_um / 2.0,
        )
        for iy in range(topology.cluster_count_y)
        for ix in range(topology.cluster_count_x)
    )


def _edge_ports(
        *, edge: str, count: int, slab_x_um: float, slab_y_um: float,
        ) -> tuple[float, tuple[tuple[float, float], ...]]:
    parallel_length = slab_x_um if edge.startswith("y_") else slab_y_um
    pitch = parallel_length / count
    parallel = tuple((index + 0.5) * pitch for index in range(count))
    if edge == "y_min":
        ports = tuple((position, 0.0) for position in parallel)
    elif edge == "y_max":
        ports = tuple((position, slab_y_um) for position in parallel)
    elif edge == "x_min":
        ports = tuple((0.0, position) for position in parallel)
    else:
        ports = tuple((slab_x_um, position) for position in parallel)
    return pitch, ports


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    """Return a linearly interpolated percentile of a non-empty sample."""
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def calculate_feol_route(
        spec: FEOLRouteInput, topology: M3DSubarrayResult,
        ) -> FEOLRouteResult:
    """Map every cluster center to its nearest centered-bin edge channel."""
    if (not math.isfinite(spec.wire.capacitance_fF_per_um)
            or spec.wire.capacitance_fF_per_um <= 0.0):
        raise ValueError("FEOL capacitance per length must be positive")
    centers = _cluster_centers(topology)
    pitch, ports = _edge_ports(
        edge=spec.edge,
        count=spec.io_channels,
        slab_x_um=topology.slab_x_um,
        slab_y_um=topology.slab_y_um,
    )
    nearest_indices: list[int] = []
    lengths: list[float] = []
    lateral_components: list[float] = []
    perpendicular_components: list[float] = []
    for xc, yc in centers:
        distances = tuple(abs(xc - xp) + abs(yc - yp) for xp, yp in ports)
        nearest = min(range(len(ports)), key=distances.__getitem__)
        xp, yp = ports[nearest]
        lateral = abs(xc - xp) if spec.edge.startswith("y_") else abs(yc - yp)
        perpendicular = (
            abs(yc - yp) if spec.edge.startswith("y_") else abs(xc - xp))
        nearest_indices.append(nearest)
        lateral_components.append(lateral)
        perpendicular_components.append(perpendicular)
        lengths.append(lateral + perpendicular)
    average_length = sum(lengths) / len(lengths)
    average_capacitance = (
        spec.wire.capacitance_fF_per_um * average_length)
    wire_capacitances_pF = tuple(
        spec.wire.capacitance_fF_per_um * length * 1e-3
        for length in lengths)
    # Framework convention: supply energy is alpha*C*V^2. fJ -> pJ is 1e-3.
    energy = (
        spec.wire.activity_factor
        * average_capacitance
        * spec.wire.voltage_V ** 2
        * 1e-3)

    wire_resistances = None
    driver_cap_terms = None
    wire_load_terms = None
    distributed_wire_terms = None
    time_constants = None
    driver_cap_delays = None
    wire_load_delays = None
    distributed_wire_delays = None
    delays = None
    min_delay = None
    median_delay = None
    p90_delay = None
    max_delay = None
    average_delay = None
    delay_spread = None
    far_near_ratio = None
    capacitance_pF_per_um = None
    model_name = None
    latency_provenance = None
    latency_status = None
    unit_conversion = None
    rise_target = None
    resistance_per_um = spec.wire.resistance_ohm_per_um
    driver_resistance = spec.wire.fixed_driver_resistance_ohm
    fixed_load = spec.wire.fixed_load_pF
    if resistance_per_um is not None:
        if (not math.isfinite(resistance_per_um)
                or resistance_per_um <= 0.0):
            raise ValueError("FEOL resistance per length must be positive")
        if (driver_resistance is None
                or not math.isfinite(driver_resistance)
                or driver_resistance <= 0.0):
            raise ValueError("FEOL fixed driver resistance must be positive")
        if (fixed_load is None
                or not math.isfinite(fixed_load)
                or fixed_load <= 0.0):
            raise ValueError("FEOL fixed load capacitance must be positive")
        provenance_inputs = (
            spec.wire.resistance_provenance,
            spec.wire.driver_resistance_provenance,
            spec.wire.load_capacitance_provenance,
        )
        if any(value is None for value in provenance_inputs):
            raise ValueError("FEOL latency parameter provenance is required")
        capacitance_pF_per_um = spec.wire.capacitance_fF_per_um * 1e-3
        wire_resistances = tuple(
            resistance_per_um * length for length in lengths)
        driver_cap_terms = tuple(
            driver_resistance * (wire_capacitance + fixed_load)
            for wire_capacitance in wire_capacitances_pF)
        wire_load_terms = tuple(
            wire_resistance * fixed_load
            for wire_resistance in wire_resistances)
        distributed_wire_terms = tuple(
            0.5 * wire_resistance * wire_capacitance
            for wire_resistance, wire_capacitance in zip(
                wire_resistances, wire_capacitances_pF, strict=True))
        time_constants = tuple(
            driver + wire_load + distributed
            for driver, wire_load, distributed in zip(
                driver_cap_terms, wire_load_terms, distributed_wire_terms,
                strict=True))
        rise_target = 0.8
        rise_coefficient = -math.log(1.0 - rise_target)
        to_delay_ns = rise_coefficient * 1e-3
        driver_cap_delays = tuple(
            value * to_delay_ns for value in driver_cap_terms)
        wire_load_delays = tuple(
            value * to_delay_ns for value in wire_load_terms)
        distributed_wire_delays = tuple(
            value * to_delay_ns for value in distributed_wire_terms)
        delays = tuple(value * to_delay_ns for value in time_constants)
        min_delay = min(delays)
        median_delay = statistics.median(delays)
        p90_delay = _percentile(delays, 0.9)
        max_delay = max(delays)
        average_delay = sum(delays) / len(delays)
        delay_spread = max_delay - min_delay
        far_near_ratio = max_delay / min_delay
        model_name = "FIRST_ORDER_DISTRIBUTED_RC_ELMORE"
        latency_provenance = {
            "resistance": spec.wire.resistance_provenance.model_dump(),
            "driver_resistance": (
                spec.wire.driver_resistance_provenance.model_dump()),
            "load_capacitance": (
                spec.wire.load_capacitance_provenance.model_dump()),
        }
        latency_status = "CONDITIONAL_MODELING_CHOICE"
        unit_conversion = (
            "FF_PER_UM_TO_PF_PER_UM_BY_1E-3__"
            "OHM_TIMES_PF_EQUALS_PS__PS_TO_NS_BY_1E-3")
    return FEOLRouteResult(
        feol_route_type=spec.type,
        feol_io_edge=spec.edge,
        feol_io_channel_count=spec.io_channels,
        feol_io_channel_pitch_um=pitch,
        feol_io_channel_distribution=spec.io_channel_distribution,
        feol_io_channel_coordinates_um=ports,
        feol_route_cluster_count=len(centers),
        feol_route_cluster_centers_um=centers,
        feol_route_nearest_port_index=tuple(nearest_indices),
        feol_route_length_per_cluster_um=tuple(lengths),
        feol_route_lateral_component_per_cluster_um=tuple(lateral_components),
        feol_route_perpendicular_component_per_cluster_um=(
            tuple(perpendicular_components)),
        feol_route_min_length_um=min(lengths),
        feol_route_median_length_um=statistics.median(lengths),
        feol_route_p90_length_um=_percentile(tuple(lengths), 0.9),
        feol_route_max_length_um=max(lengths),
        feol_route_average_length_um=average_length,
        feol_route_average_lateral_component_um=(
            sum(lateral_components) / len(lateral_components)),
        feol_route_average_perpendicular_component_um=(
            sum(perpendicular_components) / len(perpendicular_components)),
        feol_route_access_assumption=spec.access_assumption,
        feol_wire_capacitance_fF_per_um=spec.wire.capacitance_fF_per_um,
        feol_wire_voltage_V=spec.wire.voltage_V,
        feol_wire_activity_factor=spec.wire.activity_factor,
        feol_wire_provenance=spec.wire.provenance,
        feol_wire_capacitance_per_cluster_pF=wire_capacitances_pF,
        feol_average_wire_capacitance_fF=average_capacitance,
        feol_route_energy_pj_per_bit=energy,
        feol_wire_resistance_per_cluster_ohm=wire_resistances,
        feol_driver_cap_time_constant_component_per_cluster_ps=(
            driver_cap_terms),
        feol_wire_load_time_constant_component_per_cluster_ps=(
            wire_load_terms),
        feol_distributed_wire_time_constant_component_per_cluster_ps=(
            distributed_wire_terms),
        feol_time_constant_per_cluster_ps=time_constants,
        feol_driver_cap_delay_component_ns=driver_cap_delays,
        feol_wire_load_delay_component_ns=wire_load_delays,
        feol_distributed_wire_delay_component_ns=distributed_wire_delays,
        feol_delay_per_cluster_ns=delays,
        feol_min_delay_ns=min_delay,
        feol_median_delay_ns=median_delay,
        feol_p90_delay_ns=p90_delay,
        feol_max_delay_ns=max_delay,
        feol_uniform_average_delay_ns=average_delay,
        feol_delay_spread_ns=delay_spread,
        feol_far_near_ratio=far_near_ratio,
        feol_resistance_ohm_per_um=resistance_per_um,
        feol_capacitance_pF_per_um=capacitance_pF_per_um,
        feol_fixed_driver_resistance_ohm=driver_resistance,
        feol_fixed_load_pF=fixed_load,
        feol_rise_target_fraction=rise_target,
        feol_latency_model_name=model_name,
        feol_latency_provenance=latency_provenance,
        feol_latency_status=latency_status,
        feol_latency_unit_conversion=unit_conversion,
        feol_latency_workload_weighted=False,
        feol_latency_serialization_included=False,
        feol_route_start="MIV_FEOL_LANDING",
        feol_route_end="EDGE_IO_INTERFACE_INPUT",
        interface_boundary="EDGE_IO_INTERFACE_INPUT_TO_GPU_RECEIVER",
        feol_route_topology_provenance=spec.topology_provenance,
        feol_io_channel_count_source=spec.io_channel_count_source,
        feol_serialization_applied=False,
    )
