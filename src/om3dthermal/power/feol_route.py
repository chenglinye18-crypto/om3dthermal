"""Distributed cluster-to-nearest-edge FEOL routing model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

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
    feol_route_max_length_um: float
    feol_route_average_length_um: float
    feol_route_average_lateral_component_um: float
    feol_route_average_perpendicular_component_um: float
    feol_route_access_assumption: str
    feol_wire_capacitance_fF_per_um: float
    feol_wire_voltage_V: float
    feol_wire_activity_factor: float
    feol_wire_provenance: str
    feol_average_wire_capacitance_fF: float
    feol_route_energy_pj_per_bit: float
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


def calculate_feol_route(
        spec: FEOLRouteInput, topology: M3DSubarrayResult,
        ) -> FEOLRouteResult:
    """Map every cluster center to its nearest centered-bin edge channel."""
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
    # Framework convention: supply energy is alpha*C*V^2. fJ -> pJ is 1e-3.
    energy = (
        spec.wire.activity_factor
        * average_capacitance
        * spec.wire.voltage_V ** 2
        * 1e-3)
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
        feol_average_wire_capacitance_fF=average_capacitance,
        feol_route_energy_pj_per_bit=energy,
        feol_route_start="MIV_FEOL_LANDING",
        feol_route_end="EDGE_IO_INTERFACE_INPUT",
        interface_boundary="EDGE_IO_INTERFACE_INPUT_TO_GPU_RECEIVER",
        feol_route_topology_provenance=spec.topology_provenance,
        feol_io_channel_count_source=spec.io_channel_count_source,
        feol_serialization_applied=False,
    )
