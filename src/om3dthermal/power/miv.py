"""Topology-only adapter for monolithic memory layers and shared MIVs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class MIVTopology:
    m3d_layers: int
    layer_pitch_um: float
    layer_access_assumption: str
    layer_access_probability: tuple[float, ...]
    miv_length_per_layer_um: tuple[float, ...]
    miv_average_length_um: float
    miv_segments_per_layer: tuple[int, ...]
    miv_average_segments: float
    data_width_before_vertical: int
    vertical_serialization_factor: int | None
    vertical_serialization_status: str
    active_data_miv_count: int | None
    row_miv_count: int
    col_miv_count: int
    vertical_interconnect_type: str
    miv_connection_model: str
    direct_bitline_to_feol: bool
    tsv_energy_included: bool
    base_route_included: bool
    dq_included: bool
    miv_dedicated_koz_area_modeled: bool
    miv_planar_footprint_basis: str
    miv_capacitance_status: str
    miv_energy_status: str
    miv_components: tuple[str, ...]
    excluded_hbm_components: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_miv_topology(
        *, m3d_layers: int, layer_pitch_um: float,
        data_width_before_vertical: int,
        vertical_serialization_factor: int | str,
        row_miv_count: int, col_miv_count: int,
        layer_access_probability: tuple[float, ...] | None = None,
        capacitance_fF: float | str = "unresolved",
        ) -> MIVTopology:
    """Resolve MIV lengths/counts without using TSV geometry or capacitance."""
    if m3d_layers <= 0 or layer_pitch_um <= 0.0:
        raise ValueError("M3D layers and layer pitch must be positive")
    if data_width_before_vertical <= 0:
        raise ValueError("data width before vertical must be positive")
    if row_miv_count <= 0 or col_miv_count <= 0:
        raise ValueError("row/column MIV counts must be positive")

    lengths = tuple(
        layer_pitch_um * layer_index
        for layer_index in range(1, m3d_layers + 1))
    if layer_access_probability is None:
        probabilities = (1.0 / m3d_layers,) * m3d_layers
        access_assumption = "uniform"
    else:
        if len(layer_access_probability) != m3d_layers:
            raise ValueError(
                "layer access probability count must equal m3d_layers")
        if (any(value < 0.0 for value in layer_access_probability)
                or not math.isclose(
                    sum(layer_access_probability), 1.0,
                    rel_tol=0.0, abs_tol=1e-12)):
            raise ValueError("layer access probabilities must sum to 1")
        probabilities = layer_access_probability
        access_assumption = "workload_configured"
    average_length = sum(
        probability * length
        for probability, length in zip(probabilities, lengths, strict=True))
    segments = tuple(range(1, m3d_layers + 1))
    average_segments = sum(
        probability * segment
        for probability, segment in zip(probabilities, segments, strict=True))

    if vertical_serialization_factor == "unresolved":
        serialization = None
        serialization_status = "unresolved"
        active_count = None
    else:
        serialization = int(vertical_serialization_factor)
        if serialization <= 0:
            raise ValueError("vertical serialization factor must be positive")
        serialization_status = "resolved"
        active_count = math.ceil(data_width_before_vertical / serialization)

    capacitance_status = (
        "unresolved" if capacitance_fF == "unresolved" else "resolved")
    energy_status = (
        "resolved"
        if capacitance_status == "resolved"
        and serialization_status == "resolved"
        else "unresolved")
    return MIVTopology(
        m3d_layers=m3d_layers,
        layer_pitch_um=layer_pitch_um,
        layer_access_assumption=access_assumption,
        layer_access_probability=probabilities,
        miv_length_per_layer_um=lengths,
        miv_average_length_um=average_length,
        miv_segments_per_layer=segments,
        miv_average_segments=average_segments,
        data_width_before_vertical=data_width_before_vertical,
        vertical_serialization_factor=serialization,
        vertical_serialization_status=serialization_status,
        active_data_miv_count=active_count,
        row_miv_count=row_miv_count,
        col_miv_count=col_miv_count,
        vertical_interconnect_type="MIV",
        miv_connection_model="per_layer_local_selection_to_shared_vertical",
        direct_bitline_to_feol=False,
        tsv_energy_included=False,
        base_route_included=False,
        dq_included=False,
        miv_dedicated_koz_area_modeled=False,
        miv_planar_footprint_basis="bankdie_without_tsv_koz_bands",
        miv_capacitance_status=capacitance_status,
        miv_energy_status=energy_status,
        miv_components=("row-miv", "col-miv", "data-miv"),
        excluded_hbm_components=(
            "row-tsv", "col-tsv", "tsv",
            "row-base", "col-base", "base",
            "row-dq", "col-dq", "dq"),
    )


@dataclass(frozen=True)
class MIVEnergy:
    miv_distributed_capacitance_per_layer_pF: tuple[float, ...]
    miv_effective_capacitance_per_layer_pF: tuple[float, ...]
    miv_average_distributed_capacitance_pF: float
    miv_average_effective_capacitance_pF: float
    row_miv_energy_pJ: float
    col_miv_energy_pJ: float
    data_miv_energy_pJ: float
    miv_total_energy_pJ: float
    row_miv_access_energy_pJ_per_bit: float
    col_miv_access_energy_pJ_per_bit: float
    data_miv_access_energy_pJ_per_bit: float
    miv_access_energy_pJ_per_bit: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def calculate_length_scaled_miv_energy(
        topology: MIVTopology, *, vertical_capacitance_pF_per_um: float,
        fixed_load_pF: float,
        row_voltage_product_V2: float, col_voltage_product_V2: float,
        data_voltage_product_V2: float, data_pumps: int,
        data_transition_factor: float, rd_per_act: int, atom_size_bits: int,
        ) -> MIVEnergy:
    """Use physical MIV length with a distributed-C reference and one load."""
    if topology.active_data_miv_count is None:
        raise ValueError("MIV serialization must be resolved before energy")
    if vertical_capacitance_pF_per_um <= 0.0 or fixed_load_pF <= 0.0:
        raise ValueError("MIV distributed capacitance and fixed load must be positive")
    if data_pumps <= 0 or rd_per_act <= 0 or atom_size_bits <= 0:
        raise ValueError("MIV access accounting counts must be positive")
    distributed = tuple(
        vertical_capacitance_pF_per_um * length
        for length in topology.miv_length_per_layer_um)
    effective = tuple(fixed_load_pF + value for value in distributed)
    average_distributed = sum(
        probability * value
        for probability, value in zip(
            topology.layer_access_probability, distributed, strict=True))
    average_effective = sum(
        probability * value
        for probability, value in zip(
            topology.layer_access_probability, effective, strict=True))
    row = (
        topology.row_miv_count / 2
        * average_effective * row_voltage_product_V2)
    col = (
        topology.col_miv_count / 2
        * average_effective * col_voltage_product_V2)
    data = (
        topology.active_data_miv_count * data_pumps / 2
        * average_effective * data_voltage_product_V2
        * data_transition_factor)
    denominator = rd_per_act * atom_size_bits
    row_access = 1.5 * row / denominator
    col_access = rd_per_act * col / denominator
    data_access = rd_per_act * data / denominator
    return MIVEnergy(
        miv_distributed_capacitance_per_layer_pF=distributed,
        miv_effective_capacitance_per_layer_pF=effective,
        miv_average_distributed_capacitance_pF=average_distributed,
        miv_average_effective_capacitance_pF=average_effective,
        row_miv_energy_pJ=row,
        col_miv_energy_pJ=col,
        data_miv_energy_pJ=data,
        miv_total_energy_pJ=row + col + data,
        row_miv_access_energy_pJ_per_bit=row_access,
        col_miv_access_energy_pJ_per_bit=col_access,
        data_miv_access_energy_pJ_per_bit=data_access,
        miv_access_energy_pJ_per_bit=row_access + col_access + data_access,
    )
