"""Topology-only adapter for monolithic memory layers and shared MIVs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class MIVTopology:
    m3d_layers: int
    layer_pitch_um: float
    layer_access_assumption: str
    miv_length_per_layer_um: tuple[float, ...]
    miv_average_length_um: float
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
    return MIVTopology(
        m3d_layers=m3d_layers,
        layer_pitch_um=layer_pitch_um,
        layer_access_assumption=access_assumption,
        miv_length_per_layer_um=lengths,
        miv_average_length_um=average_length,
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
        miv_energy_status="unresolved",
        miv_components=("row-miv", "col-miv", "data-miv"),
        excluded_hbm_components=(
            "row-tsv", "col-tsv", "tsv",
            "row-base", "col-base", "base",
            "row-dq", "col-dq", "dq"),
    )
