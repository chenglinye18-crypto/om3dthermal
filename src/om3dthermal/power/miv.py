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


def _length_scaled_miv_capacitance(
        topology: MIVTopology, *, vertical_capacitance_pF_per_um: float,
        fixed_load_pF: float,
        ) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the shared per-layer distributed and effective capacitances."""
    if (not math.isfinite(vertical_capacitance_pF_per_um)
            or vertical_capacitance_pF_per_um <= 0.0):
        raise ValueError("MIV distributed capacitance slope must be positive")
    if not math.isfinite(fixed_load_pF) or fixed_load_pF <= 0.0:
        raise ValueError("MIV fixed load capacitance must be positive")
    lengths = topology.miv_length_per_layer_um
    if len(lengths) != topology.m3d_layers or not lengths:
        raise ValueError("MIV topology must contain one length per layer")
    if any(not math.isfinite(length) or length <= 0.0 for length in lengths):
        raise ValueError("MIV lengths must be finite and positive")
    if any(right <= left for left, right in zip(lengths, lengths[1:])):
        raise ValueError("MIV lengths must increase monotonically with layer")
    distributed = tuple(
        vertical_capacitance_pF_per_um * length for length in lengths)
    effective = tuple(fixed_load_pF + value for value in distributed)
    return distributed, effective


def calculate_length_scaled_miv_energy(
        topology: MIVTopology, *, vertical_capacitance_pF_per_um: float,
        fixed_load_pF: float,
        row_voltage_product_V2: float, col_voltage_product_V2: float,
        data_voltage_product_V2: float, data_pumps: int,
        data_transition_factor: float, control_address_reuse: int,
        atom_size_bits: int,
        ) -> MIVEnergy:
    """Resolve MIV energy with explicit address/control-selection reuse."""
    if topology.active_data_miv_count is None:
        raise ValueError("MIV serialization must be resolved before energy")
    if data_pumps <= 0 or control_address_reuse <= 0 or atom_size_bits <= 0:
        raise ValueError("MIV access accounting counts must be positive")
    distributed, effective = _length_scaled_miv_capacitance(
        topology,
        vertical_capacitance_pF_per_um=vertical_capacitance_pF_per_um,
        fixed_load_pF=fixed_load_pF,
    )
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
    denominator = control_address_reuse * atom_size_bits
    row_access = 1.5 * row / denominator
    col_access = control_address_reuse * col / denominator
    data_access = control_address_reuse * data / denominator
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


@dataclass(frozen=True)
class MIVLatency:
    """Pure per-layer first-order distributed-RC delay for one MIV."""

    miv_length_per_layer_um: tuple[float, ...]
    miv_effective_capacitance_per_layer_pF: tuple[float, ...]
    miv_wire_resistance_per_layer_ohm: tuple[float, ...]
    driver_cap_time_constant_component_per_layer_ps: tuple[float, ...]
    wire_load_time_constant_component_per_layer_ps: tuple[float, ...]
    distributed_wire_time_constant_component_per_layer_ps: tuple[float, ...]
    miv_time_constant_per_layer_ps: tuple[float, ...]
    driver_cap_delay_component_ns: tuple[float, ...]
    wire_load_delay_component_ns: tuple[float, ...]
    distributed_wire_delay_component_ns: tuple[float, ...]
    miv_delay_per_layer_ns: tuple[float, ...]
    miv_min_delay_ns: float
    miv_max_delay_ns: float
    miv_uniform_average_delay_ns: float
    miv_delay_spread_ns: float
    miv_far_near_ratio: float
    resistance_ohm_per_um: float
    fixed_driver_resistance_ohm: float
    rise_target_fraction: float
    model_name: str
    parameter_status: str
    provenance: str
    fixed_driver_resistance_provenance: str
    unit_conversion: str
    serialization_included: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def calculate_miv_propagation_latency(
        topology: MIVTopology, *, vertical_capacitance_pF_per_um: float,
        fixed_load_pF: float, miv_load_resistance_ohm: float,
        miv_resistance_ohm_per_um: float,
        rise_target_fraction: float = 0.8,
        parameter_status: str = "MODELING_SENSITIVITY_ONLY",
        provenance: str = "NO_MIV_SPECIFIC_RESISTANCE_PER_LENGTH_SOURCE",
        ) -> MIVLatency:
    """Resolve z-dependent MIV propagation delay without serialization.

    The Elmore-style time constant is ``Rdrv*(Cwire+Cload) +
    Rwire*Cload + 0.5*Rwire*Cwire``.  Ohm times pF is ps, so multiplying
    by ``-ln(1-target) * 1e-3`` converts the result to rise-time ns.
    The average is an unweighted geometric mean over layers and deliberately
    ignores ``topology.layer_access_probability``.
    """
    if (not math.isfinite(miv_load_resistance_ohm)
            or miv_load_resistance_ohm <= 0.0):
        raise ValueError(
            "MIV fixed driver resistance must be finite and positive")
    if (not math.isfinite(miv_resistance_ohm_per_um)
            or miv_resistance_ohm_per_um < 0.0):
        raise ValueError(
            "MIV resistance per unit length must be finite and non-negative")
    if (not math.isfinite(rise_target_fraction)
            or not 0.0 < rise_target_fraction < 1.0):
        raise ValueError("MIV rise target fraction must lie within (0, 1)")
    distributed, effective = _length_scaled_miv_capacitance(
        topology,
        vertical_capacitance_pF_per_um=vertical_capacitance_pF_per_um,
        fixed_load_pF=fixed_load_pF,
    )
    wire_resistances = tuple(
        miv_resistance_ohm_per_um * length
        for length in topology.miv_length_per_layer_um)
    driver_cap_terms = tuple(
        miv_load_resistance_ohm * capacitance_pF
        for capacitance_pF in effective)
    wire_load_terms = tuple(
        wire_resistance_ohm * fixed_load_pF
        for wire_resistance_ohm in wire_resistances)
    distributed_wire_terms = tuple(
        0.5 * wire_resistance_ohm * wire_capacitance_pF
        for wire_resistance_ohm, wire_capacitance_pF in zip(
            wire_resistances, distributed, strict=True))
    time_constants = tuple(
        driver + wire_load + wire
        for driver, wire_load, wire in zip(
            driver_cap_terms, wire_load_terms, distributed_wire_terms,
            strict=True))
    rise_coefficient = -math.log(1.0 - rise_target_fraction)
    # Unit closure: ohm * pF = ps; 1e-3 ns/ps converts every delay to ns.
    to_delay_ns = rise_coefficient * 1e-3
    driver_delays = tuple(value * to_delay_ns for value in driver_cap_terms)
    wire_load_delays = tuple(value * to_delay_ns for value in wire_load_terms)
    distributed_wire_delays = tuple(
        value * to_delay_ns for value in distributed_wire_terms)
    delays = tuple(value * to_delay_ns for value in time_constants)
    if any(right <= left for left, right in zip(delays, delays[1:])):
        raise RuntimeError("MIV propagation delay must increase with layer")
    minimum = min(delays)
    maximum = max(delays)
    return MIVLatency(
        miv_length_per_layer_um=topology.miv_length_per_layer_um,
        miv_effective_capacitance_per_layer_pF=effective,
        miv_wire_resistance_per_layer_ohm=wire_resistances,
        driver_cap_time_constant_component_per_layer_ps=driver_cap_terms,
        wire_load_time_constant_component_per_layer_ps=wire_load_terms,
        distributed_wire_time_constant_component_per_layer_ps=(
            distributed_wire_terms),
        miv_time_constant_per_layer_ps=time_constants,
        driver_cap_delay_component_ns=driver_delays,
        wire_load_delay_component_ns=wire_load_delays,
        distributed_wire_delay_component_ns=(
            distributed_wire_delays),
        miv_delay_per_layer_ns=delays,
        miv_min_delay_ns=minimum,
        miv_max_delay_ns=maximum,
        miv_uniform_average_delay_ns=sum(delays) / len(delays),
        miv_delay_spread_ns=maximum - minimum,
        miv_far_near_ratio=maximum / minimum,
        resistance_ohm_per_um=miv_resistance_ohm_per_um,
        fixed_driver_resistance_ohm=miv_load_resistance_ohm,
        rise_target_fraction=rise_target_fraction,
        model_name="FIRST_ORDER_DISTRIBUTED_RC_ELMORE",
        parameter_status=parameter_status,
        provenance=provenance,
        fixed_driver_resistance_provenance="DREAMRAM_REFERENCE_PLACEHOLDER",
        unit_conversion="OHM_TIMES_PF_EQUALS_PS_CONVERTED_TO_NS_BY_1E-3",
        serialization_included=False,
    )
