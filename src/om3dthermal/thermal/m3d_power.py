"""Workload-dependent power accounting for Orthogonal M3D 2T0C eDRAM."""

from __future__ import annotations

from dataclasses import dataclass

from om3dthermal.config import (
    M3DOperationEnergyPowerConfig,
    OrthogonalM3DTemplateConfig,
)


FEMTOJOULE_TO_JOULE = 1e-15


class UnresolvedM3DActivityError(ValueError):
    """Operation-energy mode was requested without complete activity."""

    def __init__(self, parameters: list[str]):
        self.parameters = tuple(parameters)
        joined = ", ".join(parameters)
        super().__init__(
            "operation_energy power mode requires resolved activity; "
            f"unresolved parameters: {joined}")


@dataclass(frozen=True)
class M3DOperationPowerBreakdown:
    read_W: float
    write_W: float
    refresh_W: float
    hold_W: float
    memory_total_W: float

    @property
    def dynamic_W(self) -> float:
        return self.read_W + self.write_W + self.refresh_W


@dataclass(frozen=True)
class M3DMemoryPowerResolution:
    mode: str
    memory_total_W: float
    per_bitcell_layer_W: float
    target_region: str
    operation_breakdown: M3DOperationPowerBreakdown | None = None


def femtojoules_to_joules(energy_fJ: float) -> float:
    """Convert operation energy to joules; this is not a power conversion."""
    return float(energy_fJ) * FEMTOJOULE_TO_JOULE


def calculate_array_read_power(
        *, delivered_bandwidth_bit_per_s: float,
        read_fraction: float,
        state_0_probability: float,
        state_1_probability: float,
        read_0_energy_fJ_per_bit: float,
        read_1_energy_fJ_per_bit: float) -> float:
    """Matched-bandwidth array-read power without peripheral assumptions."""
    if delivered_bandwidth_bit_per_s <= 0:
        raise ValueError("delivered_bandwidth_bit_per_s must be positive")
    if not 0.0 <= read_fraction <= 1.0:
        raise ValueError("read_fraction must be within [0, 1]")
    if not all(0.0 <= p <= 1.0 for p in (
            state_0_probability, state_1_probability)):
        raise ValueError("read-state probabilities must be within [0, 1]")
    if abs(state_0_probability + state_1_probability - 1.0) > 1e-12:
        raise ValueError("read-state probabilities must sum to 1")
    read_bit_rate = delivered_bandwidth_bit_per_s * read_fraction
    weighted_energy_J = (
        state_0_probability * femtojoules_to_joules(
            read_0_energy_fJ_per_bit)
        + state_1_probability * femtojoules_to_joules(
            read_1_energy_fJ_per_bit))
    return read_bit_rate * weighted_energy_J


def calculate_operation_energy_power(
        model: M3DOperationEnergyPowerConfig,
        *, total_memory_bits: float) -> M3DOperationPowerBreakdown:
    """Calculate 2T0C read/write/refresh/hold power from resolved activity.

    ``total_memory_bits`` is used only to convert refresh period into refreshed
    bit operations per second. No bandwidth, state probability, or active-row
    default is inferred.
    """
    if total_memory_bits <= 0:
        raise ValueError("total_memory_bits must be positive")
    unresolved = model.activity.unresolved_parameters()
    if unresolved:
        raise UnresolvedM3DActivityError(unresolved)

    activity = model.activity
    energy = model.operation_energy_fJ_per_bit
    read_rate = float(activity.read_bit_rate_per_s)
    write_rate = float(activity.write_bit_rate_per_s)
    refresh_rate = total_memory_bits / float(activity.refresh_period_s)

    read_W = read_rate * (
        float(activity.read_state_probability.p0)
        * femtojoules_to_joules(energy.read_0)
        + float(activity.read_state_probability.p1)
        * femtojoules_to_joules(energy.read_1))
    write_probability = activity.write_transition_probability
    write_W = write_rate * (
        float(write_probability.p00)
        * femtojoules_to_joules(energy.write_0_to_0)
        + float(write_probability.p01)
        * femtojoules_to_joules(energy.write_0_to_1)
        + float(write_probability.p10)
        * femtojoules_to_joules(energy.write_1_to_0)
        + float(write_probability.p11)
        * femtojoules_to_joules(energy.write_1_to_1))
    refresh_W = refresh_rate * (
        float(activity.refresh_state_probability.p0)
        * femtojoules_to_joules(energy.refresh_0)
        + float(activity.refresh_state_probability.p1)
        * femtojoules_to_joules(energy.refresh_1))
    hold_W = float(activity.active_rows) * model.hold_power_W_per_row
    total_W = read_W + write_W + refresh_W + hold_W
    return M3DOperationPowerBreakdown(
        read_W=read_W,
        write_W=write_W,
        refresh_W=refresh_W,
        hold_W=hold_W,
        memory_total_W=total_W,
    )


def resolve_m3d_memory_power(
        template: OrthogonalM3DTemplateConfig,
        *, mode: str | None = None) -> M3DMemoryPowerResolution:
    """Resolve the selected M3D memory-power model without mapping cells."""
    selected = mode or template.power.default_mode
    layers = template.m3d_memory.layers
    if selected == "iso_total":
        model = template.power_models.iso_total
        return M3DMemoryPowerResolution(
            mode=selected,
            memory_total_W=model.memory_total_W,
            per_bitcell_layer_W=model.memory_total_W / layers,
            target_region=model.distribution.target_region,
        )
    if selected == "operation_energy":
        model = template.power_models.operation_energy
        workload = model.nominal_workload
        energy = model.operation_energy_fJ_per_bit
        probability = workload.read_state_probability
        read_W = calculate_array_read_power(
            delivered_bandwidth_bit_per_s=(
                workload.delivered_bandwidth_bit_per_s),
            read_fraction=workload.read_fraction,
            state_0_probability=float(probability.p0),
            state_1_probability=float(probability.p1),
            read_0_energy_fJ_per_bit=energy.read_0,
            read_1_energy_fJ_per_bit=energy.read_1,
        )
        breakdown = M3DOperationPowerBreakdown(
            read_W=read_W,
            write_W=0.0,
            refresh_W=0.0,
            hold_W=0.0,
            memory_total_W=read_W,
        )
        return M3DMemoryPowerResolution(
            mode=selected,
            memory_total_W=breakdown.memory_total_W,
            per_bitcell_layer_W=breakdown.memory_total_W / layers,
            target_region=model.distribution.target_region,
            operation_breakdown=breakdown,
        )
    raise ValueError(
        f"unsupported M3D power mode {selected!r}; expected "
        "'iso_total' or 'operation_energy'")
