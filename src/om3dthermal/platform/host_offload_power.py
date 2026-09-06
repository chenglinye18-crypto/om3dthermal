"""Canonical incremental dynamic-power model for host DDR + PCIe offload."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict


class HostOffloadPowerOperatingPoint(BaseModel):
    """Resolved host transport rate and its dynamic-only power components."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host_bandwidth_demand_bytes_per_second: float
    host_bandwidth_actual_bytes_per_second: float
    host_bandwidth_utilization: float
    host_bandwidth_saturated: bool
    pcie_dynamic_power_W: float
    ddr_dynamic_power_W: float
    host_offload_dynamic_power_W: float


def resolve_host_offload_power(
    *,
    host_transfer_demand_bytes_per_second: float,
    host_effective_bandwidth_bytes_per_second: float,
    e_pcie_dynamic_J_per_bit: float,
    e_ddr_dynamic_J_per_bit: float,
) -> HostOffloadPowerOperatingPoint:
    """Clamp host traffic at its transport ceiling and resolve dynamic power.

    ``host_bandwidth_saturated`` uses strict exceedance semantics: an exact
    boundary operating point is not flagged as saturated, while any demand
    above the boundary is.  No static/idle host power or compute branch is
    part of this model.
    """
    values = {
        "host_transfer_demand_bytes_per_second": (
            host_transfer_demand_bytes_per_second),
        "host_effective_bandwidth_bytes_per_second": (
            host_effective_bandwidth_bytes_per_second),
        "e_pcie_dynamic_J_per_bit": e_pcie_dynamic_J_per_bit,
        "e_ddr_dynamic_J_per_bit": e_ddr_dynamic_J_per_bit,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if host_transfer_demand_bytes_per_second < 0.0:
        raise ValueError(
            "host_transfer_demand_bytes_per_second must be non-negative")
    if host_effective_bandwidth_bytes_per_second <= 0.0:
        raise ValueError(
            "host_effective_bandwidth_bytes_per_second must be positive")
    if e_pcie_dynamic_J_per_bit < 0.0 or e_ddr_dynamic_J_per_bit < 0.0:
        raise ValueError("host dynamic energy coefficients must be non-negative")

    actual = min(
        host_transfer_demand_bytes_per_second,
        host_effective_bandwidth_bytes_per_second,
    )
    bit_rate_actual = 8.0 * actual
    pcie_power = e_pcie_dynamic_J_per_bit * bit_rate_actual
    ddr_power = e_ddr_dynamic_J_per_bit * bit_rate_actual
    return HostOffloadPowerOperatingPoint(
        host_bandwidth_demand_bytes_per_second=(
            host_transfer_demand_bytes_per_second),
        host_bandwidth_actual_bytes_per_second=actual,
        host_bandwidth_utilization=(
            actual / host_effective_bandwidth_bytes_per_second),
        host_bandwidth_saturated=(
            host_transfer_demand_bytes_per_second
            > host_effective_bandwidth_bytes_per_second),
        pcie_dynamic_power_W=pcie_power,
        ddr_dynamic_power_W=ddr_power,
        host_offload_dynamic_power_W=pcie_power + ddr_power,
    )
