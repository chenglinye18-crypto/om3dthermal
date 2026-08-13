"""Tang-style shared-global / embedded-local M3D subarray topology."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .config import M3DSubarrayInput, RoutingElectricalInput
from .geometry import M3DGeometry


@dataclass(frozen=True)
class M3DSubarrayResult:
    slab_x_um: float
    slab_y_um: float
    cell_area_um2: float
    cell_area_source: str
    F_um: float
    cell_pitch_x_um: float
    cell_pitch_y_um: float
    core_width_um: float
    core_height_um: float
    shared_row_selection_band_um: float
    shared_column_write_selection_band_um: float
    row_selection_band_axis: str
    column_write_selection_band_axis: str
    usable_width_um: float
    usable_height_um: float
    local_mux_footprint_height_um: float
    subarray_width_um: float
    subarray_height_um: float
    nx: int
    ny: int
    nx_source: str
    ny_source: str
    subarrays_per_layer: int
    global_peripheral_instance_count: int
    local_mux_instances_per_layer: int
    bits_per_subarray: int
    bits_per_layer: int
    placed_width_um: float
    placed_height_um: float
    layout_utilization: float
    row_address_bits: int
    column_address_bits: int
    accessed_subarrays_per_access: int
    selected_bits_per_subarray: int
    delivered_bits_per_access: int
    data_width_before_vertical: int
    global_rwl_route_length_um: float
    global_wwl_route_length_um: float
    global_wbl_route_length_um: float
    local_rbl_route_length_um: float
    global_rwl_raw_energy_pJ_per_access: float
    global_wwl_raw_energy_pJ_per_access: float
    global_wbl_raw_energy_pJ_per_access: float
    global_rwl_energy_pj_per_bit: float
    global_wwl_energy_pj_per_bit: float
    global_wbl_energy_pj_per_bit: float
    global_control_routing_energy_pj_per_bit: float
    local_rbl_raw_energy_pJ_per_access: float
    local_mux_raw_energy_pJ_per_access: float
    local_rbl_energy_pj_per_bit: float
    local_mux_energy_pj_per_bit: float
    local_read_routing_energy_pj_per_bit: float
    memory_internal_energy_excluding_operation_pj_per_bit: float
    physical_topology: str
    topology_provenance: str
    access_provenance: str
    global_peripheral_provenance: str
    interconnect_electrical: dict[str, dict[str, object]]
    local_mux_provenance: str
    dreamram_hierarchy_included: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _address_bits(count: int) -> int:
    return max(1, math.ceil(math.log2(count)))


def _line_energy_pJ(
        electrical: RoutingElectricalInput, *, length_um: float,
        instance_multiplier: int = 1) -> float:
    # fF * V^2 is fJ; 1e-3 converts fJ to pJ. Resistance and physical metal
    # dimensions remain explicit inputs/diagnostics for future RC validation,
    # but are not silently converted into extra dynamic energy.
    return (
        instance_multiplier
        * electrical.active_line_count
        * electrical.activity_factor
        * electrical.capacitance_fF_per_um
        * length_um
        * electrical.voltage_V ** 2
        * 1e-3
    )


def _subtract_shared_band(
        *, slab_x: float, slab_y: float, row_band: float, col_band: float,
        row_axis: str, col_axis: str) -> tuple[float, float]:
    usable_x = slab_x
    usable_y = slab_y
    if row_axis == "x":
        usable_x -= row_band
    else:
        usable_y -= row_band
    if col_axis == "x":
        usable_x -= col_band
    else:
        usable_y -= col_band
    if usable_x <= 0.0 or usable_y <= 0.0:
        raise ValueError("shared global peripheral bands consume the M3D slab")
    return usable_x, usable_y


def calculate_m3d_subarray(
        spec: M3DSubarrayInput, geometry: M3DGeometry,
        ) -> M3DSubarrayResult:
    """Resolve shared global bands, local subarrays, and access energy."""
    F_um = math.sqrt(geometry.cell_area_um2) / 2.0
    pitch_um = 2.0 * F_um
    core = spec.subarray
    core_width = core.n_cols * pitch_um
    core_height = core.n_rows * pitch_um
    global_spec = spec.global_peripheral
    row_band = global_spec.row_selection_band_f * F_um
    col_band = global_spec.column_write_selection_band_f * F_um
    usable_width, usable_height = _subtract_shared_band(
        slab_x=geometry.slab_x_um,
        slab_y=geometry.slab_y_um,
        row_band=row_band,
        col_band=col_band,
        row_axis=global_spec.row_selection_band_axis,
        col_axis=global_spec.column_write_selection_band_axis,
    )
    mux_footprint = spec.local_mux.footprint_height_f * F_um
    subarray_width = core_width
    subarray_height = core_height + mux_footprint
    max_nx = math.floor(usable_width / subarray_width)
    max_ny = math.floor(usable_height / subarray_height)
    if max_nx < 1 or max_ny < 1:
        raise ValueError("one M3D subarray does not fit the usable slab")
    nx = max_nx if core.grid.nx == "auto" else core.grid.nx
    ny = max_ny if core.grid.ny == "auto" else core.grid.ny
    if nx > max_nx or ny > max_ny:
        raise ValueError(
            f"explicit M3D subarray layout {nx}x{ny} exceeds fit "
            f"limit {max_nx}x{max_ny}")
    subarrays = nx * ny
    access = spec.access
    if access.accessed_subarrays_per_access > subarrays:
        raise ValueError("accessed subarrays exceed instantiated subarrays")
    if access.selected_bits_per_subarray > core.n_cols:
        raise ValueError("selected bits per subarray exceed Ncol")
    delivered_bits = (
        access.accessed_subarrays_per_access
        * access.selected_bits_per_subarray)
    placed_width = nx * subarray_width
    placed_height = ny * subarray_height
    bits_per_subarray = core.n_rows * core.n_cols

    # Global structures are single shared layer-level networks. Their energy
    # depends on activated global lines, never on total instantiated subarrays.
    global_rwl_length = placed_width
    global_wwl_length = placed_width
    global_wbl_length = placed_height
    rwl_raw = _line_energy_pJ(
        spec.interconnect.global_rwl, length_um=global_rwl_length)
    wwl_raw = _line_energy_pJ(
        spec.interconnect.global_wwl, length_um=global_wwl_length)
    wbl_raw = _line_energy_pJ(
        spec.interconnect.global_wbl, length_um=global_wbl_length)

    # Local RBL and MUX are activated only in participating subarrays.
    local_rbl_length = core_height
    local_rbl_raw = _line_energy_pJ(
        spec.interconnect.local_rbl,
        length_um=local_rbl_length,
        instance_multiplier=access.accessed_subarrays_per_access,
    )
    mux_raw = (
        spec.local_mux.energy_pj_per_selected_bit * delivered_bits)
    rwl = rwl_raw / delivered_bits
    wwl = wwl_raw / delivered_bits
    wbl = wbl_raw / delivered_bits
    local_rbl = local_rbl_raw / delivered_bits
    mux = mux_raw / delivered_bits
    global_energy = rwl + wwl + wbl
    local_energy = local_rbl + mux

    return M3DSubarrayResult(
        slab_x_um=geometry.slab_x_um,
        slab_y_um=geometry.slab_y_um,
        cell_area_um2=geometry.cell_area_um2,
        cell_area_source="geometry_source.m3d_memory.cell_area_um2",
        F_um=F_um,
        cell_pitch_x_um=pitch_um,
        cell_pitch_y_um=pitch_um,
        core_width_um=core_width,
        core_height_um=core_height,
        shared_row_selection_band_um=row_band,
        shared_column_write_selection_band_um=col_band,
        row_selection_band_axis=global_spec.row_selection_band_axis,
        column_write_selection_band_axis=(
            global_spec.column_write_selection_band_axis),
        usable_width_um=usable_width,
        usable_height_um=usable_height,
        local_mux_footprint_height_um=mux_footprint,
        subarray_width_um=subarray_width,
        subarray_height_um=subarray_height,
        nx=nx,
        ny=ny,
        nx_source="auto_floor" if core.grid.nx == "auto" else "explicit_override",
        ny_source="auto_floor" if core.grid.ny == "auto" else "explicit_override",
        subarrays_per_layer=subarrays,
        global_peripheral_instance_count=1,
        local_mux_instances_per_layer=subarrays,
        bits_per_subarray=bits_per_subarray,
        bits_per_layer=subarrays * bits_per_subarray,
        placed_width_um=placed_width,
        placed_height_um=placed_height,
        layout_utilization=(
            (placed_width * placed_height + row_band * geometry.slab_y_um
             + col_band * geometry.slab_x_um - row_band * col_band)
            / (geometry.slab_x_um * geometry.slab_y_um)),
        row_address_bits=_address_bits(ny) + _address_bits(core.n_rows),
        column_address_bits=_address_bits(nx) + _address_bits(core.n_cols),
        accessed_subarrays_per_access=access.accessed_subarrays_per_access,
        selected_bits_per_subarray=access.selected_bits_per_subarray,
        delivered_bits_per_access=delivered_bits,
        data_width_before_vertical=delivered_bits,
        global_rwl_route_length_um=global_rwl_length,
        global_wwl_route_length_um=global_wwl_length,
        global_wbl_route_length_um=global_wbl_length,
        local_rbl_route_length_um=local_rbl_length,
        global_rwl_raw_energy_pJ_per_access=rwl_raw,
        global_wwl_raw_energy_pJ_per_access=wwl_raw,
        global_wbl_raw_energy_pJ_per_access=wbl_raw,
        global_rwl_energy_pj_per_bit=rwl,
        global_wwl_energy_pj_per_bit=wwl,
        global_wbl_energy_pj_per_bit=wbl,
        global_control_routing_energy_pj_per_bit=global_energy,
        local_rbl_raw_energy_pJ_per_access=local_rbl_raw,
        local_mux_raw_energy_pJ_per_access=mux_raw,
        local_rbl_energy_pj_per_bit=local_rbl,
        local_mux_energy_pj_per_bit=mux,
        local_read_routing_energy_pj_per_bit=local_energy,
        memory_internal_energy_excluding_operation_pj_per_bit=(
            global_energy + local_energy),
        physical_topology="tang_shared_global_embedded_local_subarray",
        topology_provenance=spec.topology_provenance,
        access_provenance=access.provenance,
        global_peripheral_provenance=global_spec.provenance,
        interconnect_electrical={
            "global_rwl": spec.interconnect.global_rwl.model_dump(),
            "global_wwl": spec.interconnect.global_wwl.model_dump(),
            "global_wbl": spec.interconnect.global_wbl.model_dump(),
            "local_rbl": spec.interconnect.local_rbl.model_dump(),
        },
        local_mux_provenance=spec.local_mux.provenance,
        dreamram_hierarchy_included=False,
    )
