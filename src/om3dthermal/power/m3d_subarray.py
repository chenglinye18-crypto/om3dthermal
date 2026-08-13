"""Tang-style SubarrayCluster topology for orthogonal M3D memory."""

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
    Nrow: int
    Ncol: int
    core_width_um: float
    core_height_um: float
    subarray_width_um: float
    subarray_height_um: float
    local_mux_footprint_height_um: float
    shared_row_selection_band_um: float
    shared_column_write_selection_band_um: float
    row_selection_band_axis: str
    column_write_selection_band_axis: str
    subarray_gap_x_f: float
    subarray_gap_y_f: float
    subarray_gap_x_um: float
    subarray_gap_y_um: float
    cluster_gap_x_f: float
    cluster_gap_y_f: float
    cluster_gap_x_um: float
    cluster_gap_y_um: float
    spacing_provenance: str
    cluster_subarrays_x: int
    cluster_subarrays_y: int
    subarrays_per_cluster: int
    cluster_array_width_without_spacing_um: float
    cluster_array_height_without_spacing_um: float
    subarray_spacing_width_overhead_um: float
    subarray_spacing_height_overhead_um: float
    cluster_array_width_um: float
    cluster_array_height_um: float
    cluster_width_um: float
    cluster_height_um: float
    cluster_count_x: int
    cluster_count_y: int
    cluster_grid_x_source: str
    cluster_grid_y_source: str
    clusters_per_layer: int
    subarrays_per_layer: int
    global_peripheral_instances_per_layer: int
    local_mux_instances_per_layer: int
    bits_per_subarray: int
    bits_per_layer: int
    cluster_spacing_width_overhead_um: float
    cluster_spacing_height_overhead_um: float
    placed_width_um: float
    placed_height_um: float
    layout_utilization: float
    effective_density_Mb_per_mm2: float
    row_address_bits: int
    column_address_bits: int
    accessed_clusters_per_access: int
    accessed_subarrays_per_access: int
    selected_bits_per_subarray: int
    delivered_bits_per_access: int
    data_width_before_vertical: int
    global_rwl_route_length_um_per_cluster: float
    global_wwl_route_length_um_per_cluster: float
    global_wbl_route_length_um_per_cluster: float
    local_rbl_route_length_um: float
    global_rwl_raw_energy_pJ_per_active_cluster: float
    global_wwl_raw_energy_pJ_per_active_cluster: float
    global_wbl_raw_energy_pJ_per_active_cluster: float
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
    global_control_scope: str
    topology_provenance: str
    cluster_provenance: str
    access_provenance: str
    global_peripheral_provenance: str
    interconnect_electrical: dict[str, dict[str, object]]
    local_mux_provenance: str
    local_mux_geometry_modeled: bool
    local_mux_energy_modeled: bool
    local_mux_energy_status: str
    local_rbl_separate_energy_modeled: bool
    dreamram_hierarchy_included: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _address_bits(count: int) -> int:
    return 0 if count == 1 else math.ceil(math.log2(count))


def _line_energy_pJ(
        electrical: RoutingElectricalInput, *, length_um: float,
        instance_multiplier: int = 1) -> float:
    # fF * V^2 is fJ; 1e-3 converts fJ to pJ. Metal geometry and R/um remain
    # explicit MODELING_CHOICE diagnostics for later RC validation.
    return (
        instance_multiplier
        * electrical.active_line_count
        * electrical.activity_factor
        * electrical.capacitance_fF_per_um
        * length_um
        * electrical.voltage_V ** 2
        * 1e-3
    )


def _add_axis_bands(
        *, width: float, height: float, row_band: float, col_band: float,
        row_axis: str, col_axis: str) -> tuple[float, float]:
    """Apply each shared peripheral band once using configured axes."""
    resolved_width = width
    resolved_height = height
    if row_axis == "x":
        resolved_width += row_band
    else:
        resolved_height += row_band
    if col_axis == "x":
        resolved_width += col_band
    else:
        resolved_height += col_band
    return resolved_width, resolved_height


def calculate_m3d_subarray(
        spec: M3DSubarrayInput, geometry: M3DGeometry,
        ) -> M3DSubarrayResult:
    """Resolve complete SubarrayClusters and per-access routing energy."""
    F_um = math.sqrt(geometry.cell_area_um2) / 2.0
    pitch_um = 2.0 * F_um
    core = spec.subarray
    core_width = core.n_cols * pitch_um
    core_height = core.n_rows * pitch_um
    mux_footprint = spec.local_mux.footprint_height_f * F_um
    subarray_width = core_width
    subarray_height = core_height + mux_footprint

    cluster = spec.subarray_cluster
    spacing = spec.spacing
    subarray_gap_x = spacing.subarray_gap_x_f * F_um
    subarray_gap_y = spacing.subarray_gap_y_f * F_um
    cluster_gap_x = spacing.cluster_gap_x_f * F_um
    cluster_gap_y = spacing.cluster_gap_y_f * F_um
    subarrays_per_cluster = cluster.subarrays_x * cluster.subarrays_y
    cluster_array_width_without_spacing = cluster.subarrays_x * subarray_width
    cluster_array_height_without_spacing = cluster.subarrays_y * subarray_height
    subarray_spacing_width = (cluster.subarrays_x - 1) * subarray_gap_x
    subarray_spacing_height = (cluster.subarrays_y - 1) * subarray_gap_y
    cluster_array_width = (
        cluster_array_width_without_spacing + subarray_spacing_width)
    cluster_array_height = (
        cluster_array_height_without_spacing + subarray_spacing_height)
    global_spec = spec.global_peripheral
    row_band = global_spec.row_selection_band_f * F_um
    col_band = global_spec.column_write_selection_band_f * F_um
    cluster_width, cluster_height = _add_axis_bands(
        width=cluster_array_width,
        height=cluster_array_height,
        row_band=row_band,
        col_band=col_band,
        row_axis=global_spec.row_selection_band_axis,
        col_axis=global_spec.column_write_selection_band_axis,
    )

    max_cluster_x = math.floor(
        (geometry.slab_x_um + cluster_gap_x)
        / (cluster_width + cluster_gap_x))
    max_cluster_y = math.floor(
        (geometry.slab_y_um + cluster_gap_y)
        / (cluster_height + cluster_gap_y))
    if max_cluster_x < 1 or max_cluster_y < 1:
        raise ValueError("one complete SubarrayCluster does not fit the M3D slab")
    cluster_count_x = (
        max_cluster_x if cluster.grid.nx == "auto" else cluster.grid.nx)
    cluster_count_y = (
        max_cluster_y if cluster.grid.ny == "auto" else cluster.grid.ny)
    if cluster_count_x > max_cluster_x or cluster_count_y > max_cluster_y:
        raise ValueError(
            "explicit SubarrayCluster grid "
            f"{cluster_count_x}x{cluster_count_y} exceeds fit limit "
            f"{max_cluster_x}x{max_cluster_y}")
    clusters_per_layer = cluster_count_x * cluster_count_y
    subarrays_per_layer = clusters_per_layer * subarrays_per_cluster

    access = spec.access
    if access.accessed_clusters_per_access > clusters_per_layer:
        raise ValueError("accessed clusters exceed instantiated clusters")
    cluster_access_capacity = (
        access.accessed_clusters_per_access * subarrays_per_cluster)
    if access.accessed_subarrays_per_access > cluster_access_capacity:
        raise ValueError(
            "accessed subarrays exceed capacity of accessed clusters")
    if (access.accessed_subarrays_per_access
            < access.accessed_clusters_per_access):
        raise ValueError(
            "each accessed cluster must contain an accessed subarray")
    if access.selected_bits_per_subarray > core.n_cols:
        raise ValueError("selected bits per subarray exceed Ncol")
    delivered_bits = (
        access.accessed_subarrays_per_access
        * access.selected_bits_per_subarray)

    # Tang Global lines span one cluster's subarray array, excluding the thin
    # peripheral-band footprint. Only participating clusters switch.
    global_rwl_length = cluster_array_width
    global_wwl_length = cluster_array_width
    global_wbl_length = cluster_array_height
    rwl_per_cluster = _line_energy_pJ(
        spec.interconnect.global_rwl, length_um=global_rwl_length)
    wwl_per_cluster = _line_energy_pJ(
        spec.interconnect.global_wwl, length_um=global_wwl_length)
    wbl_per_cluster = _line_energy_pJ(
        spec.interconnect.global_wbl, length_um=global_wbl_length)
    active_clusters = access.accessed_clusters_per_access
    rwl_raw = active_clusters * rwl_per_cluster
    wwl_raw = active_clusters * wwl_per_cluster
    wbl_raw = active_clusters * wbl_per_cluster

    local_rbl_length = core_height
    # Zhu's complete local-read primitive is the sole v1 local dynamic-energy
    # boundary. Retain physical Local RBL/MUX geometry, but do not add another
    # standalone local energy term.
    local_rbl_raw = 0.0
    mux_raw = 0.0
    rwl = rwl_raw / delivered_bits
    wwl = wwl_raw / delivered_bits
    wbl = wbl_raw / delivered_bits
    local_rbl = 0.0
    mux = 0.0
    global_energy = rwl + wwl + wbl
    local_energy = local_rbl + mux
    cluster_spacing_width = (cluster_count_x - 1) * cluster_gap_x
    cluster_spacing_height = (cluster_count_y - 1) * cluster_gap_y
    placed_width = cluster_count_x * cluster_width + cluster_spacing_width
    placed_height = cluster_count_y * cluster_height + cluster_spacing_height
    bits_per_subarray = core.n_rows * core.n_cols
    slab_area_mm2 = geometry.slab_x_um * geometry.slab_y_um * 1e-6

    return M3DSubarrayResult(
        slab_x_um=geometry.slab_x_um,
        slab_y_um=geometry.slab_y_um,
        cell_area_um2=geometry.cell_area_um2,
        cell_area_source="resolved_geometry.cell_area_um2",
        F_um=F_um,
        cell_pitch_x_um=pitch_um,
        cell_pitch_y_um=pitch_um,
        Nrow=core.n_rows,
        Ncol=core.n_cols,
        core_width_um=core_width,
        core_height_um=core_height,
        subarray_width_um=subarray_width,
        subarray_height_um=subarray_height,
        local_mux_footprint_height_um=mux_footprint,
        shared_row_selection_band_um=row_band,
        shared_column_write_selection_band_um=col_band,
        row_selection_band_axis=global_spec.row_selection_band_axis,
        column_write_selection_band_axis=(
            global_spec.column_write_selection_band_axis),
        subarray_gap_x_f=spacing.subarray_gap_x_f,
        subarray_gap_y_f=spacing.subarray_gap_y_f,
        subarray_gap_x_um=subarray_gap_x,
        subarray_gap_y_um=subarray_gap_y,
        cluster_gap_x_f=spacing.cluster_gap_x_f,
        cluster_gap_y_f=spacing.cluster_gap_y_f,
        cluster_gap_x_um=cluster_gap_x,
        cluster_gap_y_um=cluster_gap_y,
        spacing_provenance=spacing.provenance,
        cluster_subarrays_x=cluster.subarrays_x,
        cluster_subarrays_y=cluster.subarrays_y,
        subarrays_per_cluster=subarrays_per_cluster,
        cluster_array_width_without_spacing_um=(
            cluster_array_width_without_spacing),
        cluster_array_height_without_spacing_um=(
            cluster_array_height_without_spacing),
        subarray_spacing_width_overhead_um=subarray_spacing_width,
        subarray_spacing_height_overhead_um=subarray_spacing_height,
        cluster_array_width_um=cluster_array_width,
        cluster_array_height_um=cluster_array_height,
        cluster_width_um=cluster_width,
        cluster_height_um=cluster_height,
        cluster_count_x=cluster_count_x,
        cluster_count_y=cluster_count_y,
        cluster_grid_x_source=(
            "auto_floor" if cluster.grid.nx == "auto" else "explicit_override"),
        cluster_grid_y_source=(
            "auto_floor" if cluster.grid.ny == "auto" else "explicit_override"),
        clusters_per_layer=clusters_per_layer,
        subarrays_per_layer=subarrays_per_layer,
        global_peripheral_instances_per_layer=clusters_per_layer,
        local_mux_instances_per_layer=subarrays_per_layer,
        bits_per_subarray=bits_per_subarray,
        bits_per_layer=subarrays_per_layer * bits_per_subarray,
        cluster_spacing_width_overhead_um=cluster_spacing_width,
        cluster_spacing_height_overhead_um=cluster_spacing_height,
        placed_width_um=placed_width,
        placed_height_um=placed_height,
        layout_utilization=(
            clusters_per_layer * cluster_width * cluster_height
            / (geometry.slab_x_um * geometry.slab_y_um)),
        effective_density_Mb_per_mm2=(
            subarrays_per_layer * bits_per_subarray / 1e6 / slab_area_mm2),
        row_address_bits=(
            _address_bits(cluster_count_y)
            + _address_bits(cluster.subarrays_y)
            + _address_bits(core.n_rows)),
        column_address_bits=(
            _address_bits(cluster_count_x)
            + _address_bits(cluster.subarrays_x)
            + _address_bits(core.n_cols)),
        accessed_clusters_per_access=access.accessed_clusters_per_access,
        accessed_subarrays_per_access=access.accessed_subarrays_per_access,
        selected_bits_per_subarray=access.selected_bits_per_subarray,
        delivered_bits_per_access=delivered_bits,
        data_width_before_vertical=delivered_bits,
        global_rwl_route_length_um_per_cluster=global_rwl_length,
        global_wwl_route_length_um_per_cluster=global_wwl_length,
        global_wbl_route_length_um_per_cluster=global_wbl_length,
        local_rbl_route_length_um=local_rbl_length,
        global_rwl_raw_energy_pJ_per_active_cluster=rwl_per_cluster,
        global_wwl_raw_energy_pJ_per_active_cluster=wwl_per_cluster,
        global_wbl_raw_energy_pJ_per_active_cluster=wbl_per_cluster,
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
        physical_topology="tang_subarray_cluster",
        global_control_scope="SUBARRAY_CLUSTER",
        topology_provenance=spec.topology_provenance,
        cluster_provenance=cluster.provenance,
        access_provenance=access.provenance,
        global_peripheral_provenance=global_spec.provenance,
        interconnect_electrical={
            "global_rwl": spec.interconnect.global_rwl.model_dump(),
            "global_wwl": spec.interconnect.global_wwl.model_dump(),
            "global_wbl": spec.interconnect.global_wbl.model_dump(),
        },
        local_mux_provenance=spec.local_mux.provenance,
        local_mux_geometry_modeled=True,
        local_mux_energy_modeled=False,
        local_mux_energy_status="NOT_SEPARATELY_MODELED",
        local_rbl_separate_energy_modeled=False,
        dreamram_hierarchy_included=False,
    )
