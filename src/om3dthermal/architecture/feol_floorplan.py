"""Physical FEOL resources over the canonical capacity/latency geometry."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math

import numpy as np
import yaml

from om3dthermal.power import load_case_config, resolve_case_geometry, calculate_memory_power
from om3dthermal.power.m3d_subarray import calculate_m3d_subarray
from om3dthermal.power.feol_route import calculate_feol_route, distributed_elmore_delay_ns


def manhattan(a, b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])


def stats(values):
    return {k: float(v) for k, v in zip(("mean", "median", "p90", "max"),
             (np.mean(values), np.median(values), np.percentile(values, 90), np.max(values)), strict=True)}


@dataclass
class FEOLFloorplan:
    case: object
    geometry: object
    topology: object
    layout: object
    physical_latency: object
    config: dict
    clusters: list
    groups: list
    regions: list
    tiles: list
    ports: list
    links: list
    service_ns: np.ndarray
    sa_tile_ns: np.ndarray
    root_tile_ns: np.ndarray
    sa_edge_ns: np.ndarray
    root_edge_ns: np.ndarray

    def __post_init__(self):
        self.region_port_ids = [[] for _ in self.regions]
        for p, (x, _) in enumerate(self.ports):
            candidates = [r for r in self.regions if r["bounds_um"][0] <= x <= r["bounds_um"][2]]
            owner = min(candidates, key=lambda r: (abs(x-r["center_um"][0]), r["region_id"]))
            self.region_port_ids[owner["region_id"]].append(p)
        assert sum(map(len, self.region_port_ids)) == len(self.ports)
        assert all(self.region_port_ids)
        self.root_port_route_um = np.array([[manhattan(r["center_um"], p) for p in self.ports] for r in self.regions])
        self.root_port_route_ns = np.array([[self.wire_ns(x) for x in row] for row in self.root_port_route_um])

    @property
    def fabric_Bps(self):
        return self.config["fabric_ports_per_region"]*self.config["fabric_bits_per_port"]*self.config["clock_hz"]/8

    @property
    def link_Bps(self):
        return self.config["noc_bits_per_link"]*self.config["clock_hz"]/8

    @property
    def tile_flops(self):
        return self.config["macs_per_tile"]*self.config["clock_hz"]*self.config["flops_per_mac"]

    @property
    def external_Bps(self):
        return self.external_rate(float("inf"))

    def external_rate(self, demand):
        if demand < 0:
            raise ValueError("negative transfer demand")
        e = self.config["m3d_external_bandwidth"]
        return min(demand, e["raw_interface_bytes_per_s"], e["gpu_peak_bytes_per_s"], e["thermal_cap_bytes_per_s"])

    def wire_ns(self, length_um):
        return distributed_elmore_delay_ns(length_um, self.case.architecture.feol_route.wire)

    def audit(self):
        local_lengths = [min(manhattan(g["center_um"], t["center_um"]) for t in self.tiles
                             if t["region_id"] == g["region_id"]) for g in self.groups]
        external_lengths = [min(manhattan(g["center_um"], p) for p in self.ports) for g in self.groups]
        return dict(slab_dimensions_um=[self.topology.slab_x_um, self.topology.slab_y_um],
                    slabs=self.layout.slab_count, thickness_um=self.case.geometry.orthogonal.slab_pitch_x_um,
                    layers=self.layout.layers_per_cluster, capacity_GB=self.layout.total_capacity_bytes/1e9,
                    cluster_grid=[self.topology.cluster_count_x, self.topology.cluster_count_y],
                    clusters=self.clusters, service_groups=self.groups,
                    sa_banks=[dict(group_id=g["group_id"], bits=self.config["sa_bits"], center_um=g["center_um"]) for g in self.groups],
                    regions=self.regions, region_group_counts=[len(r["groups"]) for r in self.regions],
                    mac_tiles=self.tiles, macs_per_slab=len(self.tiles)*self.config["macs_per_tile"],
                    region_roots=[r["center_um"] for r in self.regions], io_ports_um=self.ports,
                    region_port_counts=list(map(len, self.region_port_ids)), region_port_ids=self.region_port_ids,
                    root_port_route_um=self.root_port_route_um.tolist(), root_port_route_ns=self.root_port_route_ns.tolist(),
                    root_local_port_length_um=stats([self.root_port_route_um[r,p] for r,ids in enumerate(self.region_port_ids) for p in ids]),
                    root_local_port_rc_ns=stats([self.root_port_route_ns[r,p] for r,ids in enumerate(self.region_port_ids) for p in ids]),
                    noc_links=self.links, sa_nearest_mac_length_um=stats(local_lengths),
                    sa_nearest_edge_length_um=stats(external_lengths),
                    root_edge_lengths_um=[min(manhattan(r["center_um"], p) for p in self.ports) for r in self.regions],
                    sa_mac_rc_ns=stats([self.wire_ns(x) for x in local_lengths]),
                    sa_edge_rc_ns=stats(self.sa_edge_ns), service_ns_by_layer=self.service_ns.tolist(),
                    array_ceiling_diagnostic_Bps=sum(32/(x*1e-9) for x in np.mean(self.service_ns, axis=1)),
                    array_ceiling_status="THEORETICAL_ALL_GROUPS_ACTIVE_ARRAY_SERVICE_CEILING__NOT_STAGE_BANDWIDTH",
                    fabric_GBps_per_region=self.fabric_Bps/1e9, noc_GBps_per_direction=self.link_Bps/1e9,
                    external_saturated_TBps=self.external_Bps/1e12, config=self.config,
                    SA_status="DISTRIBUTED_FEOL_AT_GROUP_MIV_LANDING__SENSING_INCLUDED_IN_10NS_MAT_PRIMITIVE",
                    NoC_scope="INTRA_SLAB_ONLY__MULTICAST_AND_TREE_GATHER_REDUCTION",
                    route_model="PIPELINED_DISTRIBUTED_ELMORE_STARTUP__NO_EFFICIENCY_FACTOR")


@lru_cache(maxsize=1)
def resolve_feol_floorplan(project_root: str | Path):
    root = Path(project_root)
    case = load_case_config(root/"configs/cases/orthogonal_m3d_igzo.yaml")
    geometry = resolve_case_geometry(case)
    memory = calculate_memory_power(case, project_root=root, geometry=geometry,
                                   read_bandwidth_gbps=case.workload.read_bandwidth_gbps)
    topology = calculate_m3d_subarray(case.architecture.m3d_subarray, geometry.m3d)
    route = calculate_feol_route(case.architecture, topology)
    layout = memory.physical_capacity_layout
    from om3dthermal.power import calculate_physical_access_latency
    latency = calculate_physical_access_latency(case.architecture.physical_access_latency, feol_route=route,
        miv_length_per_layer_um=memory.diagnostics["miv_length_per_layer_um"],
        miv_delay_per_layer_ns=memory.diagnostics["miv_delay_per_layer_ns"],
        miv_status=memory.diagnostics["miv_latency_status"],
        miv_parameter_status=memory.diagnostics["miv_resistance_parameter_status"],
        miv_provenance=memory.diagnostics["miv_resistance_provenance"])
    cfg = yaml.safe_load((root/"configs/architecture/m3d_feol_execution.yaml").read_text(encoding="utf-8"))
    energy_cfg = yaml.safe_load((root/"configs/architecture/m3d_feol_energy_v1.yaml").read_text(encoding="utf-8"))
    cfg["region_buffer_bytes"] = energy_cfg["pe_sram_bytes"]*cfg["macs_per_tile"]*math.prod(cfg["mac_tile_grid"])
    assert (topology.cluster_count_x, topology.cluster_count_y, layout.slab_count, layout.layers_per_cluster) == (35, 8, 318, 8)
    assert topology.delivered_bits_per_access == 256 and topology.subarrays_per_cluster == 64
    centers = route.feol_route_cluster_centers_um
    clusters = [dict(cluster_id=i, column=i % 35, row=i // 35, center_um=list(c)) for i, c in enumerate(centers)]
    # Cut at physical midplanes between neighboring cluster columns.
    cuts = [0.0]+[(centers[i-1][0]+centers[i][0])/2 for i in np.cumsum(cfg["regions_columns"])[:-1]]+[topology.slab_x_um]
    groups, regions, tiles = [], [], []
    first_col = 0
    for r, columns in enumerate(cfg["regions_columns"]):
        x0, x1 = cuts[r:r+2]
        region = dict(region_id=r, bounds_um=[x0, 0.0, x1, topology.slab_y_um],
                      center_um=[(x0+x1)/2, topology.slab_y_um/2], columns=list(range(first_col, first_col+columns)), groups=[])
        for col in region["columns"]:
            for rows in cfg["groups_rows"]:
                members = [row*35+col for row in rows]
                center = np.mean([centers[m] for m in members], axis=0).tolist()
                gid = len(groups)
                groups.append(dict(group_id=gid, region_id=r, column=col, cluster_members=members, center_um=center,
                                   capacity_bytes=4*layout.layers_per_cluster*layout.slot_capacity_bytes))
                region["groups"].append(gid)
        nx, ny = cfg["mac_tile_grid"]
        for iy in range(ny):
            for ix in range(nx):
                tiles.append(dict(tile_id=len(tiles), region_id=r, macs=cfg["macs_per_tile"],
                                  center_um=[x0+(ix+.5)*(x1-x0)/nx, (iy+.5)*topology.slab_y_um/ny]))
        regions.append(region)
        first_col += columns
    ports = list(route.feol_io_channel_coordinates_um)
    assert len(ports) == 50 and all(p[1] == 0 for p in ports)
    wire = lambda length: distributed_elmore_delay_ns(length, case.architecture.feol_route.wire)
    clock_ns = 1e9/cfg["clock_hz"]
    links = []
    for i in range(3):
        length = manhattan(regions[i]["center_um"], regions[i+1]["center_um"])
        rc = wire(length)
        cycles = math.ceil(rc/clock_ns)
        links.append(dict(link_id=i, regions=[i, i+1], length_um=length, rc_ns=rc,
                          wire_pipeline_cycles=cycles, hop_ns=(cfg["router_cycles_per_hop"]+cycles)*clock_ns))
    by_slot = {(x.cluster_id, x.layer_id): x.mat_latency_ns+x.miv_latency_ns for x in latency.locations}
    layers = sorted({x.layer_id for x in latency.locations})
    service = np.array([[max(by_slot[c, l] for c in g["cluster_members"]) for l in layers] for g in groups])
    sa_tile = np.array([[wire(manhattan(g["center_um"], t["center_um"])) for t in tiles] for g in groups])
    root_tile = np.array([wire(manhattan(t["center_um"], regions[t["region_id"]]["center_um"])) for t in tiles])
    result = FEOLFloorplan(case, geometry, topology, layout, latency, cfg, clusters, groups, regions, tiles, ports, links,
        service, sa_tile, root_tile, np.array([wire(min(manhattan(g["center_um"], p) for p in ports)) for g in groups]),
        np.array([wire(min(manhattan(r["center_um"], p) for p in ports)) for r in regions]))

    # The existing MIV primitive is linear in effective per-layer capacitance.
    d = memory.diagnostics
    result.energy_config = energy_cfg
    result.miv_pj_per_bit_by_layer = (np.array(d["miv_effective_capacitance_per_layer_pF"])
        *d["miv_access_energy_pJ_per_bit"]/d["miv_average_effective_capacitance_pF"])
    result.sa_tile_um = np.array([[manhattan(g["center_um"],t["center_um"]) for t in tiles] for g in groups])
    result.root_tile_um = np.array([manhattan(t["center_um"],regions[t["region_id"]]["center_um"]) for t in tiles])
    result.sa_edge_um = np.array([min(manhattan(g["center_um"],p) for p in ports) for g in groups])
    result.root_sa_um = np.array([manhattan(g["center_um"],regions[g["region_id"]]["center_um"]) for g in groups])
    return result
