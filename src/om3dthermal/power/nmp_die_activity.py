"""Performance-only physical group/fabric/tile execution; no power model."""
from __future__ import annotations

import numpy as np


def group_service_seconds(layer_bytes, service_ns):
    """Eight layers serialize within a group; inactive groups have zero time."""
    if np.any(layer_bytes < 0):
        raise ValueError("negative memory service bytes")
    return (((layer_bytes+31)//32)*service_ns).sum(axis=-1)*1e-9


def structured_noc(floorplan, region_bytes, *, mode, vector_bytes=0):
    """Root-0 multicast or pairwise (1->0, 3->2), then 2->0 gather/reduce.

    Link counters include every traversed directed physical edge; payload
    endpoints and hop bytes are distinct. No edge crosses a slab boundary.
    """
    f = floorplan
    region_bytes = np.asarray(region_bytes, dtype=float)
    active = region_bytes > 0
    n = len(region_bytes)
    links = np.zeros((n, 3, 2))
    if mode == "multicast":
        # All destinations of a multicast receive the same payload.
        payload = region_bytes.max(axis=1)
        for edge in range(3):
            links[:, edge, 0] = payload*np.any(active[:, edge+1:], axis=1)
        path_ns = np.zeros(n)
        for r in range(1, 4):
            path_ns = np.maximum(path_ns, active[:, r]*sum(l["hop_ns"] for l in f.links[:r]))
        time = float(np.max(links)/f.link_Bps+path_ns.max()*1e-9)
        endpoint = float(region_bytes.sum())
    elif mode in ("gather", "reduce"):
        reduce = mode == "reduce"
        first1 = region_bytes[:, 1]
        first3 = region_bytes[:, 3]
        merged23 = (np.any(active[:, 2:], axis=1)*vector_bytes if reduce else region_bytes[:, 2:].sum(axis=1))
        links[:, 0, 1] += first1
        links[:, 2, 1] += first3
        links[:, 1, 1] += merged23
        links[:, 0, 1] += merged23
        t1 = max(float(np.max(first1)/f.link_Bps + (np.any(first1 > 0)*f.links[0]["hop_ns"]*1e-9)),
                 float(np.max(first3)/f.link_Bps + (np.any(first3 > 0)*f.links[2]["hop_ns"]*1e-9)))
        t2 = float(np.max(merged23)/f.link_Bps + np.any(merged23 > 0)*sum(l["hop_ns"] for l in f.links[:2])*1e-9)
        if reduce:
            add_s = (vector_bytes/4)/(f.config["reduction_adds_per_cycle"]*f.config["clock_hz"])
            t1 += add_s*bool(np.any((active[:, 0]&active[:, 1]) | (active[:, 2]&active[:, 3])))
            t2 += add_s*bool(np.any(np.any(active[:, :2], axis=1)&np.any(active[:, 2:], axis=1)))
        time = t1+t2
        endpoint = float(region_bytes.sum())
    else:
        raise ValueError("only structured multicast/gather/reduce is supported")
    return dict(time_s=time, link_bytes=links, hop_bytes=float(links.sum()), endpoint_bytes=endpoint,
                max_link_busy_s=float(links.max()/f.link_Bps))


def external_service(f, *, group_bytes=None, root_bytes=None):
    """Nearest physical port serialization plus global thermal bandwidth cap."""
    n = f.layout.slab_count
    if not hasattr(f, "group_ports"):
        f.group_ports = np.array([min(range(len(f.ports)), key=lambda p: abs(g["center_um"][0]-f.ports[p][0])+abs(g["center_um"][1]-f.ports[p][1])) for g in f.groups])
        f.root_ports = np.array([min(range(len(f.ports)), key=lambda p: abs(r["center_um"][0]-f.ports[p][0])+abs(r["center_um"][1]-f.ports[p][1])) for r in f.regions])
        f.group_port_ids = (np.arange(n)[:, None]*len(f.ports)+f.group_ports).ravel()
        f.root_port_ids = (np.arange(n)[:, None]*len(f.ports)+f.root_ports).ravel()
    port_loads = np.zeros(n*len(f.ports))
    startup = 0.0
    if group_bytes is not None:
        group_bytes = np.asarray(group_bytes)
        port_loads += np.bincount(f.group_port_ids, weights=group_bytes.ravel(), minlength=n*len(f.ports))
        startup = float(f.sa_edge_ns[np.any(group_bytes > 0, axis=0)].max(initial=0))*1e-9
    if root_bytes is not None:
        port_loads += np.bincount(f.root_port_ids, weights=root_bytes.ravel(), minlength=n*len(f.ports))
        startup = max(startup, float(f.root_edge_ns[np.any(root_bytes > 0, axis=0)].max(initial=0))*1e-9)
    total = float(port_loads.sum())
    per_port = f.case.architecture.memory_service.coil.data_rate_gbps_per_link*1e9/8
    # 50 x 8 Gb/s physical contactless links; thermal cap remains global.
    duration = max(total/f.external_Bps, float(port_loads.max())/per_port)+startup if total else 0.0
    return duration, total


class PhysicalStageModel:
    def __init__(self, floorplan, platform, workload):
        self.f = floorplan
        self.w = workload
        self.gpu_compute = platform.gpu_compute_power.peak_compute_BF16_dense_flops_per_s
        self.gpu_bw = platform.gpu_decode_power.peak_memory_bandwidth_bytes_per_s
        self.service_prefix = np.zeros((70, 8, 9))
        for start in range(8):
            self.service_prefix[:, start, 1:] = np.cumsum(floorplan.service_ns[:, (np.arange(8)+start) % 8], axis=1)
        self.service_sum = floorplan.service_ns.sum(axis=1)

    def evaluate(self, entry, atoms, *, nmp, begin=0, write=False, details=False):
        f, w = self.f, self.w
        n = f.layout.slab_count
        end_count = entry.prefix_counts(atoms)
        begin_count = entry.prefix_counts(begin) if begin else np.zeros_like(end_count)
        occupied = end_count > begin_count
        group, die, rid = entry.group_ids[occupied], entry.die_ids[occupied], entry.region_dest[occupied]
        tiles_assigned = entry.tile_ids[occupied]
        end_count, begin_count = end_count[occupied], begin_count[occupied]
        starts = entry.start_layers[occupied]
        counts = end_count-begin_count
        bytes_ = counts*entry.atom_bytes
        active = bytes_ > 0
        # Exact cyclic-layer service sum, avoiding materialized per-step slot
        # arrays. Each stored atom is a whole number of 32-byte services.
        assert entry.atom_bytes % 32 == 0
        cycles = ((end_count//8-begin_count//8)*self.service_sum[group]
                  + self.service_prefix[group, starts, end_count % 8]
                  - self.service_prefix[group, starts, begin_count % 8])
        group_times = cycles*(entry.atom_bytes//32)*1e-9
        array_s = float(group_times.max(initial=0))
        region_bytes = np.bincount(rid, weights=bytes_, minlength=n*4).reshape(n, 4)
        region_active = region_bytes > 0
        active_dies = np.any(region_active, axis=1)
        group_total = np.zeros((n, 70))
        group_total[die, group] = bytes_
        op = entry.unit.operator_type
        attention = op in ("ATTENTION_QK", "ATTENTION_AV")
        flop_atom = 2*w.n_heads_q*w.d_head/w.n_heads_kv if attention else entry.unit.local_flops/entry.atom_count
        if write:
            flop_atom = 0
        flops = counts*flop_atom
        tile_loads = np.bincount(die*32+tiles_assigned, weights=flops, minlength=n*32).reshape(n, 32)
        tile_active = np.zeros((n, 32), dtype=bool)
        tile_active[die[active], tiles_assigned[active]] = True
        tile_counts = tile_active.reshape(n, 4, 8).sum(axis=2)
        fabric_bytes = region_bytes.copy()
        fabric_startup = np.zeros(n*4)
        np.maximum.at(fabric_startup, rid, f.sa_tile_ns[group, tiles_assigned]*1e-9)
        fabric_startup = fabric_startup.reshape(n, 4)
        noc_s = noc_bytes = noc_busy = input_boundary = output_boundary = 0.0
        noc_link_bytes = np.zeros((n, 3, 2))
        boundary_s = 0.0
        reduction_s = 0.0
        if nmp and not write:
            if op == "ATTENTION_AV":
                # Probability partitions follow the resident KV atoms.
                input_region = np.bincount(rid, weights=counts*w.n_heads_q/w.n_heads_kv*2, minlength=n*4).reshape(n, 4)
                input_root = np.zeros((n, 4)); input_root[:, 0] = input_region.sum(axis=1)
                # Singleton multicast destinations carry partitioned probabilities.
                incoming = [structured_noc(f, np.eye(4)[r][None, :]*input_region[:, r:r+1], mode="multicast") for r in range(4)]
                noc_in_s = sum(x["time_s"] for x in incoming)
                noc_bytes += sum(x["hop_bytes"] for x in incoming)
                noc_link_bytes += sum(x["link_bytes"] for x in incoming)
                fabric_bytes += input_region
                # Tile partials reduce at each region root before the pairwise tree.
                fabric_bytes += tile_counts*w.d_model*4
                reduction_s = float(np.maximum(tile_counts-1, 0).max()*w.d_model/(8*f.config["clock_hz"]))
                output_region = region_active*w.d_model*4
                outgoing = structured_noc(f, output_region, mode="reduce", vector_bytes=w.d_model*4)
                output_root = np.zeros((n, 4)); output_root[:, 0] = active_dies*w.d_model*4
            else:
                input_size = (w.d_model*2 if op == "ATTENTION_QK" else entry.unit.activation_input_bytes)
                input_region = region_active*input_size
                input_root = np.zeros((n, 4)); input_root[:, 0] = active_dies*input_size
                incoming = structured_noc(f, input_region, mode="multicast")
                noc_in_s = incoming["time_s"]
                noc_bytes += incoming["hop_bytes"]; noc_link_bytes += incoming["link_bytes"]
                fabric_bytes += tile_counts*input_size
                if op == "ATTENTION_QK":
                    outputs = counts*w.n_heads_q/w.n_heads_kv*2
                elif op == "TOKEN_EMBED_LOOKUP":
                    outputs = bytes_
                else:
                    outputs = counts*2
                output_region = np.bincount(rid, weights=outputs, minlength=n*4).reshape(n, 4)
                fabric_bytes += output_region
                outgoing = structured_noc(f, output_region, mode="gather")
                output_root = np.zeros((n, 4)); output_root[:, 0] = output_region.sum(axis=1)
            noc_s = noc_in_s+outgoing["time_s"]
            noc_bytes += outgoing["hop_bytes"]; noc_link_bytes += outgoing["link_bytes"]
            noc_busy = float(noc_link_bytes.max()/f.link_Bps)
            in_s, input_boundary = external_service(f, root_bytes=input_root)
            out_s, output_boundary = external_service(f, root_bytes=output_root)
            boundary_s = in_s+out_s
            if np.any(tile_active):
                fabric_startup += (tile_active*f.root_tile_ns).reshape(n, 4, 8).max(axis=2)*1e-9
            fabric_times = fabric_bytes/f.fabric_Bps+fabric_startup
            fabric_s = float(fabric_times.max())
            compute_s = float(tile_loads.max()/f.tile_flops)
            core_s = max(array_s, fabric_s, compute_s)
            seconds = boundary_s+noc_s+core_s+reduction_s
            components = dict(ARRAY=array_s, LOCAL_FABRIC=fabric_s, MAC=compute_s,
                              INTER_REGION_NOC=noc_s+reduction_s, EXTERNAL_BOUNDARY=boundary_s, GPU_COMPUTE=0.0)
        else:
            boundary_s, output_boundary = external_service(f, group_bytes=group_total)
            activation = entry.unit.activation_input_bytes+entry.unit.partial_output_bytes
            if attention:
                activation = (atoms-begin)/w.n_heads_kv*w.n_heads_q*2+w.d_model*2
            gpu_s = 0.0 if write else max(float(flops.sum())/self.gpu_compute, (float(bytes_.sum())+activation)/self.gpu_bw)
            seconds = max(array_s, boundary_s, gpu_s)
            components = dict(ARRAY=array_s, LOCAL_FABRIC=0.0, MAC=0.0, INTER_REGION_NOC=0.0,
                              EXTERNAL_BOUNDARY=boundary_s, GPU_COMPUTE=gpu_s)
            fabric_s = 0.0
        result = dict(operator="KV_APPEND" if write else op, layer=entry.unit.layer_id, executor="NMP" if nmp and not write else "GPU",
            latency_s=seconds, components=components, bottleneck=max(components, key=components.get),
            boundary_bytes=input_boundary+output_boundary, input_boundary_bytes=input_boundary,
            output_boundary_bytes=output_boundary, external_service_s=boundary_s,
            local_array_bytes=float(bytes_.sum()), array_service_s=array_s,
            active_groups=int(active.sum()), active_regions=int(region_active.sum()),
            active_mac_tiles=int(tile_active.sum()) if nmp and not write else 0,
            fabric_bytes=float(fabric_bytes.sum()) if nmp and not write else 0.0,
            fabric_service_s=fabric_s, fabric_region_peak_Bps=float(fabric_bytes.max()/fabric_s) if fabric_s else 0.0,
            fabric_region_average_Bps=float(fabric_bytes.sum()/region_active.sum()/seconds) if nmp and not write else 0.0,
            noc_bytes=noc_bytes, noc_s=noc_s, noc_max_link_busy_s=noc_busy,
            noc_link_busy_s=noc_link_bytes/f.link_Bps,
            nmp_flops=float(flops.sum()) if nmp and not write else 0.0,
            nmp_peak_utilization=float(flops.sum())/core_s/(n*32*f.tile_flops) if nmp and not write else 0.0,
            active_dies=int(active_dies.sum()), max_buffer_chunks=int(np.ceil(fabric_bytes.max()/f.config["region_buffer_bytes"])) if nmp else 0)
        if details:
            layer_bytes = entry.layer_bytes(atoms, begin=begin)[occupied]
            result["groups"] = [dict(die_id=int(d), group_id=int(g), active_read_bytes=int(b) if not write else 0,
                                      active_write_bytes=int(b) if write else 0, active_layer_slots=int(np.count_nonzero(lb)), service_s=float(t),
                                      assigned_mac_tile=int(tile)) for d, g, b, lb, t, tile in
                                zip(die, group, bytes_, layer_bytes, group_times, tiles_assigned, strict=True) if b]
        return result
