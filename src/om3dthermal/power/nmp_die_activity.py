"""Performance-only physical group/fabric/tile execution; no power model."""
from __future__ import annotations

import numpy as np


def group_service_seconds(layer_bytes, service_ns):
    """Eight layers serialize within a group; inactive groups have zero time."""
    if np.any(layer_bytes < 0):
        raise ValueError("negative memory service bytes")
    return (((layer_bytes+31)//32)*service_ns).sum(axis=-1)*1e-9


def ingress_regions(f, active):
    """Minimax physical multicast latency, then hop bytes, then region id."""
    if not hasattr(f, "ingress_lookup"):
        positions = np.r_[0, np.cumsum([l["hop_ns"] for l in f.links])]
        f.ingress_lookup = np.zeros(16, dtype=int)
        for mask in range(1, 16):
            destinations = [r for r in range(4) if mask & (1 << r)]
            f.ingress_lookup[mask] = min(destinations, key=lambda r: (
                max(abs(positions[r]-positions[d]) for d in destinations),
                max(destinations)-min(destinations), r))
    masks = np.asarray(active, dtype=int) @ (1 << np.arange(4))
    return f.ingress_lookup[masks]


def structured_noc(f, region_bytes, *, mode, vector_bytes=0):
    """Shortest-path multicast or balanced pairwise intra-slab reduction."""
    region_bytes = np.asarray(region_bytes, dtype=float)
    active = region_bytes > 0
    n = len(region_bytes)
    roots = ingress_regions(f, active)
    links = np.zeros((n, 3, 2))
    times = np.zeros(n)
    masks = active.astype(int) @ (1 << np.arange(4))
    positions = np.r_[0, np.cumsum([l["hop_ns"] for l in f.links])]
    for mask in np.unique(masks):
        if not mask:
            continue
        sel = masks == mask
        dest = [r for r in range(4) if mask & (1 << r)]
        root = int(roots[sel][0])
        payload = region_bytes[sel].max(axis=1)
        if mode == "multicast":
            assert np.all((region_bytes[sel] == 0) | (region_bytes[sel] == payload[:,None]))
            for edge in range(min(dest), max(dest)):
                links[sel, edge, int(edge < root)] = payload
            if len(dest) > 1:
                times[sel] = payload/f.link_Bps+max(abs(positions[root]-positions[d]) for d in dest)*1e-9
        elif mode == "reduce":
            assert np.all(region_bytes[sel][region_bytes[sel] > 0] == vector_bytes)
            partials = [(r, [r]) for r in dest]
            while len(partials) > 1:
                merged, round_s = [], 0.0
                for i in range(0, len(partials), 2):
                    if i+1 == len(partials):
                        merged.append(partials[i]); continue
                    a, am = partials[i]; b, bm = partials[i+1]
                    target = a if root in am else b if root in bm else min((a,b), key=lambda r:(abs(positions[r]-positions[root]),r))
                    source = b if target == a else a
                    for edge in range(min(source,target), max(source,target)):
                        links[sel,edge,int(source > target)] += vector_bytes
                    round_s = max(round_s, vector_bytes/f.link_Bps+abs(positions[source]-positions[target])*1e-9
                                  +vector_bytes/4/(f.config["reduction_adds_per_cycle"]*f.config["clock_hz"]))
                    merged.append((target,am+bm))
                times[sel] += round_s
                partials = merged
        else:
            raise ValueError("only structured multicast/reduce is supported")
    if mode not in ("multicast", "reduce"):
        raise ValueError("only structured multicast/reduce is supported")
    return dict(time_s=float(times.max(initial=0)), link_bytes=links, hop_bytes=float(links.sum()),
                endpoint_bytes=float(region_bytes.sum()), root_regions=roots,
                max_link_busy_s=float(links.max(initial=0)/f.link_Bps))


def external_service(f, payload_bytes, *, mode, details=False):
    """Explicit group routing, region striping, or one-copy broadcast ingress."""
    b = np.asarray(payload_bytes, dtype=float)
    if np.any(b < 0):
        raise ValueError("negative boundary payload")
    n, ports = f.layout.slab_count, len(f.ports)
    loads = np.zeros((n,ports)); startup = np.zeros_like(loads)
    ingress = None
    if mode == "GROUP_DIRECT":
        assert b.shape == (n,70)
        if not hasattr(f, "group_ports"):
            f.group_ports = np.array([min(range(ports), key=lambda p: abs(g["center_um"][0]-f.ports[p][0])+abs(g["center_um"][1]-f.ports[p][1])) for g in f.groups])
            f.group_port_ids = (np.arange(n)[:,None]*ports+f.group_ports).ravel()
        loads = np.bincount(f.group_port_ids, weights=b.ravel(), minlength=n*ports).reshape(n,ports)
        np.maximum.at(startup.ravel(), f.group_port_ids, ((b>0)*f.sa_edge_ns[None,:]*1e-9).ravel())
    elif mode in ("REGION_DIRECT", "REGION_BROADCAST_INGRESS"):
        assert b.shape == (n,4)
        if mode == "REGION_BROADCAST_INGRESS":
            ingress = ingress_regions(f, b>0)
            payload = b.max(axis=1)
            assert np.all((b == 0) | (b == payload[:,None]))
            b = np.zeros_like(b); b[np.arange(n), ingress] = payload
        for r, ids in enumerate(f.region_port_ids):
            # Equal analytical load; exact remainder goes to the final port.
            loads[:,ids] = b[:,r:r+1]/len(ids)
            loads[:,ids[-1]] = b[:,r]-loads[:,ids[:-1]].sum(axis=1)
            startup[:,ids] = (loads[:,ids]>0)*f.root_port_route_ns[r,ids]*1e-9
    else:
        raise ValueError("unknown external routing semantics")
    active = loads > 0
    total = float(b.sum())
    per_port = f.case.architecture.memory_service.coil.data_rate_gbps_per_link*1e9/8
    serial = loads/per_port
    global_s = total/f.external_Bps
    slowest = int(np.argmax(serial+startup))
    port_completion = float((serial+startup).ravel()[slowest])
    duration = max(global_s,port_completion)
    reason = ("GLOBAL_THERMAL_CAP" if global_s >= port_completion else
              "PORT_SERIALIZATION" if serial.ravel()[slowest] >= startup.ravel()[slowest] else "ROUTE_STARTUP")
    result = dict(mode=mode, total_boundary_bytes=total, active_port_count=int(active.sum()),
                  max_port_bytes=float(loads.max()), mean_active_port_bytes=float(loads[active].mean()) if total else 0.0,
                  mean_active_port_utilization=float(serial[active].mean()/duration) if total else 0.0,
                  max_port_utilization=float(serial.max()/duration) if total else 0.0,
                  global_cap_serialization_s=global_s, port_cap_serialization_s=float(serial.max()),
                  rc_startup_s=float(startup.max()), external_service_s=duration, limiting_reason=reason)
    if details:
        result["ports"] = [dict(slab=int(d), port=int(p), bytes=float(loads[d,p]), route_startup_s=float(startup[d,p]))
                           for d,p in zip(*np.nonzero(active))]
        if ingress is not None:
            result["ingress_regions"] = ingress.tolist()
    return result


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
        external_stages = []
        if nmp and not write:
            if op == "ATTENTION_AV":
                # Probability partitions follow the resident KV atoms.
                input_region = np.bincount(rid, weights=counts*w.n_heads_q/w.n_heads_kv*2, minlength=n*4).reshape(n, 4)
                input_mode = "REGION_DIRECT"
                noc_in_s = 0.0
                fabric_bytes += input_region
                # Tile partials reduce at each region root before the pairwise tree.
                fabric_bytes += tile_counts*w.d_model*4
                reduction_s = float(np.maximum(tile_counts-1, 0).max()*w.d_model/(8*f.config["clock_hz"]))
                output_region = region_active*w.d_model*4
                outgoing = structured_noc(f, output_region, mode="reduce", vector_bytes=w.d_model*4)
                output_region = np.zeros((n,4))
                output_region[np.arange(n),outgoing["root_regions"]] = active_dies*w.d_model*4
            else:
                input_size = (w.d_model*2 if op == "ATTENTION_QK" else entry.unit.activation_input_bytes)
                input_region = region_active*input_size
                input_mode = "REGION_BROADCAST_INGRESS"
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
                outgoing = dict(time_s=0.0, hop_bytes=0.0, link_bytes=np.zeros_like(noc_link_bytes))
            noc_s = noc_in_s+outgoing["time_s"]
            noc_bytes += outgoing["hop_bytes"]; noc_link_bytes += outgoing["link_bytes"]
            noc_busy = float(noc_link_bytes.max()/f.link_Bps)
            incoming_external = external_service(f, input_region, mode=input_mode, details=details)
            outgoing_external = external_service(f, output_region, mode="REGION_DIRECT", details=details)
            external_stages = [incoming_external, outgoing_external]
            input_boundary = incoming_external["total_boundary_bytes"]
            output_boundary = outgoing_external["total_boundary_bytes"]
            boundary_s = sum(x["external_service_s"] for x in external_stages)
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
            transfer = external_service(f, region_bytes if write else group_total,
                                        mode="REGION_DIRECT" if write else "GROUP_DIRECT", details=details)
            external_stages = [transfer]
            boundary_s, output_boundary = transfer["external_service_s"], transfer["total_boundary_bytes"]
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
            external_transfers=external_stages,
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
