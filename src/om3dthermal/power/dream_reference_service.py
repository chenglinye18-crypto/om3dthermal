"""DreamRAM reference-baseline memory-service audit (read-only).

Audits the pinned ``third_party/DreamRAM`` DATE2026 analytical model with
its own HBM3 baseline organization to obtain, with explicit provenance:

- first-access latency decomposition (tRP + tRCD + tCL, the reference
  model's own ``worst_latency_ns`` definition);
- repeated service cycle (core column cycle and DQ atom window);
- internal (memory-array-side) bandwidth from the reference model's own
  bus hierarchy (array/MDL -> bgbus -> gbus -> TSV stages);
- external interface (DQ) bandwidth from the reference model's own
  ``bandwidth()`` method;
- effective bandwidth as ``min(internal, interface)`` and a bottleneck
  classification from the internal/interface ratio.

This module deliberately imports no M3D machinery and accepts no M3D
inputs.  All Dream/reference quantities are produced by the pinned
reference model itself; no parameter is copied from the M3D
architecture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Literal

from .backends.dreamram import (
    DREAMRAM_BRANCH,
    DREAMRAM_COMMIT,
    _loaded_dreamram,
    _verify_pin,
)
from .config import resolve_project_path


PROVENANCE_LABELS = (
    "PAPER_REPORTED",
    "DERIVED_FROM_PAPER",
    "EXISTING_REPO_REFERENCE",
    "MODELING_CHOICE",
    "UNAVAILABLE",
)

DreamBottleneck = Literal[
    "INTERNAL_MEMORY", "EXTERNAL_INTERFACE", "BALANCED"]

GATE_PASS = "PASS"
GATE_INSUFFICIENT = "INSUFFICIENT_INFORMATION"

#: The reference model sizes its DQ interface from its internal bus
#: hierarchy, so the internal/interface ratio is exactly scale-invariant
#: to the core-cycle uncertainty; only floating-point residue is
#: tolerated when declaring the two stages balanced.
BALANCE_RELATIVE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class DreamReferenceLatency:
    trcd_ns: float
    trp_ns: float
    tcl_ns: float
    first_access_latency_ns: float
    core_tck_ns: float
    bank_clks_per_atom: int
    repeated_service_cycle_ns: float
    dq_atom_window_ns: float
    first_access_definition: str
    repeated_service_cycle_definition: str
    timing_classification: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DreamInternalStage:
    """One shared internal delivery stage of the reference hierarchy."""

    stage: str
    parallel_units_per_pseudochannel: float
    payload_bits_per_core_clock_per_unit: float
    aggregate_bits_per_core_clock_per_pseudochannel: float
    aggregate_bits_per_s: float
    classification: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DreamReferenceServiceAudit:
    branch: str
    commit: str
    memory_config: str
    technology_config: str
    organization: dict[str, object]
    latency: DreamReferenceLatency
    internal_stages: tuple[DreamInternalStage, ...]
    internal_binding_stages: tuple[str, ...]
    internal_bandwidth_bits_per_s: float
    internal_bandwidth_bytes_per_s: float
    interface_num_links: int
    interface_rate_gbps_per_link: float
    interface_payload_ecc_factor: float
    interface_bandwidth_bits_per_s: float
    interface_bandwidth_bytes_per_s: float
    effective_bandwidth_bytes_per_s: float
    ratio_internal_over_interface: float
    balance_relative_tolerance: float
    bottleneck: DreamBottleneck
    ratio_scale_invariance_note: str
    latency_gate: str
    internal_bandwidth_gate: str
    interface_bandwidth_gate: str
    aggregation_rule: str
    aggregation_rule_classification: str
    provenance: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["internal_stages"] = [stage.as_dict() for stage in
                                     self.internal_stages]
        result["latency"] = self.latency.as_dict()
        return result


def classify_bottleneck(
        ratio_internal_over_interface: float,
        *,
        balance_relative_tolerance: float = BALANCE_RELATIVE_TOLERANCE,
        ) -> DreamBottleneck:
    """Classify the reference bottleneck from the bandwidth ratio.

    ``R > 1`` means the external interface binds, ``R < 1`` means the
    internal memory side binds, and ``R ~= 1`` (within the relative
    tolerance) means the two stages are balanced.
    """
    ratio = float(ratio_internal_over_interface)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("internal/interface ratio must be finite, positive")
    if abs(ratio - 1.0) <= balance_relative_tolerance:
        return "BALANCED"
    return "EXTERNAL_INTERFACE" if ratio > 1.0 else "INTERNAL_MEMORY"


def audit_dream_reference_service(
        project_root: Path,
        *,
        memory_config: str = (
            "third_party/DreamRAM/configs/mem/baseline/hbm3_baseline.json"),
        technology_config: str = (
            "third_party/DreamRAM/configs/tech/scaled/16nm_scaled.json"),
        balance_relative_tolerance: float = BALANCE_RELATIVE_TOLERANCE,
        ) -> DreamReferenceServiceAudit:
    """Audit the pinned DreamRAM reference baseline memory service.

    Every quantity is produced by the pinned reference model's own
    methods; this function only adds the parallel-service aggregation
    rule ``B_int = N_parallel * D_service / T_service`` applied per
    shared internal stage.
    """
    root = Path(project_root)
    repo = root / "third_party" / "DreamRAM"
    _verify_pin(repo)
    memory_path = resolve_project_path(root, Path(memory_config)).resolve()
    technology_path = resolve_project_path(
        root, Path(technology_config)).resolve()

    with _loaded_dreamram(repo) as (tech_module, hbm_module, parse_module):
        import os
        previous_cwd = Path.cwd()
        try:
            # Reference configs resolve their baselines relative to the
            # pinned repository, matching its official CLI.
            os.chdir(repo)
            memory = parse_module.mem_baseline(str(memory_path))
            technology = parse_module.tech(str(technology_path))
        finally:
            os.chdir(previous_cwd)
        if memory is None or technology is None:
            raise ValueError("DreamRAM reference configuration unresolved")
        tech = tech_module.Tech(**technology)
        hbm_fields = {item.name for item in fields(hbm_module.Hbm)}
        hbm_kwargs = {key: value for key, value in memory.items()
                      if key in hbm_fields}
        hbm_kwargs["brv_sa"] = memory["brvsa"]
        dram = hbm_module.Hbm(**hbm_kwargs)

        core_tck_ns = float(dram.core_tck(tech))
        tcl_ns = float(dram.tcl(tech))
        trcd_ns = float(dram.trcd(tech))
        trp_ns = float(dram.trp(tech))
        atom_window_ns = float(dram.atom_time(tech))
        bank_clks_per_atom = int(dram.bank_clks_per_atom())
        dq_count = int(dram.dq_count())
        dq_speed_factor = float(dram.dq_speed_factor())
        md_ecc_factor = float(dram.md_ecc_factor())
        pages_per_pch = int(
            dram.ind_pages() * dram.banks * dram.vert_bg * dram.horiz_bg)
        mdl_width = float(dram.mdl_width_per_page())
        bgbus_width = float(dram.bgbus_width())
        gbus_width = float(dram.gbus_width())
        bgbuses_per_pch = pages_per_pch / float(dram.pages_per_bgbus_mux)
        gbuses_per_pch = float(dram.gbuses())
        gbus_tsv_sd = float(dram.gbus_tsv_sd)
        tsv_wires_per_pch = gbuses_per_pch * gbus_width / gbus_tsv_sd
        channels = int(dram.channels)
        pch = int(dram.pch)
        reference_interface_bytes_per_s = float(
            dram.bandwidth(tech)) * 1e9

    organization = {
        "ranks": int(dram.ranks),
        "channels": channels,
        "channels_per_die": int(dram.ch_per_die),
        "pseudochannels": pch,
        "horizontal_bankgroups": int(dram.horiz_bg),
        "vertical_bankgroups": int(dram.vert_bg),
        "banks_per_bankgroup": int(dram.banks),
        "subarrays_per_bank": int(dram.subarrays),
        "mats_per_subarray": int(dram.mats),
        "mat_rows": int(dram.mat_rows),
        "mat_cols": int(dram.mat_cols),
        "dies_stacked": int(dram.ranks * dram.channels / dram.ch_per_die),
        "capacity_gbytes": float(dram.capacity()),
        "atom_size_bits": int(dram.atom_size),
        "page_act_size_bits": int(dram.page_act_size()),
        "atoms_per_page": float(dram.atoms_per_page()),
        "pumps_per_atom": int(dram.pumps_per_atom()),
        "independent_pages_per_pseudochannel": pages_per_pch,
        "pages_per_bgbus_mux": float(dram.pages_per_bgbus_mux),
        "bgbuses_per_gbus": float(dram.bgbuses_per_gbus),
        "gbuses_per_pseudochannel": gbuses_per_pch,
        "gbus_tsv_serdes": gbus_tsv_sd,
        "tsv_dq_serdes": float(dram.tsv_dq_sd),
    }

    latency = DreamReferenceLatency(
        trcd_ns=trcd_ns,
        trp_ns=trp_ns,
        tcl_ns=tcl_ns,
        first_access_latency_ns=tcl_ns + trcd_ns + trp_ns,
        core_tck_ns=core_tck_ns,
        bank_clks_per_atom=bank_clks_per_atom,
        repeated_service_cycle_ns=bank_clks_per_atom * core_tck_ns,
        dq_atom_window_ns=atom_window_ns,
        first_access_definition=(
            "TRP_PLUS_TRCD_PLUS_TCL_CLOSED_ROW_FIRST_ACCESS_"
            "MATCHES_REFERENCE_WORST_LATENCY_NS"),
        repeated_service_cycle_definition=(
            "BANK_COLUMN_SERVICE_CYCLE_BANK_CLKS_PER_ATOM_X_CORE_TCK"),
        timing_classification="DERIVED_FROM_PAPER",
    )

    pseudochannels = channels * pch
    core_clock_hz = 1.0 / (core_tck_ns * 1e-9)

    def stage(name: str, units: float, payload_bits: float,
              ) -> DreamInternalStage:
        aggregate_bits_per_clk = units * payload_bits
        return DreamInternalStage(
            stage=name,
            parallel_units_per_pseudochannel=units,
            payload_bits_per_core_clock_per_unit=payload_bits,
            aggregate_bits_per_core_clock_per_pseudochannel=(
                aggregate_bits_per_clk),
            aggregate_bits_per_s=(
                pseudochannels * aggregate_bits_per_clk * core_clock_hz),
            classification="DERIVED_FROM_PAPER",
        )

    internal_stages = (
        stage("array_mdl", float(pages_per_pch), mdl_width),
        stage("bgbus", bgbuses_per_pch, bgbus_width),
        stage("gbus", gbuses_per_pch, gbus_width),
        # The TSV stage is rate-matched to the gbus stage by the
        # reference model's own serdes construction: fewer wires running
        # at the gbus_tsv_serdes multiple of the core clock.
        stage("tsv", tsv_wires_per_pch, gbus_tsv_sd),
    )
    internal_bits_per_s = min(
        item.aggregate_bits_per_s for item in internal_stages)
    binding = tuple(
        item.stage for item in internal_stages
        if item.aggregate_bits_per_s == internal_bits_per_s)

    interface_rate_gbps = dq_speed_factor / core_tck_ns
    interface_payload_bits_per_s = (
        dq_count * interface_rate_gbps * 1e9 / md_ecc_factor)
    interface_bytes_per_s = interface_payload_bits_per_s / 8.0
    if not math.isclose(
            interface_bytes_per_s, reference_interface_bytes_per_s,
            rel_tol=1e-12):
        raise RuntimeError(
            "DreamRAM reference interface reconstruction does not close: "
            f"{interface_bytes_per_s} != "
            f"{reference_interface_bytes_per_s}")

    internal_bytes_per_s = internal_bits_per_s / 8.0
    ratio = internal_bytes_per_s / interface_bytes_per_s
    bottleneck = classify_bottleneck(
        ratio, balance_relative_tolerance=balance_relative_tolerance)
    effective = min(internal_bytes_per_s, interface_bytes_per_s)

    return DreamReferenceServiceAudit(
        branch=DREAMRAM_BRANCH,
        commit=DREAMRAM_COMMIT,
        memory_config=str(memory_path),
        technology_config=str(technology_path),
        organization=organization,
        latency=latency,
        internal_stages=internal_stages,
        internal_binding_stages=binding,
        internal_bandwidth_bits_per_s=internal_bits_per_s,
        internal_bandwidth_bytes_per_s=internal_bytes_per_s,
        interface_num_links=dq_count,
        interface_rate_gbps_per_link=interface_rate_gbps,
        interface_payload_ecc_factor=md_ecc_factor,
        interface_bandwidth_bits_per_s=interface_payload_bits_per_s,
        interface_bandwidth_bytes_per_s=interface_bytes_per_s,
        effective_bandwidth_bytes_per_s=effective,
        ratio_internal_over_interface=ratio,
        balance_relative_tolerance=balance_relative_tolerance,
        bottleneck=bottleneck,
        ratio_scale_invariance_note=(
            "Both internal stages and the DQ interface scale with "
            "1/core_tck, and dq_count is derived from the internal bus "
            "hierarchy; the internal/interface ratio is therefore "
            "invariant to first-order core-cycle timing uncertainty."),
        latency_gate=GATE_PASS,
        internal_bandwidth_gate=GATE_PASS,
        interface_bandwidth_gate=GATE_PASS,
        aggregation_rule=(
            "B_INT=MIN_OVER_SHARED_INTERNAL_STAGES(N_PARALLEL_PER_PCH_"
            "X_PAYLOAD_BITS_PER_CORE_CLOCK)X_CHANNELS_X_PCH/CORE_TCK"),
        aggregation_rule_classification="MODELING_CHOICE",
        provenance={
            "organization": "EXISTING_REPO_REFERENCE",
            "timing": "DERIVED_FROM_PAPER",
            "internal_stage_rates": "DERIVED_FROM_PAPER",
            "aggregation_rule": "MODELING_CHOICE",
            "interface_bandwidth": "DERIVED_FROM_PAPER",
            "pinned_reference": "EXISTING_REPO_REFERENCE",
        },
    )
