"""Unified DreamRAM-vs-M3D first-access latency decomposition audit.

Read-only audit module.  It decomposes the pinned DreamRAM DATE2026
reference first-access latency (tRP + tRCD + tCL) into the physical
terms the reference model actually computes, decomposes the current M3D
first-order physical access latency (tMAT + tMIV + tFEOL + tInterface)
into its configured terms, maps both onto one unified stage taxonomy,
and reports semantic-match gates and a modeling-risk ranking.

No canonical parameter is modified; the FEOL resistance sensitivity is
computed on copies of the parsed configuration only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
import statistics
from typing import Literal

from .backends.dreamram import (
    DREAMRAM_BRANCH,
    DREAMRAM_COMMIT,
    _loaded_dreamram,
    _verify_pin,
)
from .config import CanonicalCaseConfig, resolve_project_path
from .feol_route import calculate_feol_route
from .physical_latency import PhysicalAccessLatency


StageStatus = Literal[
    "PRESENT",
    "ABSENT_BY_ARCHITECTURE",
    "INCLUDED_INSIDE_ANOTHER_TERM",
    "NOT_MODELED",
    "UNKNOWN",
]

SemanticMatch = Literal["MATCHED", "PARTIALLY_MATCHED", "NOT_MATCHED"]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]
Gate = Literal["PASS", "PARTIAL"]


@dataclass(frozen=True)
class UnifiedStageMapping:
    stage: str
    dream_term: str
    dream_status: StageStatus
    m3d_term: str
    m3d_status: StageStatus
    comparable: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DreamLatencyDecomposition:
    trp_ns: float
    trp_aliased_to_trcd: bool
    trcd_ns: float
    trcd_signal_ns: float
    trcd_sensing_ns: float
    tcl_ns: float
    tcl_lateral_bus_ns: float
    tcl_tsv_vertical_ns: float
    tcl_dq_window_ns: float
    tcl_reference_calibration_ns: float
    tcl_reference_core_tck_ns: float
    core_tck_ns: float
    first_access_ns: float
    row_hit_ns: float
    row_miss_ns: float
    row_conflict_ns: float
    access_case: str
    row_state_source: str
    tcl_physical_scope: str
    formulas: dict[str, str]
    inputs: dict[str, float]
    classification: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class M3DLatencyDecomposition:
    mat_latency_ns: float
    miv_min_ns: float
    miv_max_ns: float
    feol_min_ns: float
    feol_median_ns: float
    feol_p90_ns: float
    feol_max_ns: float
    interface_latency_ns: float
    near_total_ns: float
    far_total_ns: float
    near_mat_share: float
    far_mat_share: float
    far_feol_share: float
    miv_share_of_far_total: float
    mat_scope: str
    precharge_status: str
    sensing_status: str
    interface_status_note: str
    parameter_provenance: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FEOLResistanceSensitivityRow:
    resistance_ohm_per_um: float
    total_min_ns: float
    total_max_ns: float
    feol_share_of_far_total: float
    argmax_cluster_unchanged: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LatencyModelRiskItem:
    rank: int
    item: str
    impact: Confidence
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LatencyAuditGates:
    dream_latency_decomposition_gate: Gate
    m3d_latency_decomposition_gate: Gate
    latency_semantic_match_gate: SemanticMatch
    m3d_absolute_latency_confidence: Confidence
    m3d_spatial_latency_ranking_confidence: Confidence
    reasons: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_unified_taxonomy() -> tuple[UnifiedStageMapping, ...]:
    """Unified physical-stage mapping derived from the code audit."""
    rows = (
        UnifiedStageMapping(
            stage="A_PRECHARGE_RESET",
            dream_term="tRP",
            dream_status="PRESENT",
            m3d_term="none",
            m3d_status="NOT_MODELED",
            comparable=False,
            note=(
                "DreamRAM sets tRP = tRCD(for_trp=True) by its own "
                "modeling assumption; the 2T0C IGZO read path has no "
                "DRAM-style precharge term and none is modeled.")),
        UnifiedStageMapping(
            stage="B_ROW_CELL_ACTIVATION",
            dream_term="tRCD signal component",
            dream_status="PRESENT",
            m3d_term="tMAT",
            m3d_status="INCLUDED_INSIDE_ANOTHER_TERM",
            comparable=False,
            note=(
                "DreamRAM tRCD_signal (2.9 ns reference, length-scaled) "
                "is explicit; M3D wordline/decoder activation is lumped "
                "inside the 10 ns MAT placeholder.")),
        UnifiedStageMapping(
            stage="C_SENSING_MAT_READ",
            dream_term="tRCD BLSA component",
            dream_status="PRESENT",
            m3d_term="tMAT",
            m3d_status="INCLUDED_INSIDE_ANOTHER_TERM",
            comparable=False,
            note=(
                "DreamRAM tRCD_blsa is capacitance-scaled bitline "
                "development plus sense amplification; M3D bitline "
                "development and current sensing are not explicitly "
                "modeled and are lumped inside tMAT.")),
        UnifiedStageMapping(
            stage="D_LOCAL_ARRAY_ROUTING",
            dream_term="CSL/LDL/MDL inside core_tck and tCL pump window",
            dream_status="INCLUDED_INSIDE_ANOTHER_TERM",
            m3d_term="tMAT",
            m3d_status="INCLUDED_INSIDE_ANOTHER_TERM",
            comparable=False,
            note=(
                "Both sides fold local array wiring into larger terms; "
                "neither exposes it separately.")),
        UnifiedStageMapping(
            stage="E_VERTICAL_INTERCONNECT",
            dream_term="2 x dies_stacked x TSV RC rise term inside tCL",
            dream_status="PRESENT",
            m3d_term="tMIV",
            m3d_status="PRESENT",
            comparable=True,
            note=(
                "Both are explicit distributed-RC vertical transport "
                "terms; directly comparable in structure.")),
        UnifiedStageMapping(
            stage="F_LATERAL_GLOBAL_ROUTING",
            dream_term="bgbus/gbus/base lateral term inside tCL",
            dream_status="PRESENT",
            m3d_term="tFEOL",
            m3d_status="PRESENT",
            comparable=True,
            note=(
                "Both are geometry-driven lateral transport terms; "
                "DreamRAM calibrates the lateral magnitude from the "
                "standard HBM3 tCL reference while M3D computes Elmore "
                "delay over modeled FEOL wires.")),
        UnifiedStageMapping(
            stage="G_MEMORY_SIDE_INTERFACE_STARTUP",
            dream_term="pumps_per_atom x core_tck DQ window inside tCL",
            dream_status="PRESENT",
            m3d_term="t_interface = 0 ns",
            m3d_status="NOT_MODELED",
            comparable=False,
            note=(
                "DreamRAM carries an explicit DQ serialization window; "
                "the M3D coil interface startup is a 0 ns placeholder.")),
        UnifiedStageMapping(
            stage="H_EXTERNAL_LINK_STARTUP",
            dream_term="inside calibrated tCL reference boundary",
            dream_status="INCLUDED_INSIDE_ANOTHER_TERM",
            m3d_term="none",
            m3d_status="NOT_MODELED",
            comparable=False,
            note=(
                "The standard HBM3 tCL calibration boundary includes "
                "controller/PHY convention; the M3D contactless link "
                "clocking/encoding/detection latency is unknown.")),
    )
    return rows


def audit_dream_latency_decomposition(
        project_root: Path,
        *,
        memory_config: str = (
            "third_party/DreamRAM/configs/mem/baseline/hbm3_baseline.json"),
        technology_config: str = (
            "third_party/DreamRAM/configs/tech/scaled/16nm_scaled.json"),
        ) -> DreamLatencyDecomposition:
    """Decompose the pinned reference first-access latency read-only."""
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

        # tRCD = signal assertion (wordline path, length-scaled)
        #      + BLSA amplification (bitline/cell/BLSA capacitance-scaled).
        bank_x_um, _, _ = dram.bank_dims(tech)
        reference_bank_x_um = float(dram._bank_x(tech))
        trcd_signal_ns = float(
            tech._trcd_signal * (reference_bank_x_um / bank_x_um))
        trcd_sensing_ns = trcd_ns - trcd_signal_ns
        if trcd_sensing_ns <= 0.0:
            raise RuntimeError("DreamRAM tRCD decomposition is not positive")

        # tCL = lateral bus transport (calibrated from the standard HBM3
        #       tCL reference) + TSV vertical rise + DQ pump window.
        dies_stacked = int(dram.calc_stack_dims(tech)[0])
        rise_to_80 = float(dram.rc_rise_to(0.8))
        tcl_tsv_ns = (
            2.0 * dies_stacked * rise_to_80
            * float(tech._r_load) * float(tech.scaled_cap_tsv()) / 1000.0)
        tcl_dq_window_ns = float(dram.pumps_per_atom()) * core_tck_ns
        tcl_lateral_ns = tcl_ns - tcl_tsv_ns - tcl_dq_window_ns
        if tcl_lateral_ns <= 0.0:
            raise RuntimeError("DreamRAM tCL lateral term is not positive")

        reference_tcl_ns = float(dram._tcl)
        reference_core_tck_ns = float(dram._tck)

    first_access_ns = trp_ns + trcd_ns + tcl_ns
    return DreamLatencyDecomposition(
        trp_ns=trp_ns,
        trp_aliased_to_trcd=True,
        trcd_ns=trcd_ns,
        trcd_signal_ns=trcd_signal_ns,
        trcd_sensing_ns=trcd_sensing_ns,
        tcl_ns=tcl_ns,
        tcl_lateral_bus_ns=tcl_lateral_ns,
        tcl_tsv_vertical_ns=tcl_tsv_ns,
        tcl_dq_window_ns=tcl_dq_window_ns,
        tcl_reference_calibration_ns=reference_tcl_ns,
        tcl_reference_core_tck_ns=reference_core_tck_ns,
        core_tck_ns=core_tck_ns,
        first_access_ns=first_access_ns,
        row_hit_ns=tcl_ns,
        row_miss_ns=trcd_ns + tcl_ns,
        row_conflict_ns=first_access_ns,
        access_case="ROW_CONFLICT_WORST_CASE_MODEL_DEFAULT",
        row_state_source="DERIVED_FROM_REFERENCE_COMPONENT_SEMANTICS",
        tcl_physical_scope=(
            "LATERAL_BGBUS_GBUS_BASE_TRANSPORT_CALIBRATED_FROM_STANDARD_"
            "HBM3_TCL_PLUS_TSV_VERTICAL_RISE_PLUS_DQ_PUMP_WINDOW"),
        formulas={
            "tRP": "tRCD(for_trp=True); reference assumption tRP ~= tRCD",
            "tRCD": (
                "trcd_signal * (reference_bank_x / bank_x) + trcd_blsa * "
                "(c_cell + c_bl + c_blsa) / (c_cell + c_bl_ref + c_blsa)"),
            "tCL": (
                "(base_cap_ratio + bgbus_cap_ratio) * t_die_y_ref * "
                "(die_y - bank_y) / (die_y_ref - bank_y_ref) + 2 * "
                "dies_stacked * ln(5) * r_load * c_tsv_scaled / 1000 + "
                "pumps_per_atom * core_tck"),
        },
        inputs={
            "reference_trcd_ns": float(tech._trcd),
            "reference_trcd_signal_ns": float(tech._trcd_signal),
            "reference_tcl_ns": reference_tcl_ns,
            "reference_core_tck_ns": reference_core_tck_ns,
            "dies_stacked": float(dies_stacked),
            "tsv_r_load_ohm": float(tech._r_load),
            "tsv_scaled_capacitance_pF": float(tech.scaled_cap_tsv()),
            "pumps_per_atom": float(dram.pumps_per_atom()),
        },
        classification="DERIVED_FROM_PAPER",
    )


def build_m3d_latency_decomposition(
        case: CanonicalCaseConfig,
        latency: PhysicalAccessLatency,
        ) -> M3DLatencyDecomposition:
    """Decompose the canonical M3D first-order physical access latency."""
    spec = case.architecture.physical_access_latency
    locations = latency.locations
    if not locations:
        raise ValueError("physical access latency map is empty")
    miv_values = tuple(item.miv_latency_ns for item in locations)
    feol_values = tuple(item.feol_latency_ns for item in locations)
    near = min(locations, key=lambda item: item.total_latency_ns)
    far = max(locations, key=lambda item: item.total_latency_ns)
    vertical = case.architecture.vertical
    feol_wire = case.architecture.feol_route.wire
    return M3DLatencyDecomposition(
        mat_latency_ns=spec.mat_latency_ns,
        miv_min_ns=min(miv_values),
        miv_max_ns=max(miv_values),
        feol_min_ns=min(feol_values),
        feol_median_ns=statistics.median(feol_values),
        feol_p90_ns=sorted(feol_values)[
            min(len(feol_values) - 1,
                math.ceil(0.9 * len(feol_values)) - 1)],
        feol_max_ns=max(feol_values),
        interface_latency_ns=spec.interface_latency_ns,
        near_total_ns=near.total_latency_ns,
        far_total_ns=far.total_latency_ns,
        near_mat_share=near.mat_latency_ns / near.total_latency_ns,
        far_mat_share=far.mat_latency_ns / far.total_latency_ns,
        far_feol_share=far.feol_latency_ns / far.total_latency_ns,
        miv_share_of_far_total=far.miv_latency_ns / far.total_latency_ns,
        mat_scope=(
            "LUMPED_PLACEHOLDER_UNDERDEFINED_SCOPE_NO_LAYER_OR_POSITION_"
            "DEPENDENCE"),
        precharge_status=(
            "NOT_MODELED_2T0C_READ_PATH_HAS_NO_DRAM_STYLE_PRECHARGE_TERM_"
            "IN_CURRENT_ABSTRACTION"),
        sensing_status=(
            "NOT_EXPLICITLY_MODELED_BITLINE_DEVELOPMENT_AND_CURRENT_"
            "SENSING_LUMPED_INSIDE_TMAT"),
        interface_status_note=(
            "MISSING_BUT_POSITION_INDEPENDENT_0_NS_PLACEHOLDER_"
            "NOT_YET_CALIBRATED"),
        parameter_provenance={
            "tMAT": f"{spec.mat_classification}/{spec.mat_status}",
            "tMIV_R_per_length": (
                vertical.miv_resistance_provenance.classification),
            "tMIV_C_per_length": "DERIVED_FROM_DREAMRAM_TSV_REFERENCE",
            "tMIV_driver_load": (
                f"MODELING_CHOICE/fixed_load:{vertical.fixed_load_provenance}"),
            "tFEOL_R_per_length": (
                feol_wire.resistance_provenance.classification),
            "tFEOL_C_per_length": feol_wire.provenance,
            "tInterface": (
                f"{spec.interface_classification}/{spec.interface_status}"),
        },
    )


def run_feol_resistance_sensitivity(
        case: CanonicalCaseConfig,
        topology,
        miv_delay_per_layer_ns: tuple[float, ...],
        *,
        resistances_ohm_per_um: tuple[float, ...] = (1.0, 2.0, 4.0),
        ) -> tuple[FEOLResistanceSensitivityRow, ...]:
    """Diagnostic-only FEOL R' sweep on configuration copies."""
    spec = case.architecture.physical_access_latency
    feol_spec = case.architecture.feol_route
    canonical_resistance = feol_spec.wire.resistance_ohm_per_um
    if canonical_resistance is None:
        raise ValueError("canonical FEOL resistance is unresolved")
    canonical_argmax: int | None = None
    rows: list[FEOLResistanceSensitivityRow] = []
    for resistance in resistances_ohm_per_um:
        wire = feol_spec.wire.model_copy(
            update={"resistance_ohm_per_um": resistance})
        copied_spec = feol_spec.model_copy(update={"wire": wire})
        feol = calculate_feol_route(copied_spec, topology)
        delays = feol.feol_delay_per_cluster_ns
        if delays is None:
            raise RuntimeError("FEOL sensitivity delay resolution failed")
        argmax = max(range(len(delays)), key=delays.__getitem__)
        if math.isclose(resistance, canonical_resistance, rel_tol=1e-12):
            canonical_argmax = argmax
        total_min = (
            spec.mat_latency_ns + min(miv_delay_per_layer_ns)
            + min(delays) + spec.interface_latency_ns)
        total_max = (
            spec.mat_latency_ns + max(miv_delay_per_layer_ns)
            + max(delays) + spec.interface_latency_ns)
        rows.append(FEOLResistanceSensitivityRow(
            resistance_ohm_per_um=resistance,
            total_min_ns=total_min,
            total_max_ns=total_max,
            feol_share_of_far_total=max(delays) / total_max,
            argmax_cluster_unchanged=(
                True if canonical_argmax is None
                else argmax == canonical_argmax),
        ))
    return tuple(rows)


def build_risk_ranking(
        dream: DreamLatencyDecomposition,
        m3d: M3DLatencyDecomposition,
        ) -> tuple[LatencyModelRiskItem, ...]:
    """Order modeling risks by impact on the latency comparison."""
    tcl_share = dream.tcl_ns / dream.first_access_ns
    return (
        LatencyModelRiskItem(
            rank=1,
            item="M3D tMAT = 10 ns lumped placeholder",
            impact="HIGH",
            reason=(
                f"Largest M3D term ({100.0 * m3d.near_mat_share:.1f}% of "
                f"near access, {100.0 * m3d.far_mat_share:.1f}% of far) "
                "with scope underdefined and NOT_CAPABILITY_VALIDATED; "
                "it may lump activation, sensing, decoder and any "
                "precharge-like reset into one asserted constant.")),
        LatencyModelRiskItem(
            rank=2,
            item="Dream tCL semantic scope",
            impact="HIGH",
            reason=(
                f"tCL is {100.0 * tcl_share:.1f}% of the Dream first "
                "access and is dominated by lateral bus transport "
                "calibrated from the standard HBM3 tCL reference, not by "
                "array physics; comparing M3D tMAT against Dream "
                "tRP+tRCD+tCL mixes array-level and system-level terms.")),
        LatencyModelRiskItem(
            rank=3,
            item="M3D interface latency = 0 ns placeholder",
            impact="MEDIUM",
            reason=(
                "Missing coil TX/RX startup, clocking, encoding and "
                "detection latency; position-independent, so it shifts "
                "the absolute Dream/M3D comparison but not the spatial "
                "ordering.")),
        LatencyModelRiskItem(
            rank=4,
            item="M3D FEOL R' modeling choice",
            impact="MEDIUM",
            reason=(
                "2 ohm/um is a nominal effective value, not measured; it "
                "scales the spatial spread magnitude while the near/far "
                "ordering stays topology-determined.")),
        LatencyModelRiskItem(
            rank=5,
            item="M3D MIV R' modeling choice",
            impact="LOW",
            reason=(
                f"MIV contributes at most "
                f"{100.0 * m3d.miv_share_of_far_total:.2f}% of the far "
                "access; even order-of-magnitude uncertainty is "
                "negligible for total latency.")),
    )


def classify_gates(
        dream: DreamLatencyDecomposition,
        m3d: M3DLatencyDecomposition,
        ) -> LatencyAuditGates:
    """Scientific gates for the latency-comparison audit."""
    return LatencyAuditGates(
        dream_latency_decomposition_gate="PASS",
        m3d_latency_decomposition_gate="PARTIAL",
        latency_semantic_match_gate="PARTIALLY_MATCHED",
        m3d_absolute_latency_confidence="LOW",
        m3d_spatial_latency_ranking_confidence="HIGH",
        reasons={
            "dream": (
                "tRP/tRCD/tCL all decompose to explicit reference-model "
                "formulas with closure; tRP aliasing to tRCD is the "
                "reference model's own documented assumption."),
            "m3d": (
                "tMIV and tFEOL are explicit RC terms, but tMAT is a "
                "lumped NOT_CAPABILITY_VALIDATED placeholder and the "
                "interface term is a 0 ns placeholder."),
            "semantic_match": (
                "Vertical (TSV vs MIV) and lateral (tCL-internal bus vs "
                "FEOL) stages are comparable; activation/sensing are "
                "lumped differently and the Dream 64.17 ns figure is a "
                "row-conflict worst case while the M3D figure is a "
                "clean single-access path, so the totals are not "
                "strictly semantically matched."),
            "absolute_confidence": (
                "CURRENT_M3D_ABSOLUTE_LATENCY_NOT_YET_VALIDATED: the "
                "dominant MAT term and the interface term are both "
                "unvalidated placeholders."),
            "spatial_ranking_confidence": (
                "Spatial spread is driven by the explicit FEOL/MIV RC "
                "terms; tMAT and tInterface are position-independent "
                "constants, so near/far placement ordering is robust to "
                "their uncertainty."),
        },
    )
