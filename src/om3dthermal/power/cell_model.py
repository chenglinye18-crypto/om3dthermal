"""Component-level DRAM cell/device replacement boundary."""

from __future__ import annotations

from dataclasses import dataclass


ONE_T_ONE_C_SPECIFIC = frozenset({"bl-act", "bl-pre"})
REUSABLE_STRUCTURE = frozenset({
    "row", "mwl", "lwl", "col", "csl", "ldl", "mdl", "bgbus+gbus",
})


class MissingCellReplacementError(ValueError):
    """A required native component has no validated replacement energy."""


@dataclass(frozen=True)
class CellReplacementResolution:
    memory_internal_pj_bit: float
    native_components: dict[str, float]
    replacement_components: dict[str, float]


@dataclass(frozen=True)
class DeviceOperationEnergies:
    """Device/block operation table; not a complete memory energy model."""

    read_0: float
    read_1: float
    write_00: float
    write_01: float
    write_10: float
    write_11: float
    refresh_0: float
    refresh_1: float
    background_type: str | None
    background_value_W: float | None
    retention_s: float | None

    def weighted_read(self, *, p0: float, p1: float) -> float:
        return p0 * self.read_0 + p1 * self.read_1

    def weighted_write(
            self, *, p00: float, p01: float,
            p10: float, p11: float) -> float:
        return (
            p00 * self.write_00 + p01 * self.write_01
            + p10 * self.write_10 + p11 * self.write_11)

    def weighted_refresh(self, *, p0: float, p1: float) -> float:
        return p0 * self.refresh_0 + p1 * self.refresh_1


def apply_component_replacements(
        native_components: dict[str, float], *,
        required_components: tuple[str, ...],
        replacement_components: dict[str, float]) -> CellReplacementResolution:
    """Replace selected native blocks without retaining their native energy."""
    required = set(required_components)
    unknown = required - ONE_T_ONE_C_SPECIFIC
    if unknown:
        raise ValueError(
            "only audited 1T1C-specific components are replaceable; got "
            + ", ".join(sorted(unknown)))
    absent_native = required - set(native_components)
    if absent_native:
        raise ValueError(
            "required native components are absent: "
            + ", ".join(sorted(absent_native)))
    missing = required - set(replacement_components)
    if missing:
        raise MissingCellReplacementError(
            "required replacement components are unresolved: "
            + ", ".join(sorted(missing)))
    extra = set(replacement_components) - required
    if extra:
        raise ValueError(
            "replacement values supplied for non-required components: "
            + ", ".join(sorted(extra)))

    retained = {
        name: energy for name, energy in native_components.items()
        if name not in required
    }
    replacements = {
        name: float(replacement_components[name]) for name in required
    }
    return CellReplacementResolution(
        memory_internal_pj_bit=sum(retained.values()) + sum(replacements.values()),
        native_components=retained,
        replacement_components=replacements,
    )


def apply_operation_primitive_replacement(
        native_components: dict[str, float], *,
        required_components: tuple[str, ...],
        operation_energy_pj_per_bit: float,
        primitive_name: str = "mat_local_operation",
        ) -> CellReplacementResolution:
    """Replace a native component set with one indivisible operation primitive."""
    if operation_energy_pj_per_bit < 0.0:
        raise ValueError("operation replacement energy must be non-negative")
    # Reuse component-boundary validation without inventing a split of the
    # reported operation primitive across bl-act and bl-pre.
    validated = apply_component_replacements(
        native_components,
        required_components=required_components,
        replacement_components={name: 0.0 for name in required_components},
    )
    replacements = {primitive_name: float(operation_energy_pj_per_bit)}
    return CellReplacementResolution(
        memory_internal_pj_bit=(
            sum(validated.native_components.values())
            + operation_energy_pj_per_bit),
        native_components=validated.native_components,
        replacement_components=replacements,
    )
