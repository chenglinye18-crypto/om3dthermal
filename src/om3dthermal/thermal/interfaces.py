"""Per-material-pair interface areal resistance registry.

For every pair of materials that share an internal face, the registry
answers the question "what ``R''`` (in SI m^2*K/W) should be added to
the per-edge two-point conductance?"

Rules in priority order:

1. explicit unordered pair rule (e.g. ``[Silicon, Oxide]``);
2. the default ``default_interface_areal_resistance``.

Same-material pairs are allowed but the default is the only sane
choice (the explicit-rule form exists for completeness so users can
e.g. document that two silicon cells across a TSV interface have a
non-zero ``R''`` if they ever need to).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config import InterfaceResistanceConfig


@dataclass(frozen=True)
class InterfaceResistanceQuery:
    """Result of a registry lookup."""

    value: float                # m^2*K/W, always >= 0
    rule_index: int             # -1 means default rule
    used_default: bool


class InterfaceResistanceRegistry:
    """Holds the default + per-pair rules and resolves lookups.

    Construction is O(N_pair) and lookup is O(1) (dict). Duplicate
    unordered pairs are rejected at construction time; ordering of the
    pair in the config is ignored so ``[Silicon, Oxide]`` and
    ``[Oxide, Silicon]`` would be flagged as duplicates.
    """

    def __init__(self, default_areal_resistance: float,
                 rules: Sequence[InterfaceResistanceConfig]) -> None:
        if default_areal_resistance < 0:
            raise ValueError(
                f"default areal resistance must be non-negative, got "
                f"{default_areal_resistance}")
        self._default = float(default_areal_resistance)
        self._rules: dict[tuple[str, str], tuple[int, float]] = {}
        for index, rule in enumerate(rules):
            if len(rule.materials) != 2:
                raise ValueError(
                    f"interface rule {index} must specify exactly two "
                    f"materials, got {len(rule.materials)}")
            a, b = rule.materials
            if not a or not b:
                raise ValueError(
                    f"interface rule {index} has empty material name")
            key = tuple(sorted((a, b)))
            if key in self._rules:
                existing_index, _ = self._rules[key]
                raise ValueError(
                    f"duplicate interface rule for unordered pair "
                    f"{key!r} (rules {existing_index} and {index}); "
                    f"pairs are unordered so [{a!r}, {b!r}] and "
                    f"[{b!r}, {a!r}] are the same rule")
            if rule.areal_resistance < 0:
                raise ValueError(
                    f"interface rule {index} ({a!r}/{b!r}) has negative "
                    f"areal resistance {rule.areal_resistance}")
            self._rules[key] = (index, float(rule.areal_resistance))

    @property
    def default(self) -> float:
        return self._default

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def lookup(self, material_a: str, material_b: str
               ) -> InterfaceResistanceQuery:
        key = tuple(sorted((material_a, material_b)))
        if key in self._rules:
            rule_index, value = self._rules[key]
            return InterfaceResistanceQuery(
                value=value, rule_index=rule_index, used_default=False)
        return InterfaceResistanceQuery(
            value=self._default, rule_index=-1, used_default=True)
