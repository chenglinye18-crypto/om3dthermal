"""Configuration and execution boundary for reproducible E2E experiments."""

from .config import (
    ExperimentScenarioSpec,
    ExperimentSpec,
    load_architecture_spec,
    load_experiment_spec,
    load_platform_spec,
    load_workload_spec,
)
from .result_bundle import RESULT_FILES, write_result_bundle
from .runner import ExperimentRunResult, run_experiment

__all__ = [
    "ExperimentScenarioSpec",
    "ExperimentSpec",
    "ExperimentRunResult",
    "RESULT_FILES",
    "load_architecture_spec",
    "load_experiment_spec",
    "load_platform_spec",
    "load_workload_spec",
    "run_experiment",
    "write_result_bundle",
]
