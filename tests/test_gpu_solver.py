"""Targeted CPU/CuPy matrix-free operator and PCG equivalence tests."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from om3dthermal.thermal import (
    BoundaryLinkTable,
    ConductanceTable,
    CuPyMatrixFreeThermalOperator,
    build_matrix_free_operator,
    require_cupy,
    solve_pcg,
    solve_pcg_gpu,
)


def _small_graph(cell_count: int = 31):
    cell_a = np.arange(cell_count - 1, dtype=np.int64)
    cell_b = np.arange(1, cell_count, dtype=np.int64)
    conductance = ConductanceTable(
        edge_id=np.arange(cell_count - 1, dtype=np.int64),
        cell_a=cell_a,
        cell_b=cell_b,
        axis=np.zeros(cell_count - 1, dtype=np.int8),
        face_area_m2=np.full(cell_count - 1, 1e-6),
        half_distance_a_m=np.full(cell_count - 1, 5e-5),
        half_distance_b_m=np.full(cell_count - 1, 5e-5),
        k_normal_a_W_mK=np.full(cell_count - 1, 10.0),
        k_normal_b_W_mK=np.full(cell_count - 1, 10.0),
        interface_areal_resistance_m2K_W=np.zeros(cell_count - 1),
        resistance_K_W=np.full(cell_count - 1, 10.0),
        conductance_W_K=np.linspace(0.04, 0.07, cell_count - 1),
        material_interface=np.zeros(cell_count - 1, dtype=bool),
        interface_rule_index=np.full(cell_count - 1, -1, dtype=np.int32),
    )
    boundary = BoundaryLinkTable(
        boundary_face_id=np.arange(2, dtype=np.int64),
        cell_id=np.array([0, cell_count - 1], dtype=np.int64),
        kind=np.array([2, 1], dtype=np.int8),
        axis=np.array([0, 0], dtype=np.int8),
        side=np.array([-1, 1], dtype=np.int8),
        face_area_m2=np.full(2, 1e-6),
        half_distance_m=np.full(2, 5e-5),
        k_normal_W_mK=np.full(2, 10.0),
        areal_resistance_m2K_W=np.zeros(2),
        external_film_resistance_m2K_W=np.zeros(2),
        conductance_W_K=np.array([0.12, 0.08]),
        reference_temperature_K=np.array([350.0, 295.0]),
        rule_index=np.zeros(2, dtype=np.int32),
    )
    power = np.linspace(0.001, 0.003, cell_count)
    return build_matrix_free_operator(conductance, boundary, power), boundary


def test_cpu_package_import_does_not_import_cupy():
    code = (
        "import sys; import om3dthermal.thermal; "
        "assert 'cupy' not in sys.modules; print('cpu-import-ok')")
    completed = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True,
        text=True)
    assert completed.stdout.strip() == "cpu-import-ok"


def test_explicit_gpu_request_fails_clearly_when_cupy_is_blocked():
    code = r'''
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "cupy" or name.startswith("cupy."):
        raise ModuleNotFoundError("blocked for targeted test")
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
from om3dthermal.thermal.gpu_solver import (
    GPUBackendUnavailableError, require_cupy)
try:
    require_cupy()
except GPUBackendUnavailableError as exc:
    assert "GPU backend requested" in str(exc)
    print("clear-gpu-error")
else:
    raise AssertionError("GPU request silently succeeded/fell back")
'''
    completed = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True,
        text=True)
    assert completed.stdout.strip() == "clear-gpu-error"


def test_gpu_operator_matches_cpu_fp64():
    cp = require_cupy()
    operator, _ = _small_graph()
    gpu = CuPyMatrixFreeThermalOperator.from_cpu(operator)
    rng = np.random.default_rng(2026)
    vector = rng.normal(320.0, 15.0, operator.cell_count)
    cpu_result = operator.apply(vector)
    gpu_result = cp.asnumpy(gpu.apply(cp.asarray(vector, dtype=cp.float64)))
    cp.cuda.Stream.null.synchronize()
    error = gpu_result - cpu_result
    assert np.max(np.abs(error)) < 1e-12
    assert np.linalg.norm(error) / np.linalg.norm(cpu_result) < 1e-13


def test_small_gpu_pcg_matches_cpu_full_temperature_vector():
    operator, boundary = _small_graph()
    initial = np.full(operator.cell_count, 293.15, dtype=np.float64)
    cpu = solve_pcg(
        operator, initial, boundary,
        relative_residual_tolerance=1e-11, max_iterations=2000)
    gpu = solve_pcg_gpu(
        operator, initial, boundary,
        relative_residual_tolerance=1e-11, max_iterations=2000)
    assert cpu.converged and gpu.converged
    assert np.all(np.isfinite(gpu.temperature_K))
    np.testing.assert_allclose(
        gpu.temperature_K, cpu.temperature_K, rtol=1e-10, atol=1e-8)
    difference = gpu.temperature_K - cpu.temperature_K
    assert np.max(np.abs(difference)) < 1e-7
    assert (np.linalg.norm(difference) / np.linalg.norm(cpu.temperature_K)
            < 1e-10)
    assert gpu.min_temperature_K == pytest.approx(
        cpu.min_temperature_K, abs=1e-7)
    assert gpu.max_temperature_K == pytest.approx(
        cpu.max_temperature_K, abs=1e-7)
    assert gpu.mean_temperature_K == pytest.approx(
        cpu.mean_temperature_K, abs=1e-7)
    assert gpu.final_relative_residual <= 1e-11
    assert gpu.solver_info["per_iteration_vector_transfers"] == 0
    assert gpu.solver_info["device_vector_downloads"] == 1
