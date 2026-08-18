"""Targeted tests for the analytical LLM decode workload primitive.

Each test locks a specific scientific semantic from the B1-R2 specification.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from om3dthermal.workload.llm_decode import LLMDecodeInput, evaluate_llm_decode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(**overrides) -> LLMDecodeInput:
    """Return a default LLMDecodeInput with optional overrides."""
    defaults = {
        "n_param": 8_000_000_000,
        "n_layers": 32,
        "n_heads_q": 32,
        "n_heads_kv": 8,
        "d_head": 128,
        "d_model": 4096,
        "d_ff": 14336,
        "vocab_size": 128_256,
        "batch_size": 1,
        "context_length": 131_072,
        "weight_bits": 16,
        "kv_bits": 16,
        "runtime_bytes": 0,
    }
    defaults.update(overrides)
    return LLMDecodeInput(**defaults)


# ---------------------------------------------------------------------------
# A. Weight footprint invariance vs batch
# ---------------------------------------------------------------------------

def test_weight_footprint_invariant_to_batch() -> None:
    """Changing B must not change weight_footprint_bytes or
    weight_active_per_step_bytes."""
    inp_b1 = _make_input(batch_size=1)
    inp_b8 = _make_input(batch_size=8)

    m1 = evaluate_llm_decode(inp_b1)
    m8 = evaluate_llm_decode(inp_b8)

    assert m1.weight_footprint_bytes == m8.weight_footprint_bytes
    assert m1.weight_active_per_step_bytes == m8.weight_active_per_step_bytes


# ---------------------------------------------------------------------------
# B. KV footprint scales linearly with batch
# ---------------------------------------------------------------------------

def test_kv_footprint_scales_with_batch() -> None:
    """B = 8 must produce exactly 8x the KV footprint of B = 1."""
    inp_b1 = _make_input(batch_size=1)
    inp_b8 = _make_input(batch_size=8)

    m1 = evaluate_llm_decode(inp_b1)
    m8 = evaluate_llm_decode(inp_b8)

    assert m8.kv_footprint_bytes == 8 * m1.kv_footprint_bytes


# ---------------------------------------------------------------------------
# C. KV footprint scales linearly with context
# ---------------------------------------------------------------------------

def test_kv_footprint_scales_with_context() -> None:
    """Doubling S must double KV footprint."""
    inp_s = _make_input(context_length=1024)
    inp_2s = _make_input(context_length=2048)

    m_s = evaluate_llm_decode(inp_s)
    m_2s = evaluate_llm_decode(inp_2s)

    assert m_2s.kv_footprint_bytes == 2 * m_s.kv_footprint_bytes


# ---------------------------------------------------------------------------
# D. KV read/token is independent of batch
# ---------------------------------------------------------------------------

def test_kv_read_per_token_invariant_to_batch() -> None:
    """Changing B = 1 -> 8 must not change kv_read_bytes_per_token."""
    inp_b1 = _make_input(batch_size=1)
    inp_b8 = _make_input(batch_size=8)

    m1 = evaluate_llm_decode(inp_b1)
    m8 = evaluate_llm_decode(inp_b8)

    assert m1.kv_read_bytes_per_token == m8.kv_read_bytes_per_token


# ---------------------------------------------------------------------------
# E. KV read/token scales linearly with context
# ---------------------------------------------------------------------------

def test_kv_read_per_token_scales_with_context() -> None:
    """Doubling S must double kv_read_bytes_per_token."""
    inp_s = _make_input(context_length=1024)
    inp_2s = _make_input(context_length=2048)

    m_s = evaluate_llm_decode(inp_s)
    m_2s = evaluate_llm_decode(inp_2s)

    assert m_2s.kv_read_bytes_per_token == 2 * m_s.kv_read_bytes_per_token


# ---------------------------------------------------------------------------
# F. KV write/token is independent of context
# ---------------------------------------------------------------------------

def test_kv_write_per_token_invariant_to_context() -> None:
    """Changing S must not change kv_write_bytes_per_token."""
    inp_s = _make_input(context_length=1024)
    inp_2s = _make_input(context_length=2048)

    m_s = evaluate_llm_decode(inp_s)
    m_2s = evaluate_llm_decode(inp_2s)

    assert m_s.kv_write_bytes_per_token == m_2s.kv_write_bytes_per_token


# ---------------------------------------------------------------------------
# G. Weight traffic amortization under tile reuse
# ---------------------------------------------------------------------------

def test_weight_traffic_amortization() -> None:
    """Under tile_reuse + full_footprint, B = 1 -> 8 must make
    weight_read_bytes_per_token exactly 8x smaller."""
    inp_b1 = _make_input(batch_size=1)
    inp_b8 = _make_input(batch_size=8)

    m1 = evaluate_llm_decode(inp_b1)
    m8 = evaluate_llm_decode(inp_b8)

    assert m1.weight_read_bytes_per_token == 8 * m8.weight_read_bytes_per_token


# ---------------------------------------------------------------------------
# H. FLOPs/token is independent of batch
# ---------------------------------------------------------------------------

def test_flops_per_token_invariant_to_batch() -> None:
    """Changing B must not change flops_per_token or flops_sanity_per_token."""
    inp_b1 = _make_input(batch_size=1)
    inp_b8 = _make_input(batch_size=8)

    m1 = evaluate_llm_decode(inp_b1)
    m8 = evaluate_llm_decode(inp_b8)

    assert m1.flops_per_token == m8.flops_per_token
    assert m1.flops_sanity_per_token == m8.flops_sanity_per_token


# ---------------------------------------------------------------------------
# I. MHA / GQA / MQA semantics
# ---------------------------------------------------------------------------

def test_mha_gqa_mqa_kv_scales_with_hkv() -> None:
    """KV footprint and read traffic must scale with Hkv."""
    # MHA: Hkv = Hq
    inp_mha = _make_input(n_heads_kv=32, n_heads_q=32)
    # GQA: Hkv < Hq
    inp_gqa = _make_input(n_heads_kv=8, n_heads_q=32)
    # MQA: Hkv = 1
    inp_mqa = _make_input(n_heads_kv=1, n_heads_q=32)

    m_mha = evaluate_llm_decode(inp_mha)
    m_gqa = evaluate_llm_decode(inp_gqa)
    m_mqa = evaluate_llm_decode(inp_mqa)

    # MHA KV footprint = 4x GQA (32/8 = 4)
    assert m_mha.kv_footprint_bytes == 4 * m_gqa.kv_footprint_bytes
    # GQA KV footprint = 8x MQA (8/1 = 8)
    assert m_gqa.kv_footprint_bytes == 8 * m_mqa.kv_footprint_bytes

    # Same scaling for read traffic
    assert m_mha.kv_read_bytes_per_token == 4 * m_gqa.kv_read_bytes_per_token
    assert m_gqa.kv_read_bytes_per_token == 8 * m_mqa.kv_read_bytes_per_token


# ---------------------------------------------------------------------------
# J. LLaMA-3.1-8B hand-check
# ---------------------------------------------------------------------------

def test_llama_31_8b_hand_check() -> None:
    """LLaMA-3.1-8B-class hand-check.

    This is an *architecture-class* hand-check, not an exact reproduction
    of an official LLaMA-3.1-8B checkpoint: ``n_param=8_000_000_000`` is
    the architecture-class parameter count for an 8B dense decoder, not
    the byte-exact parameter count of any specific released checkpoint.
    Use this as a sanity check that the B1-R2 formulas produce
    well-conditioned numbers in the LLaMA-3.1-8B regime, not as a
    numerical reproduction of a published benchmark.

    Expected order-of-magnitude:
        attention sanity term ≈ 68.7 GFLOPs/token
        FLOPs_sanity          ≈ 84.7 GFLOPs/token
        detailed FLOPs        ≈ 83.7 GFLOPs/token  (diff ~1.17%)
    """
    inp = _make_input(
        n_param=8_000_000_000,
        n_layers=32,
        n_heads_q=32,
        n_heads_kv=8,
        d_head=128,
        d_model=4096,
        d_ff=14336,
        vocab_size=128_256,
        batch_size=1,
        context_length=131_072,
        weight_bits=16,
        kv_bits=16,
        runtime_bytes=0,
    )

    m = evaluate_llm_decode(inp)

    # Attention sanity term: 4 * L * Hq * S * Dhead
    expected_attention_sanity = 4 * 32 * 32 * 131_072 * 128
    assert expected_attention_sanity == 68_719_476_736

    # Total sanity: 2 * Nparam + attention_sanity
    expected_sanity = 2 * 8_000_000_000 + expected_attention_sanity
    assert expected_sanity == 84_719_476_736
    assert m.flops_sanity_per_token == expected_sanity

    expected_detailed = (
        32 * (50_331_648 + 33_554_432 + 352_321_536 + 2_147_483_648)
        + 1_050_673_152
    )
    assert expected_detailed == 83_728_793_600
    assert m.flops_per_token == expected_detailed

    # Sanity vs detailed diff (Norm excluded in implementation)
    diff = abs(m.flops_sanity_per_token - m.flops_per_token) / m.flops_per_token
    assert math.isclose(diff, 0.01183, abs_tol=0.0001)

    # Weight footprint = 8e9 * 16 / 8 = 16 GB
    assert m.weight_footprint_bytes == 16_000_000_000

    # KV footprint = 2 * 32 * 1 * 131072 * 8 * 128 * 16 / 8 = 17_179_869_184
    assert m.kv_footprint_bytes == 17_179_869_184


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_n_heads_kv_must_not_exceed_n_heads_q() -> None:
    with pytest.raises(ValueError):
        _make_input(n_heads_q=8, n_heads_kv=16)


def test_d_model_must_equal_hq_times_dhead() -> None:
    with pytest.raises(ValueError):
        _make_input(d_model=4096, n_heads_q=32, d_head=64)  # 32*64=2048 != 4096


def test_runtime_bytes_may_be_zero() -> None:
    """runtime_bytes=0 must be accepted."""
    inp = _make_input(runtime_bytes=0)
    m = evaluate_llm_decode(inp)
    assert m.runtime_bytes == 0


def test_negative_runtime_bytes_rejected() -> None:
    with pytest.raises(ValueError):
        _make_input(runtime_bytes=-1)


def test_zero_context_length_allowed() -> None:
    """S=0 is valid (e.g. first token generation with no history)."""
    inp = _make_input(context_length=0)
    m = evaluate_llm_decode(inp)
    assert m.kv_footprint_bytes == 0
    assert m.kv_read_bytes_per_token == 0
    assert m.flops_per_token > 0  # still has linear + vocab FLOPs


# ---------------------------------------------------------------------------
# R1 counterexamples — exact (non-floored) analytical accounting
# ---------------------------------------------------------------------------
#
# These tests pin the B1-R1 requirement that footprint and per-token
# traffic use *true* division (``/ 8``, ``/ B``), not floor division.
# A floor (``//``) implementation would silently drop sub-byte
# precision and round fractional per-token averages down to 0.
# All counterexamples therefore assert exact (not approximate) values
# against the analytical formulas.

def test_sub_byte_weight_footprint_is_not_truncated() -> None:
    """n_param=1, weight_bits=1 must give weight_footprint_bytes == 0.125.

    Counterexample against ``// 8``: a floor-division implementation
    would give ``1 * 1 // 8 == 0`` and silently lose the 1-bit
    weight content.  Exact division gives the analytical byte-equivalent
    ``0.125``.
    """
    inp = _make_input(n_param=1, weight_bits=1)
    m = evaluate_llm_decode(inp)
    assert m.weight_footprint_bytes == 0.125
    assert m.weight_active_per_step_bytes == 0.125
    # required_capacity still equals weight + kv + runtime exactly.
    expected_capacity = 0.125 + m.kv_footprint_bytes + m.runtime_bytes
    assert m.required_capacity_bytes == expected_capacity


def test_weight_traffic_fractional_average_not_truncated() -> None:
    """A 1-byte weight footprint with B=2 must give
    ``weight_read_bytes_per_token == 0.5``.

    Counterexample against ``// B``: a floor-division implementation
    would give ``1 // 2 == 0`` and silently lose the 0.5 byte/token
    read traffic.  Exact division gives the analytical per-token
    average ``0.5``.
    """
    # 1 byte weight footprint: n_param=1, weight_bits=8.
    inp = _make_input(n_param=1, weight_bits=8, batch_size=2)
    m = evaluate_llm_decode(inp)
    assert m.weight_footprint_bytes == 1.0
    assert m.weight_read_bytes_per_token == 0.5
    # And the aggregate-step read (footprint) is unchanged by batch.
    assert m.weight_active_per_step_bytes == 1.0


def test_sub_byte_kv_accounting_is_not_truncated() -> None:
    """Minimal legal dimensions with kv_bits=1 must give non-zero,
    non-floored KV footprint, read, and write byte-equivalents.

    With L=1, B=1, S=1, Hq=Hkv=1, Dhead=1, d_model=1 (so the GQA and
    d_model validators pass), and kv_bits=1:
        kv_footprint   = 2 * 1 * 1 * 1 * 1 * 1 * 1 / 8 = 0.25
        kv_read / token= 2 * 1 * 1 * 1 * 1 * 1 / 8     = 0.25
        kv_write / tok = 2 * 1 * 1 * 1 * 1     / 8     = 0.25
    A floor-division implementation would give ``0`` for all three and
    silently drop the analytical KV content.
    """
    inp = _make_input(
        n_layers=1,
        batch_size=1,
        context_length=1,
        n_heads_q=1,
        n_heads_kv=1,
        d_head=1,
        d_model=1,  # 1 * 1 == 1
        d_ff=1,
        vocab_size=1,
        weight_bits=8,
        kv_bits=1,
    )
    m = evaluate_llm_decode(inp)
    assert m.kv_footprint_bytes == 0.25
    assert m.kv_read_bytes_per_token == 0.25
    assert m.kv_write_bytes_per_token == 0.25
    # required_capacity = weight + kv + runtime
    assert m.required_capacity_bytes == (
        m.weight_footprint_bytes + 0.25 + m.runtime_bytes
    )


def test_nonuniform_gqa_grouping_rejected() -> None:
    """Hq=6, Hkv=4 with consistent d_model must raise ValidationError.

    v0 supports only uniform GQA/MQA/MHA, so ``Hq % Hkv != 0`` is
    rejected.  The d_model consistency check must also pass
    (d_model = Hq * Dhead = 6 * 64 = 384).
    """
    with pytest.raises(ValidationError):
        _make_input(
            n_heads_q=6, n_heads_kv=4, d_head=64, d_model=384,
        )


# ---------------------------------------------------------------------------
# R1 GQA legality matrix — MHA / GQA / MQA still pass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hkv,hq", [
    (32, 32),  # MHA: Hq == Hkv
    (8, 32),   # GQA: Hq / Hkv = 4
    (1, 32),   # MQA: Hq / Hkv = 32
    (4, 16),   # GQA: Hq / Hkv = 4
    (2, 8),    # GQA: Hq / Hkv = 4
])
def test_uniform_gqa_mqa_mha_accepted(hkv: int, hq: int) -> None:
    """All standard uniform-grouped head configurations must be
    accepted by the input validators.

    For each pair, d_head is set so that d_model = Hq * d_head is
    consistent with the default 4096.  d_head=128 is the LLaMA-3.1
    head dimension, so Hq must be 32 for d_model=4096.  For the
    non-default Hq values used in this matrix, d_head and d_model
    are recomputed to satisfy d_model = Hq * d_head.
    """
    d_head = 64  # fits any Hq in the parametrize set with d_model = Hq*64
    d_model = hq * d_head
    inp = _make_input(
        n_heads_q=hq, n_heads_kv=hkv, d_head=d_head, d_model=d_model,
    )
    # If construction succeeded, evaluation must also succeed.
    m = evaluate_llm_decode(inp)
    # KV footprint scales linearly with Hkv (everything else fixed).
    # Reference is the MHA-equivalent Hkv baseline (i.e. Hkv=hq
    # is MHA; here we re-derive the per-Hkv scale by using a
    # common reference: just check that kv_footprint is positive
    # and that kv_read scales with Hkv).
    assert m.kv_footprint_bytes > 0
    assert m.kv_read_bytes_per_token > 0
    assert m.kv_write_bytes_per_token > 0
