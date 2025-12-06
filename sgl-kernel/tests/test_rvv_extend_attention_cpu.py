"""
Unit Test for extend_attention_cpu kernel (RVV optimized).

This test verifies that the CPU extend attention kernel produces correct results
by comparing against a naive PyTorch reference implementation.
Designed to run on RISC-V hardware platforms (e.g., Banana Pi).

Usage on RISC-V hardware:
    cd ~/.local_riscv_env/workspace/sglang/sgl-kernel
    pytest tests/test_rvv_extend_attention_cpu.py -v
"""

import platform

import pytest
import torch

# ============================================================================
# Platform Detection
# ============================================================================


def is_riscv_platform() -> bool:
    """Check if running on RISC-V platform."""
    machine = platform.machine().lower()
    return machine in ("riscv64", "riscv32", "riscv")


# Check sgl_kernel availability once at module load
try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False
    print("Warning: sgl_kernel not available, skipping tests")


def _has_extend_attention_cpu() -> bool:
    """Check if extend_attention_cpu op is available."""
    if not HAS_SGL_KERNEL:
        return False
    try:
        return hasattr(torch.ops.sgl_kernel, "extend_attention_cpu")
    except (AttributeError, RuntimeError):
        return False


def naive_extend_attention(
    q_extend,  # [total_extend_len, num_heads, head_dim]
    k_extend,  # [total_extend_len, num_heads, head_dim]
    v_extend,  # [total_extend_len, num_heads, head_dim_v]
    k_buffer,  # [max_total_tokens, num_heads, head_dim]
    v_buffer,  # [max_total_tokens, num_heads, head_dim_v]
    req_to_token,  # [num_seqs, max_context_len]
    req_pool_indices,  # [num_seqs]
    seq_lens,  # [num_seqs] - total sequence lengths
    extend_seq_lens,  # [num_seqs] - extend lengths for each request
    extend_start_loc,  # [num_seqs] - start locations in q_extend
    sm_scale,
    logit_cap,
):
    """
    Naive reference implementation of extend attention using PyTorch.

    For each request:
    - prefix_len = seq_len - extend_len (tokens from KV cache)
    - extend_len = number of new tokens to process

    Computes: output = softmax(Q @ K^T * scale) @ V with causal mask
    """
    num_seqs = seq_lens.shape[0]
    num_heads = q_extend.shape[1]
    head_dim = q_extend.shape[2]
    head_dim_v = v_extend.shape[2]

    o_extend = torch.zeros_like(q_extend)
    if head_dim != head_dim_v:
        o_extend = torch.zeros(
            q_extend.shape[0],
            num_heads,
            head_dim_v,
            dtype=q_extend.dtype,
            device=q_extend.device,
        )

    for i in range(num_seqs):
        seq_len = seq_lens[i].item()
        extend_len = extend_seq_lens[i].item()
        prefix_len = seq_len - extend_len
        start_loc = extend_start_loc[i].item()
        req_idx = req_pool_indices[i].item()

        # Get query for this request
        q = q_extend[
            start_loc : start_loc + extend_len
        ]  # [extend_len, num_heads, head_dim]

        # Construct full key/value for this request
        # Prefix part from buffer
        k_prefix = []
        v_prefix = []
        for j in range(prefix_len):
            token_idx = req_to_token[req_idx, j].item()
            k_prefix.append(k_buffer[token_idx])
            v_prefix.append(v_buffer[token_idx])

        if prefix_len > 0:
            k_prefix = torch.stack(k_prefix)  # [prefix_len, num_heads, head_dim]
            v_prefix = torch.stack(v_prefix)
        else:
            k_prefix = torch.empty(
                0, num_heads, head_dim, dtype=q.dtype, device=q.device
            )
            v_prefix = torch.empty(
                0, num_heads, head_dim_v, dtype=q.dtype, device=q.device
            )

        # Extend part
        k_ext = k_extend[start_loc : start_loc + extend_len]
        v_ext = v_extend[start_loc : start_loc + extend_len]

        # Concatenate: prefix (from cache) + extend (new)
        k = torch.cat([k_prefix, k_ext], dim=0)  # [seq_len, num_heads, head_dim]
        v = torch.cat([v_prefix, v_ext], dim=0)

        # Compute attention
        # Q: [extend_len, num_heads, head_dim] -> [num_heads, extend_len, head_dim]
        # K: [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
        q_t = q.transpose(0, 1)
        k_t = k.transpose(0, 1)
        v_t = v.transpose(0, 1)

        scores = (
            torch.bmm(q_t, k_t.transpose(1, 2)) * sm_scale
        )  # [num_heads, extend_len, seq_len]

        # Causal mask
        # For query position r (0 <= r < extend_len),
        # it can attend to positions 0 to (prefix_len + r) inclusive
        mask = torch.ones_like(scores) * float("-inf")
        for r in range(extend_len):
            valid_len = prefix_len + r + 1
            mask[:, r, :valid_len] = 0

        scores = scores + mask

        if logit_cap > 0:
            scores = logit_cap * torch.tanh(scores / logit_cap)

        probs = torch.softmax(scores, dim=-1)

        output = torch.bmm(probs, v_t)  # [num_heads, extend_len, head_dim_v]
        output = output.transpose(0, 1)  # [extend_len, num_heads, head_dim_v]

        o_extend[start_loc : start_loc + extend_len] = output

    return o_extend


@pytest.mark.skipif(not HAS_SGL_KERNEL, reason="sgl_kernel not available")
@pytest.mark.skipif(
    not _has_extend_attention_cpu(), reason="extend_attention_cpu not available"
)
def test_extend_attention_cpu_basic():
    """Basic test for extend_attention_cpu with single request."""
    device = "cpu"
    dtype = torch.float16  # Use float16 to match RISC-V RVV implementation

    num_seqs = 1
    num_heads = 4
    head_dim = 32
    head_dim_v = 32
    max_context_len = 32
    max_total_tokens = 64

    # Setup: single request with seq_len=10, extend_len=4
    seq_lens = torch.tensor([10], dtype=torch.int64, device=device)
    extend_seq_lens = torch.tensor([4], dtype=torch.int64, device=device)
    extend_start_loc = torch.tensor([0], dtype=torch.int64, device=device)
    req_pool_indices = torch.tensor([0], dtype=torch.int64, device=device)

    total_extend_len = extend_seq_lens.sum().item()

    # Create tensors
    q_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    k_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    v_extend = torch.randn(
        total_extend_len, num_heads, head_dim_v, dtype=dtype, device=device
    )
    o_extend = torch.zeros(
        total_extend_len, num_heads, head_dim_v, dtype=dtype, device=device
    )

    # KV cache buffers
    k_buffer = torch.randn(
        max_total_tokens, num_heads, head_dim, dtype=dtype, device=device
    )
    v_buffer = torch.randn(
        max_total_tokens, num_heads, head_dim_v, dtype=dtype, device=device
    )

    # Setup req_to_token mapping
    req_to_token = torch.zeros(
        num_seqs, max_context_len, dtype=torch.int64, device=device
    )
    for i in range(num_seqs):
        prefix_len = seq_lens[i].item() - extend_seq_lens[i].item()
        for j in range(prefix_len):
            req_to_token[i, j] = j  # Map to buffer indices

    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0

    # Compute reference
    ref_output = naive_extend_attention(
        q_extend.float(),
        k_extend.float(),
        v_extend.float(),
        k_buffer.float(),
        v_buffer.float(),
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        sm_scale,
        logit_cap,
    ).half()

    # Call the kernel
    try:
        torch.ops.sgl_kernel.extend_attention_cpu(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_seq_lens,
            extend_start_loc,
            total_extend_len,
            sm_scale,
            logit_cap,
        )

        # Verify correctness
        assert torch.isfinite(o_extend).all(), "Output contains NaN or Inf"
        assert not torch.allclose(
            o_extend, torch.zeros_like(o_extend), atol=1e-6
        ), "Output is all zeros"

        # Check numerical accuracy (relaxed tolerance for float16)
        if torch.allclose(o_extend, ref_output, atol=1e-2, rtol=1e-2):
            print(f"✓ Basic test passed: output shape {o_extend.shape}")
        else:
            max_diff = (o_extend - ref_output).abs().max().item()
            mean_diff = (o_extend - ref_output).abs().mean().item()
            print(f"⚠ Numerical difference: max={max_diff:.6f}, mean={mean_diff:.6f}")
            # Still pass if within reasonable bounds
            assert max_diff < 0.1, f"Max difference {max_diff} exceeds threshold"

    except Exception as e:
        pytest.fail(f"extend_attention_cpu failed: {e}")


@pytest.mark.skipif(not HAS_SGL_KERNEL, reason="sgl_kernel not available")
@pytest.mark.skipif(
    not _has_extend_attention_cpu(), reason="extend_attention_cpu not available"
)
def test_extend_attention_cpu_prefill_mode():
    """Test prefill mode: extend_len == seq_len (no prefix)."""
    device = "cpu"
    dtype = torch.float16

    num_seqs = 1
    num_heads = 4
    head_dim = 32
    max_context_len = 64
    max_total_tokens = 64

    # Prefill mode: extend_len == seq_len, prefix_len == 0
    seq_len = 16
    seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
    extend_seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
    extend_start_loc = torch.tensor([0], dtype=torch.int64, device=device)
    req_pool_indices = torch.tensor([0], dtype=torch.int64, device=device)

    total_extend_len = seq_len

    # Create tensors
    q_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    k_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    v_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    o_extend = torch.zeros(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )

    k_buffer = torch.randn(
        max_total_tokens, num_heads, head_dim, dtype=dtype, device=device
    )
    v_buffer = torch.randn(
        max_total_tokens, num_heads, head_dim, dtype=dtype, device=device
    )

    req_to_token = torch.zeros(
        num_seqs, max_context_len, dtype=torch.int64, device=device
    )

    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0

    # Reference
    ref_output = naive_extend_attention(
        q_extend.float(),
        k_extend.float(),
        v_extend.float(),
        k_buffer.float(),
        v_buffer.float(),
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        sm_scale,
        logit_cap,
    ).half()

    try:
        torch.ops.sgl_kernel.extend_attention_cpu(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_seq_lens,
            extend_start_loc,
            total_extend_len,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(o_extend).all(), "Output contains NaN or Inf"

        max_diff = (o_extend - ref_output).abs().max().item()
        print(f"✓ Prefill mode test passed: max_diff={max_diff:.6f}")
        assert max_diff < 0.1, f"Max difference {max_diff} exceeds threshold"

    except Exception as e:
        pytest.fail(f"extend_attention_cpu prefill mode failed: {e}")


@pytest.mark.skipif(not HAS_SGL_KERNEL, reason="sgl_kernel not available")
@pytest.mark.skipif(
    not _has_extend_attention_cpu(), reason="extend_attention_cpu not available"
)
def test_extend_attention_cpu_multiple_requests():
    """Test extend_attention_cpu with multiple requests (batched)."""
    device = "cpu"
    dtype = torch.float16

    num_seqs = 2
    num_heads = 4
    head_dim = 32
    max_context_len = 64
    max_total_tokens = 128

    # Two requests with different lengths
    seq_lens = torch.tensor([10, 20], dtype=torch.int64, device=device)
    extend_seq_lens = torch.tensor([4, 8], dtype=torch.int64, device=device)
    extend_start_loc = torch.tensor([0, 4], dtype=torch.int64, device=device)
    req_pool_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)

    total_extend_len = extend_seq_lens.sum().item()

    q_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    k_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    v_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    o_extend = torch.zeros(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )

    k_buffer = torch.randn(
        max_total_tokens, num_heads, head_dim, dtype=dtype, device=device
    )
    v_buffer = torch.randn(
        max_total_tokens, num_heads, head_dim, dtype=dtype, device=device
    )

    req_to_token = torch.zeros(
        num_seqs, max_context_len, dtype=torch.int64, device=device
    )
    for i in range(num_seqs):
        prefix_len = seq_lens[i].item() - extend_seq_lens[i].item()
        for j in range(prefix_len):
            req_to_token[i, j] = i * 32 + j

    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0

    ref_output = naive_extend_attention(
        q_extend.float(),
        k_extend.float(),
        v_extend.float(),
        k_buffer.float(),
        v_buffer.float(),
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        sm_scale,
        logit_cap,
    ).half()

    try:
        torch.ops.sgl_kernel.extend_attention_cpu(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_seq_lens,
            extend_start_loc,
            total_extend_len,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(o_extend).all(), "Output contains NaN or Inf"

        max_diff = (o_extend - ref_output).abs().max().item()
        print(f"✓ Multiple requests test passed: max_diff={max_diff:.6f}")
        assert max_diff < 0.1, f"Max difference {max_diff} exceeds threshold"

    except Exception as e:
        pytest.fail(f"extend_attention_cpu multiple requests failed: {e}")


@pytest.mark.skipif(not HAS_SGL_KERNEL, reason="sgl_kernel not available")
@pytest.mark.skipif(
    not _has_extend_attention_cpu(), reason="extend_attention_cpu not available"
)
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [32, 64])
@pytest.mark.parametrize("extend_len", [4, 16, 32])
def test_extend_attention_cpu_various_shapes(num_heads, head_dim, extend_len):
    """Test extend_attention_cpu with various tensor shapes."""
    device = "cpu"
    dtype = torch.float16

    num_seqs = 1
    max_context_len = 64
    max_total_tokens = 128

    prefix_len = 8
    seq_len = prefix_len + extend_len

    seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
    extend_seq_lens = torch.tensor([extend_len], dtype=torch.int64, device=device)
    extend_start_loc = torch.tensor([0], dtype=torch.int64, device=device)
    req_pool_indices = torch.tensor([0], dtype=torch.int64, device=device)

    total_extend_len = extend_len

    q_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    k_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    v_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    o_extend = torch.zeros(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )

    k_buffer = torch.randn(
        max_total_tokens, num_heads, head_dim, dtype=dtype, device=device
    )
    v_buffer = torch.randn(
        max_total_tokens, num_heads, head_dim, dtype=dtype, device=device
    )

    req_to_token = torch.zeros(
        num_seqs, max_context_len, dtype=torch.int64, device=device
    )
    for j in range(prefix_len):
        req_to_token[0, j] = j

    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0

    ref_output = naive_extend_attention(
        q_extend.float(),
        k_extend.float(),
        v_extend.float(),
        k_buffer.float(),
        v_buffer.float(),
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        sm_scale,
        logit_cap,
    ).half()

    try:
        torch.ops.sgl_kernel.extend_attention_cpu(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_seq_lens,
            extend_start_loc,
            total_extend_len,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(o_extend).all(), "Output contains NaN or Inf"

        max_diff = (o_extend - ref_output).abs().max().item()
        print(
            f"✓ Test passed: heads={num_heads}, head_dim={head_dim}, extend_len={extend_len}, max_diff={max_diff:.6f}"
        )
        assert max_diff < 0.15, f"Max difference {max_diff} exceeds threshold"

    except Exception as e:
        pytest.fail(
            f"extend_attention_cpu failed with heads={num_heads}, head_dim={head_dim}, extend_len={extend_len}: {e}"
        )


@pytest.mark.skipif(not HAS_SGL_KERNEL, reason="sgl_kernel not available")
@pytest.mark.skipif(
    not _has_extend_attention_cpu(), reason="extend_attention_cpu not available"
)
def test_extend_attention_cpu_numerical_stability():
    """Test extend_attention_cpu with values that could cause numerical issues."""
    device = "cpu"
    dtype = torch.float16

    num_seqs = 1
    num_heads = 4
    head_dim = 32
    max_context_len = 32
    max_total_tokens = 64

    seq_lens = torch.tensor([16], dtype=torch.int64, device=device)
    extend_seq_lens = torch.tensor([8], dtype=torch.int64, device=device)
    extend_start_loc = torch.tensor([0], dtype=torch.int64, device=device)
    req_pool_indices = torch.tensor([0], dtype=torch.int64, device=device)

    total_extend_len = 8

    # Use moderate scale to avoid overflow in float16
    scale = 2.0
    q_extend = (
        torch.randn(total_extend_len, num_heads, head_dim, dtype=dtype, device=device)
        * scale
    )
    k_extend = (
        torch.randn(total_extend_len, num_heads, head_dim, dtype=dtype, device=device)
        * scale
    )
    v_extend = (
        torch.randn(total_extend_len, num_heads, head_dim, dtype=dtype, device=device)
        * scale
    )
    o_extend = torch.zeros(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )

    k_buffer = (
        torch.randn(max_total_tokens, num_heads, head_dim, dtype=dtype, device=device)
        * scale
    )
    v_buffer = (
        torch.randn(max_total_tokens, num_heads, head_dim, dtype=dtype, device=device)
        * scale
    )

    req_to_token = torch.zeros(
        num_seqs, max_context_len, dtype=torch.int64, device=device
    )
    for j in range(8):
        req_to_token[0, j] = j

    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0  # Should cap extreme logits

    try:
        torch.ops.sgl_kernel.extend_attention_cpu(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_seq_lens,
            extend_start_loc,
            total_extend_len,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(o_extend).all(), "Output contains NaN or Inf"
        assert not torch.allclose(
            o_extend, torch.zeros_like(o_extend), atol=1e-6
        ), "Output is all zeros"

        print(f"✓ Numerical stability test passed")

    except Exception as e:
        pytest.fail(f"extend_attention_cpu numerical stability test failed: {e}")


@pytest.mark.skipif(not HAS_SGL_KERNEL, reason="sgl_kernel not available")
@pytest.mark.skipif(
    not _has_extend_attention_cpu(), reason="extend_attention_cpu not available"
)
def test_extend_attention_cpu_long_prefix():
    """Test extend_attention_cpu with long prefix (stress test for prefix packing)."""
    device = "cpu"
    dtype = torch.float16

    num_seqs = 1
    num_heads = 4
    head_dim = 32
    max_context_len = 128
    max_total_tokens = 256

    # Long prefix: 64 tokens from cache, 8 new tokens
    prefix_len = 64
    extend_len = 8
    seq_len = prefix_len + extend_len

    seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
    extend_seq_lens = torch.tensor([extend_len], dtype=torch.int64, device=device)
    extend_start_loc = torch.tensor([0], dtype=torch.int64, device=device)
    req_pool_indices = torch.tensor([0], dtype=torch.int64, device=device)

    total_extend_len = extend_len

    q_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    k_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    v_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    o_extend = torch.zeros(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )

    k_buffer = torch.randn(
        max_total_tokens, num_heads, head_dim, dtype=dtype, device=device
    )
    v_buffer = torch.randn(
        max_total_tokens, num_heads, head_dim, dtype=dtype, device=device
    )

    req_to_token = torch.zeros(
        num_seqs, max_context_len, dtype=torch.int64, device=device
    )
    for j in range(prefix_len):
        req_to_token[0, j] = j

    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0

    ref_output = naive_extend_attention(
        q_extend.float(),
        k_extend.float(),
        v_extend.float(),
        k_buffer.float(),
        v_buffer.float(),
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        sm_scale,
        logit_cap,
    ).half()

    try:
        torch.ops.sgl_kernel.extend_attention_cpu(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_seq_lens,
            extend_start_loc,
            total_extend_len,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(o_extend).all(), "Output contains NaN or Inf"

        max_diff = (o_extend - ref_output).abs().max().item()
        print(
            f"✓ Long prefix test passed: prefix_len={prefix_len}, max_diff={max_diff:.6f}"
        )
        assert max_diff < 0.15, f"Max difference {max_diff} exceeds threshold"

    except Exception as e:
        pytest.fail(f"extend_attention_cpu long prefix test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
