import os
import sys

import torch

# Try to import sgl_kernel
try:
    import sgl_kernel

    print("[INFO] sgl_kernel imported successfully.")
except ImportError as e:
    print(f"[ERROR] Failed to import sgl_kernel: {e}")
    sys.exit(1)


def test_vlen_bindings():
    """Test basic C++ bindings and VLEN value consistency."""
    print("\n=== 1. Testing C++ VLEN Bindings (Basic) ===")

    if not hasattr(torch.ops.sgl_kernel, "get_rvv_vlenb"):
        print("[ERROR] get_rvv_vlenb not found in torch.ops.sgl_kernel")
        return None

    try:
        # 1. Get VLEN
        vlenb = torch.ops.sgl_kernel.get_rvv_vlenb()  # Bytes
        vlen_bits = torch.ops.sgl_kernel.get_rvv_vlen()  # Bits

        print(f"✅ Detected VLEN: {vlen_bits} bits ({vlenb} bytes)")

        if vlenb == 0:
            print("[WARNING] VLEN is 0. Are you running on an RVV-capable device?")
            return 0

        # Consistency check
        if vlen_bits != vlenb * 8:
            print(f"❌ Consistency Error: {vlen_bits} bits != {vlenb} * 8")
        else:
            print("✅ VLEN bits and bytes are consistent.")

        return vlenb

    except Exception as e:
        print(f"[ERROR] An error occurred during binding test: {e}")
        return 0


def test_alignment_cases(vlenb):
    """Test various byte sizes to verify alignment logic."""
    if vlenb == 0:
        return

    print(f"\n=== 2. Extensive Alignment Logic Testing (VLEN={vlenb} bytes) ===")

    # Test cases: (Size in bytes, Description, Expected Result)
    test_cases = [
        (vlenb, "Exact VLEN", True),
        (vlenb * 2, "2x VLEN", True),
        (vlenb * 10, "10x VLEN", True),
        (0, "Zero bytes", True),  # Usually 0 is considered aligned (empty)
        (vlenb // 2, "Half VLEN", False),
        (1, "1 Byte", False if vlenb > 1 else True),
        (vlenb + 1, "VLEN + 1 byte", False),
        (vlenb - 1, "VLEN - 1 byte", False),
        (vlenb * 2 + 1, "2x VLEN + 1 byte", False),
        (128, "128 bytes (Common Vector Size)", 128 % vlenb == 0),
        (256, "256 bytes", 256 % vlenb == 0),
        (16, "16 bytes", 16 % vlenb == 0),
    ]

    print(f"{'Result':<10} | {'Size':<6} | {'Status':<12} | {'Description'}")
    print("-" * 60)

    for size, desc, expected in test_cases:
        # Skip invalid test cases (e.g. half vlen is 0)
        if size == 0 and desc == "Half VLEN":
            continue

        is_aligned = torch.ops.sgl_kernel.check_vlen_alignment(size)

        status = "✅ PASS" if is_aligned == expected else "❌ FAIL"
        match_str = "Aligned" if is_aligned else "Unaligned"
        expected_str = "Aligned" if expected else "Unaligned"

        # If passed, just show status. If failed, show expectation.
        if is_aligned != expected:
            match_str += f" (Exp: {expected_str})"

        print(f"{status:<10} | {size:<6} | {match_str:<12} | {desc}")


def test_backend_logic_extended(vlenb):
    """Simulate RVVAttnBackend checks for various Dtypes and Head Dimensions."""
    if vlenb == 0:
        return

    print("\n=== 3. Backend Logic Simulation (Head Dim & Dtypes) ===")

    # Simulate _check_head_dim_alignment from rvv_backend.py
    def check_head_dim_alignment(head_dim, dtype_size):
        size_bytes = head_dim * dtype_size
        return (size_bytes % vlenb) == 0

    # Common dtypes in LLMs
    dtypes = {"INT8": 1, "FP16": 2, "BF16": 2, "FP32": 4}

    # Head dimensions to test
    # 64: TinyLlama, Llama2-70B
    # 80: Uncommon
    # 96: Qwen-14B/72B
    # 128: Llama2-7B, Llama3-8B/70B, Mistral-7B
    # 256: Very Large Models
    head_dims = [32, 64, 80, 96, 100, 128, 256]

    print(f"{'Head Dim':<8} | {'Dtype':<6} | {'Size(B)':<7} | {'Support':<10}")
    print("-" * 40)

    for hdim in head_dims:
        for dtype_name, size_per_elem in dtypes.items():
            # Skip BF16 if it's redundant with FP16 for this display
            if dtype_name == "BF16":
                continue

            total_size = hdim * size_per_elem
            is_supported = check_head_dim_alignment(hdim, size_per_elem)

            # Verify with C++ op check
            cpp_check = torch.ops.sgl_kernel.check_vlen_alignment(total_size)

            if is_supported != cpp_check:
                status_icon = "❌ ERR"  # Logic mismatch
            else:
                status_icon = "✅ OK" if is_supported else "⚠️ Pad"

            print(f"{hdim:<8} | {dtype_name:<6} | {total_size:<7} | {status_icon}")


if __name__ == "__main__":
    vlenb_val = test_vlen_bindings()
    if vlenb_val:
        test_alignment_cases(vlenb_val)
        test_backend_logic_extended(vlenb_val)
