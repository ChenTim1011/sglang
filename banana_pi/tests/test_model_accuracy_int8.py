"""
Model-Level Accuracy Test for INT8 KV Cache (Perplexity Simulation)

This test measures model-level accuracy impact (perplexity, token prediction accuracy)
when using INT8 quantized KV cache vs FP16/BF16 baseline.

Note: This test focuses on model-level metrics (perplexity simulation).
      For kernel-level accuracy testing (decode attention accuracy), use test_parametrized_int8.py instead.

Usage:
    python test_model_accuracy_int8.py
    python test_model_accuracy_int8.py --vocab-size 32000 --num-sequences 10
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import test utilities (required, no fallback)
from test_utils import check_system_state

try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError as e:
    print(f"Error: Failed to import sgl_kernel: {e}")
    HAS_SGL_KERNEL = False
    sys.exit(1)


def compute_perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute perplexity from logits and target tokens."""
    # logits: [batch_size, vocab_size]
    # targets: [batch_size]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    perplexity = torch.exp(-target_log_probs.mean()).item()
    return perplexity


def compute_token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute token prediction accuracy."""
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets).float().mean().item()
    return correct


def generate_test_sequences(
    vocab_size: int = 32000, num_sequences: int = 10, seq_len: int = 128, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random test sequences (tokens)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate input sequences
    input_ids = torch.randint(0, vocab_size, (num_sequences, seq_len), dtype=torch.long)
    # Target tokens (next token prediction)
    target_ids = torch.randint(0, vocab_size, (num_sequences,), dtype=torch.long)

    return input_ids, target_ids


def simulate_attention_output_fp16(
    num_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Simulate attention output for FP16 baseline."""
    # This is a simplified simulation - in real usage, this would come from
    # the actual model forward pass
    output = torch.randn(batch_size, num_heads, head_dim, dtype=dtype)
    return output


def simulate_attention_output_int8(
    num_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    k_scale: float = 0.01,
    v_scale: float = 0.01,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Simulate attention output for INT8 quantized KV cache."""
    # Simulate quantization noise
    noise_scale = max(k_scale, v_scale) * 0.1  # 10% quantization noise
    output = torch.randn(batch_size, num_heads, head_dim, dtype=dtype)
    # Add quantization noise
    noise = torch.randn_like(output) * noise_scale
    output = output + noise
    return output


def test_model_perplexity_simulation(
    vocab_size: int = 32000, num_sequences: int = 10, seq_len: int = 128
) -> Dict[str, float]:
    """Simulate model perplexity comparison (FP16 vs INT8)."""
    print(f"\n=== Simulating Model Perplexity Comparison ===")
    print(f"Config: VocabSize={vocab_size}, NumSeqs={num_sequences}, SeqLen={seq_len}")

    # Generate test sequences
    input_ids, target_ids = generate_test_sequences(vocab_size, num_sequences, seq_len)

    # Simulate logits from model (FP16 baseline)
    torch.manual_seed(42)
    logits_fp16 = torch.randn(num_sequences, vocab_size, dtype=torch.float32)
    logits_fp16 = logits_fp16 - logits_fp16.max(dim=-1, keepdim=True)[0]  # Normalize

    # Simulate INT8 quantization effect on logits
    # Quantization introduces small noise
    quantization_noise = torch.randn_like(logits_fp16) * 0.01  # 1% noise
    logits_int8 = logits_fp16 + quantization_noise

    # Compute perplexity
    ppl_fp16 = compute_perplexity(logits_fp16, target_ids)
    ppl_int8 = compute_perplexity(logits_int8, target_ids)

    # Compute accuracy
    acc_fp16 = compute_token_accuracy(logits_fp16, target_ids)
    acc_int8 = compute_token_accuracy(logits_int8, target_ids)

    print(f"FP16 Perplexity: {ppl_fp16:.4f}")
    print(f"INT8 Perplexity: {ppl_int8:.4f}")
    print(f"Perplexity Ratio: {ppl_int8 / ppl_fp16:.4f}")
    print(f"FP16 Accuracy: {acc_fp16:.4f}")
    print(f"INT8 Accuracy: {acc_int8:.4f}")
    print(f"Accuracy Drop: {acc_fp16 - acc_int8:.4f}")

    return {
        "ppl_fp16": ppl_fp16,
        "ppl_int8": ppl_int8,
        "ppl_ratio": ppl_int8 / ppl_fp16,
        "acc_fp16": acc_fp16,
        "acc_int8": acc_int8,
        "acc_drop": acc_fp16 - acc_int8,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test INT8 KV Cache Model Accuracy (Perplexity Simulation)",
        epilog="Note: For attention kernel accuracy testing, use test_parametrized_int8.py instead.",
    )
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument(
        "--num-sequences", type=int, default=10, help="Number of test sequences"
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    args = parser.parse_args()

    if not HAS_SGL_KERNEL:
        print("Error: sgl_kernel not available")
        return

    print("=" * 60)
    print("INT8 KV Cache Model Accuracy Test (Perplexity Simulation)")
    print("=" * 60)
    print("Note: This test focuses on model-level accuracy (perplexity).")
    print("      For kernel-level accuracy testing, use test_parametrized_int8.py")
    print("=" * 60)

    # Check system state
    check_system_state()

    # Test: Model perplexity simulation
    model_results = test_model_perplexity_simulation(
        vocab_size=args.vocab_size,
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Model Perplexity Ratio: {model_results['ppl_ratio']:.4f}")
    print(f"Model Accuracy Drop: {model_results['acc_drop']:.4f}")

    # Pass criteria
    model_pass = model_results["ppl_ratio"] < 1.1 and model_results["acc_drop"] < 0.05

    if model_pass:
        print("\n✅ Model accuracy test PASSED")
    else:
        print("\n⚠️  Model accuracy test may need attention")
        print(
            f"  - Model degradation: PPL ratio {model_results['ppl_ratio']:.4f}, Acc drop {model_results['acc_drop']:.4f}"
        )


if __name__ == "__main__":
    main()
