#!/usr/bin/env python3
"""
Shared Server Launcher for RVV Backend

This script handles the complexities of launching SGLang on Banana Pi/RISC-V, including:
1. Monkey-patching transformers to avoid circular imports.
2. Verifying RVV custom ops.
3. Launching the actual SGLang server.
"""

import os
import sys
import warnings

# Suppress common warnings on RISC-V
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="sglang")


def check_stubs():
    """Verify stubs are installed in the environment"""
    try:
        import triton_stub
        import vllm_stub

        # torchvision stub is implicit via import check
    except ImportError as e:
        print(f"❌ Critical Error: Stubs not installed ({e}).")
        print("   Please run: python3 setup_stubs.py")
        sys.exit(1)


def apply_patches():
    """Apply necessary monkey-patches for RISC-V environment"""
    try:
        import transformers.generation

        # Try direct import from utils if package import fails
        try:
            from transformers.generation import GenerationMixin
        except ImportError:
            try:
                from transformers.generation.utils import GenerationMixin

                # Monkey-patch it back into generation package
                transformers.generation.GenerationMixin = GenerationMixin
                sys.modules["transformers.generation"].GenerationMixin = GenerationMixin
            except ImportError:
                print("❌ Could not find GenerationMixin in utils.")

        # Trigger AutoTokenizer
        from transformers import AutoTokenizer

        # Trigger AutoProcessor (mock if needed for multimodals on RISC-V)
        try:
            from transformers import AutoProcessor
        except ImportError:
            import transformers

            class MockProcessor:
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    return cls()

                @classmethod
                def register(cls, *args, **kwargs):
                    pass

                def __call__(self, *args, **kwargs):
                    return {}

            transformers.AutoProcessor = MockProcessor
            try:
                sys.modules["transformers"].AutoProcessor = MockProcessor
            except:
                pass

            # Patch other missing processors
            for proc_name in [
                "Qwen2_5_VLProcessor",
                "DeepseekVLV2Processor",
                "LlavaProcessor",
                "Idefics2Processor",
            ]:
                if not hasattr(transformers, proc_name):
                    setattr(transformers, proc_name, MockProcessor)

    except Exception:
        import traceback

        traceback.print_exc()


def verify_rvv_ops():
    """Verify that sgl-kernel ops are available"""
    try:
        import torch

        if hasattr(torch.ops, "sgl_kernel"):
            if hasattr(torch.ops.sgl_kernel, "decode_attention_cpu"):
                return True
        print(
            "⚠ RVV Kernel Warning: torch.ops.sgl_kernel or decode_attention_cpu NOT found."
        )
        return False
    except ImportError:
        return False


def main():
    # 1. Check Stubs
    check_stubs()

    # 2. Apply Patches
    apply_patches()

    # 3. Verify RVV
    verify_rvv_ops()

    # 4. Import SGLang and Run
    try:
        # Ensure sglang is in path
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        python_path = os.path.join(repo_root, "python")
        if os.path.exists(python_path) and python_path not in sys.path:
            sys.path.insert(0, python_path)

        from sglang.launch_server import run_server
        from sglang.srt.server_args import prepare_server_args
        from sglang.srt.utils import kill_process_tree
    except ImportError:
        print("❌ Could not import sglang. Ensure it is in PYTHONPATH.")
        sys.exit(1)

    # 5. Run Server
    # Pass all arguments except the script name
    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
