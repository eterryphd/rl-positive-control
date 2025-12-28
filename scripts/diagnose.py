#!/usr/bin/env python3
"""
Diagnostic script to check GRPO training setup.
Run this before training to catch common issues.

Usage:
    python diagnose.py
"""

import sys

def check_imports():
    """Check all required packages are importable."""
    print("=" * 60)
    print("PACKAGE IMPORT CHECK")
    print("=" * 60)
    
    packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('trl', 'trl'),
        ('accelerate', 'accelerate'),
        ('deepspeed', 'deepspeed'),
        ('datasets', 'datasets'),
    ]
    
    all_ok = True
    for name, module in packages:
        try:
            m = __import__(module)
            version = getattr(m, '__version__', 'unknown')
            print(f"  ✓ {name}: {version}")
        except ImportError as e:
            print(f"  ✗ {name}: FAILED - {e}")
            all_ok = False
    
    # Optional vLLM
    try:
        import vllm
        print(f"  ✓ vllm: {vllm.__version__}")
    except ImportError:
        print(f"  ⚠ vllm: not installed (optional)")
    
    return all_ok


def check_cuda():
    """Check CUDA and GPU availability."""
    print("\n" + "=" * 60)
    print("CUDA/GPU CHECK")
    print("=" * 60)
    
    import torch
    
    if not torch.cuda.is_available():
        print("  ✗ CUDA not available!")
        return False
    
    print(f"  ✓ CUDA available: {torch.version.cuda}")
    print(f"  ✓ GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1e9
        print(f"    GPU {i}: {props.name} ({mem_gb:.1f} GB)")
    
    total_mem = sum(
        torch.cuda.get_device_properties(i).total_memory 
        for i in range(torch.cuda.device_count())
    ) / 1e9
    print(f"  Total VRAM: {total_mem:.1f} GB")
    
    if torch.cuda.device_count() < 4:
        print(f"  ⚠ Warning: Expected 4 GPUs, found {torch.cuda.device_count()}")
    
    return True


def check_trl_grpo():
    """Check TRL GRPO API compatibility."""
    print("\n" + "=" * 60)
    print("TRL GRPO API CHECK")
    print("=" * 60)
    
    try:
        from trl import GRPOConfig, GRPOTrainer
        print("  ✓ GRPOConfig importable")
        print("  ✓ GRPOTrainer importable")
        
        # Check for key parameters
        import inspect
        sig = inspect.signature(GRPOConfig)
        params = list(sig.parameters.keys())
        
        expected_params = [
            'use_vllm', 'vllm_mode', 'beta', 'num_generations',
            'max_completion_length', 'gradient_checkpointing'
        ]
        
        for param in expected_params:
            if param in params:
                print(f"  ✓ GRPOConfig.{param} available")
            else:
                print(f"  ⚠ GRPOConfig.{param} NOT found (API may differ)")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ⚠ Warning: {e}")
        return True


def check_accelerate():
    """Check accelerate configuration."""
    print("\n" + "=" * 60)
    print("ACCELERATE CHECK")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        ['accelerate', 'env'], 
        capture_output=True, 
        text=True
    )
    
    if result.returncode == 0:
        print("  ✓ accelerate env works")
        # Print key lines
        for line in result.stdout.split('\n'):
            if any(k in line.lower() for k in ['distributed', 'gpu', 'cuda', 'deepspeed']):
                print(f"    {line.strip()}")
    else:
        print(f"  ✗ accelerate env failed: {result.stderr}")
        return False
    
    return True


def check_deepspeed():
    """Check DeepSpeed setup."""
    print("\n" + "=" * 60)
    print("DEEPSPEED CHECK")
    print("=" * 60)
    
    try:
        import deepspeed
        print(f"  ✓ DeepSpeed version: {deepspeed.__version__}")
        
        # Check ops
        from deepspeed.ops.op_builder import AllToAllBuilder, FusedAdamBuilder
        print("  ✓ DeepSpeed ops available")
        
        return True
    except Exception as e:
        print(f"  ⚠ Warning: {e}")
        return True


def check_model_access():
    """Check HuggingFace model access."""
    print("\n" + "=" * 60)
    print("MODEL ACCESS CHECK")
    print("=" * 60)
    
    from huggingface_hub import HfApi
    
    api = HfApi()
    
    try:
        user = api.whoami()
        print(f"  ✓ Logged in as: {user.get('name', 'unknown')}")
    except Exception:
        print("  ⚠ Not logged in to HuggingFace")
        print("    Run: huggingface-cli login")
    
    # Check model access
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    try:
        api.model_info(model_id)
        print(f"  ✓ Access to {model_id}")
    except Exception as e:
        print(f"  ✗ Cannot access {model_id}: {e}")
        print("    Make sure you've accepted the license at:")
        print(f"    https://huggingface.co/{model_id}")
        return False
    
    return True


def main():
    print("\n" + "=" * 60)
    print("GRPO TRAINING DIAGNOSTICS")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", check_imports()))
    results.append(("CUDA", check_cuda()))
    results.append(("TRL GRPO", check_trl_grpo()))
    results.append(("Accelerate", check_accelerate()))
    results.append(("DeepSpeed", check_deepspeed()))
    results.append(("Model Access", check_model_access()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All checks passed! Ready for training.")
    else:
        print("\n✗ Some checks failed. Fix issues above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()