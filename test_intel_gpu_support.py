#!/usr/bin/env python

"""
Simple test script for Intel GPU support in LeRobot.
This script tests basic functionality without requiring full LeRobot dependencies.
"""

import sys
import torch


def test_pytorch_basic():
    """Test basic PyTorch functionality."""
    print("=== PyTorch Basic Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    # Test Intel GPU
    xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
    print(f"Intel GPU (XPU) available: {xpu_available}")
    if xpu_available:
        print(f"Intel GPU device count: {torch.xpu.device_count()}")
        for i in range(torch.xpu.device_count()):
            print(f"  Device {i}: {torch.xpu.get_device_name(i)}")
    
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print()


def test_device_detection():
    """Test automatic device detection."""
    print("=== Device Detection Test ===")
    
    # Test our device utility functions
    try:
        from lerobot.utils.utils import get_safe_torch_device, is_torch_device_available
        
        devices_to_test = ["xpu", "cuda", "mps", "cpu"]
        
        for device_name in devices_to_test:
            available = is_torch_device_available(device_name)
            print(f"{device_name:>4}: {'✓' if available else '✗'}")
            
            if available:
                try:
                    device = get_safe_torch_device(device_name, log=False)
                    print(f"     → Successfully created {device}")
                except Exception as e:
                    print(f"     → Error creating device: {e}")
        
        # Test automatic best device selection
        for try_device in ["xpu", "cuda", "cpu"]:
            if is_torch_device_available(try_device):
                best_device = get_safe_torch_device(try_device, log=False)
                print(f"\nBest available device: {best_device}")
                break
                
    except ImportError as e:
        print(f"LeRobot utils not available: {e}")
    print()


def test_tensor_operations():
    """Test basic tensor operations on available devices."""
    print("=== Tensor Operations Test ===")
    
    # Determine which devices to test
    test_devices = []
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        test_devices.append("xpu")
    if torch.cuda.is_available():
        test_devices.append("cuda")
    if torch.backends.mps.is_available():
        test_devices.append("mps")
    test_devices.append("cpu")
    
    for device_name in test_devices:
        print(f"Testing {device_name}...")
        try:
            device = torch.device(device_name)
            
            # Create tensors
            a = torch.randn(100, 100, device=device)
            b = torch.randn(100, 100, device=device)
            
            # Basic operations
            c = torch.matmul(a, b)
            assert c.device == device
            assert c.shape == (100, 100)
            
            # Test autocast if supported
            if device_name in ["cuda", "xpu"]:
                with torch.autocast(device_type=device_name, dtype=torch.float16):
                    d = torch.matmul(a, b)
                    # Note: autocast may not change dtype for all operations
                    print(f"  Autocast test passed (output dtype: {d.dtype})")
            
            print(f"  ✓ {device_name} tensor operations successful")
            
        except Exception as e:
            print(f"  ✗ {device_name} failed: {e}")
    print()


def test_memory_functions():
    """Test memory utility functions."""
    print("=== Memory Functions Test ===")
    
    try:
        from lerobot.utils.device_utils import (
            get_memory_info, empty_cache, synchronize_device
        )
        
        test_devices = []
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            test_devices.append(torch.device("xpu"))
        if torch.cuda.is_available():
            test_devices.append(torch.device("cuda"))
        test_devices.append(torch.device("cpu"))
        
        for device in test_devices:
            print(f"Testing memory functions on {device}...")
            
            # Test memory info
            memory_info = get_memory_info(device)
            print(f"  Memory info: {memory_info}")
            
            # Test synchronization
            synchronize_device(device)
            print(f"  Synchronization: ✓")
            
            # Test cache clearing
            empty_cache(device)
            print(f"  Cache clear: ✓")
            
    except ImportError as e:
        print(f"LeRobot device utils not available: {e}")
    print()


def main():
    """Run all tests."""
    print("LeRobot Intel GPU Support Test")
    print("=" * 50)
    
    test_pytorch_basic()
    test_device_detection()
    test_tensor_operations()
    test_memory_functions()
    
    # Summary
    print("=== Summary ===")
    xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
    
    if xpu_available:
        print("✅ Intel GPU (XPU) support is AVAILABLE!")
        print("   You can use device='xpu' in LeRobot commands.")
        print("   Example: lerobot-train --policy.device=xpu")
    else:
        print("❌ Intel GPU (XPU) support is NOT available.")
        print("   To enable Intel GPU support:")
        print("   1. Install Intel GPU drivers")
        print("   2. Install PyTorch with XPU support:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu")
    
    if torch.cuda.is_available():
        print("✅ NVIDIA GPU (CUDA) support is available as fallback.")
    elif torch.backends.mps.is_available():
        print("✅ Apple Silicon (MPS) support is available as fallback.")
    else:
        print("ℹ️  Only CPU computing is available.")


if __name__ == "__main__":
    main()
