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


def test_intel_gpu_device_indexing():
    """Test Intel GPU device indexing behavior and compatibility."""
    print("=== Intel GPU Device Indexing Test ===")
    
    if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        print("Intel GPU not available, skipping indexing test.")
        print()
        return
    
    print("Testing different XPU device creation methods...")
    
    # Test different ways of creating XPU devices
    device_variants = [
        ("torch.device('xpu')", torch.device('xpu')),
        ("torch.device('xpu:0')", torch.device('xpu:0')),
        ("torch.device('xpu', 0)", torch.device('xpu', 0)),
    ]
    
    for desc, device in device_variants:
        print(f"  {desc} → {device}")
        
        # Create a tensor and check its device
        tensor = torch.tensor([1.0], device=device)
        print(f"    Tensor device: {tensor.device}")
        
        # Check if they refer to the same device
        same_device = (
            tensor.device.type == device.type and 
            (tensor.device.index or 0) == (device.index or 0)
        )
        print(f"    Same physical device: {same_device}")
    
    # Test cross-device operations
    print("\n  Testing cross-device operations...")
    try:
        x = torch.tensor([1.0], device='xpu')      # No index
        y = torch.tensor([2.0], device='xpu:0')    # With index
        z = x + y  # Should work since both are on same physical device
        print(f"    Cross-device add: {x.device} + {y.device} → {z.device}")
        print(f"    Result: {z.item()}")
        print("    ✓ Cross-device operations work correctly")
    except Exception as e:
        print(f"    ✗ Cross-device operation failed: {e}")
    
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
            # For XPU, explicitly use index 0 to avoid confusion
            if device_name == "xpu":
                device = torch.device("xpu:0")
            else:
                device = torch.device(device_name)
            
            # Create tensors
            a = torch.randn(100, 100, device=device)
            b = torch.randn(100, 100, device=device)
            
            # Basic operations
            c = torch.matmul(a, b)
            
            # Fix: Compare device types and indices correctly for XPU
            if device_name == "xpu":
                device_match = (
                    c.device.type == device.type and 
                    (c.device.index or 0) == (device.index or 0)
                )
                print(f"    Input device: {device}")
                print(f"    Output device: {c.device}")
                assert device_match, f"Device mismatch: {device} vs {c.device}"
            else:
                assert c.device == device
            
            assert c.shape == (100, 100)
            
            # Test autocast if supported
            if device_name in ["cuda", "xpu"]:
                try:
                    with torch.autocast(device_type=device_name, dtype=torch.float16):
                        d = torch.matmul(a, b)
                        # Note: autocast may not change dtype for all operations
                        print(f"    Autocast test passed (output dtype: {d.dtype})")
                except Exception as autocast_error:
                    print(f"    Autocast limited support: {autocast_error}")
            
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


def test_device_equality():
    """Test device equality and comparison functions."""
    print("=== Device Equality Test ===")
    
    if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        print("Intel GPU not available, skipping device equality test.")
        print()
        return
    
    print("Testing device equality comparisons...")
    
    # Test different device object representations
    device_a = torch.device('xpu')
    device_b = torch.device('xpu:0')
    device_c = torch.device('xpu', 0)
    
    print(f"device_a = torch.device('xpu') → {device_a}")
    print(f"device_b = torch.device('xpu:0') → {device_b}")
    print(f"device_c = torch.device('xpu', 0) → {device_c}")
    
    # Test standard equality
    print(f"\nStandard equality tests:")
    print(f"device_a == device_b: {device_a == device_b}")
    print(f"device_b == device_c: {device_b == device_c}")
    print(f"device_a == device_c: {device_a == device_c}")
    
    # Test our custom equality function
    def is_same_device(dev1, dev2):
        """Check if two devices refer to the same physical device."""
        if dev1.type != dev2.type:
            return False
        if dev1.type == "xpu":
            # For XPU, treat None index as 0
            idx1 = dev1.index if dev1.index is not None else 0
            idx2 = dev2.index if dev2.index is not None else 0
            return idx1 == idx2
        return dev1 == dev2
    
    print(f"\nCustom equality tests (same physical device):")
    print(f"is_same_device(device_a, device_b): {is_same_device(device_a, device_b)}")
    print(f"is_same_device(device_b, device_c): {is_same_device(device_b, device_c)}")
    print(f"is_same_device(device_a, device_c): {is_same_device(device_a, device_c)}")
    
    # Test with actual tensors
    print(f"\nTensor device tests:")
    tensor_a = torch.tensor([1.0], device=device_a)
    tensor_b = torch.tensor([2.0], device=device_b)
    
    print(f"tensor_a.device: {tensor_a.device}")
    print(f"tensor_b.device: {tensor_b.device}")
    print(f"Same device: {is_same_device(tensor_a.device, tensor_b.device)}")
    
    # Test cross-tensor operations
    try:
        result = tensor_a + tensor_b
        print(f"Cross-tensor operation successful: {result.device}")
    except Exception as e:
        print(f"Cross-tensor operation failed: {e}")
    
    print()


def main():
    """Run all tests."""
    print("LeRobot Intel GPU Support Test")
    print("=" * 50)
    
    test_pytorch_basic()
    test_device_detection()
    test_intel_gpu_device_indexing()
    test_tensor_operations()
    test_memory_functions()
    test_device_equality()
    
    # Summary
    print("=== Summary ===")
    xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
    
    if xpu_available:
        print("✅ Intel GPU (XPU) support is AVAILABLE!")
        print("   Key points:")
        print("   - Use device='xpu' or device='xpu:0' (both work)")
        print("   - Intel GPU auto-assigns index 0 if not specified")
        print("   - Cross-device operations work between xpu and xpu:0")
        print("   - Example: lerobot-train --policy.device=xpu")
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
