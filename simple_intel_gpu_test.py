#!/usr/bin/env python

"""
Simple Intel GPU test for LeRobot
"""

try:
    import torch
    print("✅ PyTorch imported successfully")
    print(f"PyTorch version: {torch.__version__}")
    
    # Test XPU availability
    if hasattr(torch, 'xpu'):
        print("✅ XPU module available")
        if torch.xpu.is_available():
            print("✅ Intel GPU (XPU) is available!")
            print(f"XPU device count: {torch.xpu.device_count()}")
            for i in range(torch.xpu.device_count()):
                print(f"  Device {i}: {torch.xpu.get_device_name(i)}")
        else:
            print("❌ Intel GPU (XPU) is not available")
    else:
        print("❌ XPU module not available")
    
    # Test basic tensor operations
    print("\nTesting tensor operations...")
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
        x = torch.randn(10, 10, device=device)
        y = torch.randn(10, 10, device=device)
        z = torch.matmul(x, y)
        print(f"✅ Tensor operations on {device} successful")
        print(f"Result shape: {z.shape}, device: {z.device}")
    else:
        print("Skipping XPU tensor tests - XPU not available")
    
    # Test LeRobot utils
    print("\nTesting LeRobot utils...")
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from lerobot.utils.utils import get_safe_torch_device, is_torch_device_available
        
        print("✅ LeRobot utils imported successfully")
        
        # Test device availability
        devices = ["xpu", "cuda", "mps", "cpu"]
        for device_name in devices:
            available = is_torch_device_available(device_name)
            print(f"{device_name:>4}: {'✅' if available else '❌'}")
        
        # Test automatic device selection
        if is_torch_device_available("xpu"):
            device = get_safe_torch_device("xpu", log=True)
            print(f"✅ Intel GPU device created: {device}")
        
    except ImportError as e:
        print(f"❌ LeRobot utils import failed: {e}")
    except Exception as e:
        print(f"❌ LeRobot utils test failed: {e}")
    
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")
except Exception as e:
    print(f"❌ Test failed: {e}")

print("\n" + "="*50)
print("Intel GPU Support Test Complete")
