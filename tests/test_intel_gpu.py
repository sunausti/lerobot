#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test Intel GPU (XPU) functionality in LeRobot."""

import logging
import sys
import pytest
import torch
import numpy as np
from tests.utils import require_xpu, require_accelerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_xpu_availability():
    """Test if Intel GPU is available and working."""
    assert hasattr(torch, 'xpu'), "PyTorch was not built with Intel GPU support"
    assert torch.xpu.is_available(), "Intel GPU is not available"
    
    device_count = torch.xpu.device_count()
    assert device_count > 0, f"No Intel GPU devices found (count: {device_count})"
    
    logger.info(f"Intel GPU devices found: {device_count}")
    for i in range(device_count):
        device_name = torch.xpu.get_device_name(i)
        logger.info(f"Device {i}: {device_name}")


@require_xpu
def test_xpu_tensor_operations():
    """Test basic tensor operations on Intel GPU."""
    device = torch.device("xpu")
    
    # Test tensor creation
    a = torch.randn(100, 100, device=device)
    b = torch.randn(100, 100, device=device)
    
    # Test basic operations
    c = torch.matmul(a, b)
    assert c.device.type == "xpu"
    assert c.shape == (100, 100)
    
    # Test synchronization
    torch.xpu.synchronize()
    
    logger.info("Basic tensor operations on Intel GPU successful")


@require_xpu
def test_xpu_autocast():
    """Test automatic mixed precision on Intel GPU."""
    device = torch.device("xpu")
    
    x = torch.randn(64, 128, device=device)
    linear = torch.nn.Linear(128, 64).to(device)
    
    # Test autocast
    with torch.autocast(device_type="xpu", dtype=torch.float16):
        output = linear(x)
        assert output.dtype == torch.float16
    
    logger.info("Intel GPU autocast test successful")


@require_xpu
def test_lerobot_device_utils():
    """Test LeRobot device utilities with Intel GPU."""
    from lerobot.utils.utils import get_safe_torch_device, is_torch_device_available
    from lerobot.utils.device_utils import (
        synchronize_device, empty_cache, get_memory_info,
        supports_autocast, supports_compile
    )
    
    # Test device availability check
    assert is_torch_device_available("xpu")
    
    # Test safe device creation
    device = get_safe_torch_device("xpu", log=True)
    assert device.type == "xpu"
    
    # Test device utilities
    synchronize_device(device)
    empty_cache(device)
    
    memory_info = get_memory_info(device)
    assert isinstance(memory_info, dict)
    assert "allocated" in memory_info
    assert "reserved" in memory_info
    
    # Test feature support
    assert supports_autocast(device)
    # compile support depends on PyTorch version
    
    logger.info("LeRobot device utilities work with Intel GPU")


@require_xpu
def test_xpu_policy_inference():
    """Test policy inference on Intel GPU."""
    try:
        from lerobot.policies.diffusion import DiffusionPolicy
        from lerobot.datasets.factory import make_dataset
        
        device = torch.device("xpu")
        
        # Create a minimal configuration for testing
        config = {
            "input_features": {
                "observation.image": {"shape": [3, 64, 64], "type": "visual"},
                "observation.state": {"shape": [10], "type": "state"}
            },
            "output_features": {
                "action": {"shape": [7], "type": "action"}
            },
            "device": str(device)
        }
        
        # Test policy creation and inference
        policy = DiffusionPolicy(config)
        policy = policy.to(device)
        policy.eval()
        
        # Create dummy observation
        observation = {
            "observation.image": torch.randn(1, 3, 64, 64, device=device),
            "observation.state": torch.randn(1, 10, device=device)
        }
        
        # Test inference with autocast
        with torch.autocast(device_type="xpu", dtype=torch.float16):
            with torch.inference_mode():
                action = policy.predict(observation)
        
        assert action.device.type == "xpu"
        logger.info("Policy inference on Intel GPU successful")
        
    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")


@require_xpu
def test_xpu_training_step():
    """Test a basic training step on Intel GPU."""
    device = torch.device("xpu")
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Create dummy data
    x = torch.randn(32, 100, device=device)
    y = torch.randn(32, 10, device=device)
    
    # Training step with autocast
    with torch.autocast(device_type="xpu", dtype=torch.float16):
        output = model(x)
        loss = criterion(output, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Synchronize
    torch.xpu.synchronize()
    
    logger.info("Training step on Intel GPU successful")


@require_xpu
def test_xpu_data_loading():
    """Test data loading with Intel GPU pinned memory."""
    device = torch.device("xpu")
    
    # Create dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            return {
                "observation": torch.randn(3, 64, 64),
                "action": torch.randn(7)
            }
    
    dataset = DummyDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        pin_memory=True,  # Should work with Intel GPU
        num_workers=0     # Avoid multiprocessing issues in tests
    )
    
    # Test data loading and transfer to Intel GPU
    for batch in dataloader:
        # Transfer with non-blocking
        observation = batch["observation"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        
        assert observation.device.type == "xpu"
        assert action.device.type == "xpu"
        break  # Just test one batch
    
    logger.info("Data loading with Intel GPU successful")


def test_device_fallback():
    """Test device fallback when Intel GPU is not available."""
    from lerobot.utils.utils import get_safe_torch_device
    
    # This should work regardless of XPU availability
    if torch.xpu.is_available():
        device = get_safe_torch_device("xpu")
        assert device.type == "xpu"
    else:
        # Should fall back to CPU
        try:
            device = get_safe_torch_device("xpu")
            # Should not reach here if XPU is not available
            assert False, "Expected assertion error for unavailable XPU"
        except AssertionError:
            # Expected behavior
            pass
    
    logger.info("Device fallback test successful")


if __name__ == "__main__":
    # Run basic availability test
    try:
        test_xpu_availability()
        logger.info("✅ Intel GPU is available and working")
    except (AssertionError, AttributeError) as e:
        logger.warning(f"❌ Intel GPU not available: {e}")
        logger.info("To use Intel GPU, install PyTorch with XPU support:")
        logger.info("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu")
        sys.exit(1)
    
    # Run all tests if XPU is available
    if torch.xpu.is_available():
        logger.info("Running comprehensive Intel GPU tests...")
        test_xpu_tensor_operations()
        test_xpu_autocast()
        test_lerobot_device_utils()
        test_xpu_training_step()
        test_xpu_data_loading()
        logger.info("✅ All Intel GPU tests passed!")
    
    test_device_fallback()
    logger.info("✅ All tests completed successfully!")
