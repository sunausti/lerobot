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

"""Device factory for automatic device selection and optimization in LeRobot."""

import logging
import torch
from typing import Optional, Union

from lerobot.utils.utils import get_safe_torch_device, is_torch_device_available
from lerobot.utils.device_utils import log_device_info, get_optimal_batch_size_multiplier


class DeviceManager:
    """Manages device selection and optimization for LeRobot."""
    
    def __init__(self, preferred_device: Optional[str] = None, log_info: bool = True):
        """
        Initialize device manager.
        
        Args:
            preferred_device: Preferred device ("xpu", "cuda", "mps", "cpu") or None for auto-selection
            log_info: Whether to log device information
        """
        self.device = self._select_device(preferred_device)
        self.log_info = log_info
        
        if log_info:
            log_device_info(self.device)
    
    def _select_device(self, preferred_device: Optional[str]) -> torch.device:
        """Select the best available device."""
        if preferred_device:
            if is_torch_device_available(preferred_device):
                return get_safe_torch_device(preferred_device, log=self.log_info)
            else:
                logging.warning(f"Preferred device '{preferred_device}' not available, auto-selecting...")
        
        # Auto-selection priority: Intel GPU > CUDA > MPS > CPU
        device_priority = ["xpu", "cuda", "mps", "cpu"]
        
        for device_name in device_priority:
            if is_torch_device_available(device_name):
                return get_safe_torch_device(device_name, log=self.log_info)
        
        # Fallback to CPU (should always work)
        return torch.device("cpu")
    
    def get_device(self) -> torch.device:
        """Get the selected device."""
        return self.device
    
    def to_device(self, tensor_or_model: Union[torch.Tensor, torch.nn.Module], **kwargs):
        """Move tensor or model to the selected device."""
        # For Intel GPU and CUDA, enable non-blocking transfers by default
        if self.device.type in ["cuda", "xpu"] and "non_blocking" not in kwargs:
            kwargs["non_blocking"] = True
        
        return tensor_or_model.to(self.device, **kwargs)
    
    def get_optimal_batch_size(self, base_batch_size: int) -> int:
        """Get optimal batch size for the selected device."""
        multiplier = get_optimal_batch_size_multiplier(self.device)
        
        # Round to nearest multiple
        if multiplier > 1:
            return ((base_batch_size + multiplier - 1) // multiplier) * multiplier
        return base_batch_size
    
    def create_autocast_context(self, dtype: torch.dtype = torch.float16, enabled: Optional[bool] = None):
        """Create autocast context for mixed precision."""
        if enabled is None:
            enabled = self.device.type in ["cuda", "xpu"]
        
        if enabled and self.device.type in ["cuda", "xpu"]:
            return torch.autocast(device_type=self.device.type, dtype=dtype)
        else:
            from contextlib import nullcontext
            return nullcontext()
    
    def create_grad_scaler(self, enabled: bool = True):
        """Create gradient scaler for mixed precision training."""
        from torch.amp import GradScaler
        return GradScaler(device=self.device.type, enabled=enabled and self.device.type in ["cuda", "xpu"])
    
    def synchronize(self):
        """Synchronize device operations."""
        from lerobot.utils.device_utils import synchronize_device
        synchronize_device(self.device)
    
    def empty_cache(self):
        """Empty device cache."""
        from lerobot.utils.device_utils import empty_cache
        empty_cache(self.device)
    
    def get_memory_info(self) -> dict:
        """Get device memory information."""
        from lerobot.utils.device_utils import get_memory_info
        return get_memory_info(self.device)


def create_device_manager(device: Optional[str] = None, **kwargs) -> DeviceManager:
    """
    Factory function to create a device manager.
    
    Args:
        device: Preferred device name or None for auto-selection
        **kwargs: Additional arguments for DeviceManager
        
    Returns:
        DeviceManager instance
    """
    return DeviceManager(preferred_device=device, **kwargs)


def auto_select_device(log_info: bool = False) -> torch.device:
    """
    Automatically select the best available device.
    
    Args:
        log_info: Whether to log device information
        
    Returns:
        Selected torch.device
    """
    manager = DeviceManager(preferred_device=None, log_info=log_info)
    return manager.get_device()


def get_device_config(device: Union[str, torch.device]) -> dict:
    """
    Get recommended configuration for a device.
    
    Args:
        device: Device to get configuration for
        
    Returns:
        Dictionary with recommended settings
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    config = {
        "device": device,
        "supports_autocast": device.type in ["cuda", "xpu"],
        "supports_compile": device.type in ["cuda", "xpu", "cpu"],
        "optimal_batch_multiplier": get_optimal_batch_size_multiplier(device),
        "pin_memory": device.type in ["cuda", "xpu"],
        "non_blocking_transfer": device.type in ["cuda", "xpu"]
    }
    
    # Device-specific recommendations
    if device.type == "xpu":
        config.update({
            "recommended_dtype": torch.float32,  # Intel GPU works well with FP32
            "use_amp": True,  # AMP can improve performance
            "max_batch_size_hint": 32,  # Conservative estimate
        })
    elif device.type == "cuda":
        config.update({
            "recommended_dtype": torch.float32,
            "use_amp": True,
            "max_batch_size_hint": 64,  # Can often handle larger batches
        })
    elif device.type == "mps":
        config.update({
            "recommended_dtype": torch.float32,  # MPS doesn't support FP64
            "use_amp": False,  # MPS autocast is limited
            "max_batch_size_hint": 16,
        })
    else:  # CPU
        config.update({
            "recommended_dtype": torch.float32,
            "use_amp": False,  # CPU doesn't benefit from autocast
            "max_batch_size_hint": 8,  # CPU is typically slower
        })
    
    return config
