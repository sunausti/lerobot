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

"""
Intel GPU utilities for LeRobot to handle Intel GPU (XPU) specific optimizations and compatibility issues.
"""

import logging
import torch
from typing import Dict, Any


def is_intel_gpu_available() -> bool:
    """Check if Intel GPU (XPU) is available."""
    try:
        return hasattr(torch, 'xpu') and torch.xpu.is_available()
    except Exception:
        return False


def get_intel_gpu_device_count() -> int:
    """Get the number of available Intel GPU devices."""
    if is_intel_gpu_available():
        try:
            return torch.xpu.device_count()
        except Exception:
            return 0
    return 0


def get_video_backend_for_intel_gpu() -> str:
    """Get the recommended video backend for Intel GPU environment."""
    if is_intel_gpu_available():
        logging.info("Intel GPU detected, using 'pyav' video backend for compatibility")
        return "pyav"
    else:
        # Check if torchcodec is available
        try:
            import importlib
            if importlib.util.find_spec("torchcodec"):
                return "torchcodec"
            else:
                return "pyav"
        except Exception:
            return "pyav"


def optimize_dataloader_for_intel_gpu(dataloader_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize DataLoader settings for Intel GPU compatibility."""
    if is_intel_gpu_available():
        # Intel GPU specific optimizations
        optimized_kwargs = dataloader_kwargs.copy()
        
        # Disable pin_memory as Intel GPU doesn't support it
        optimized_kwargs['pin_memory'] = False
        
        # Reduce num_workers to avoid multiprocessing issues with Intel GPU
        # Intel GPU drivers may have issues with multiple worker processes
        if 'num_workers' in optimized_kwargs:
            original_workers = optimized_kwargs['num_workers']
            optimized_workers = min(original_workers, 2)  # Limit to 2 workers max
            optimized_kwargs['num_workers'] = optimized_workers
            
            if original_workers > optimized_workers:
                logging.info(
                    f"Intel GPU detected: reducing num_workers from {original_workers} to {optimized_workers} "
                    "for better compatibility"
                )
        
        # Set prefetch_factor to reduce memory pressure
        if optimized_kwargs.get('num_workers', 0) > 0:
            optimized_kwargs['prefetch_factor'] = min(optimized_kwargs.get('prefetch_factor', 2), 2)
        
        logging.info("DataLoader optimized for Intel GPU compatibility")
        return optimized_kwargs
    
    return dataloader_kwargs


def transfer_to_intel_gpu(tensor: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
    """Transfer tensor to Intel GPU if available."""
    if is_intel_gpu_available():
        try:
            return tensor.to('xpu', non_blocking=non_blocking)
        except Exception as e:
            logging.warning(f"Failed to transfer tensor to Intel GPU: {e}")
            return tensor
    return tensor


def empty_intel_gpu_cache():
    """Empty Intel GPU cache to free memory."""
    if is_intel_gpu_available():
        try:
            torch.xpu.empty_cache()
            logging.debug("Intel GPU cache emptied")
        except Exception as e:
            logging.warning(f"Failed to empty Intel GPU cache: {e}")


def get_intel_gpu_memory_info() -> Dict[str, int]:
    """Get Intel GPU memory information."""
    if is_intel_gpu_available():
        try:
            device = torch.device('xpu:0')
            allocated = torch.xpu.memory_allocated(device)
            cached = torch.xpu.memory_reserved(device) if hasattr(torch.xpu, 'memory_reserved') else 0
            return {
                'allocated': allocated,
                'cached': cached,
                'allocated_mb': allocated // (1024 * 1024),
                'cached_mb': cached // (1024 * 1024)
            }
        except Exception as e:
            logging.warning(f"Failed to get Intel GPU memory info: {e}")
            return {}
    return {}


def log_intel_gpu_info():
    """Log Intel GPU information for debugging."""
    if is_intel_gpu_available():
        device_count = get_intel_gpu_device_count()
        logging.info(f"Intel GPU (XPU) detected: {device_count} device(s) available")
        
        for i in range(device_count):
            try:
                device_name = torch.xpu.get_device_name(i)
                logging.info(f"  Device {i}: {device_name}")
            except Exception:
                logging.info(f"  Device {i}: Intel GPU")
        
        memory_info = get_intel_gpu_memory_info()
        if memory_info:
            logging.info(f"Intel GPU memory: {memory_info['allocated_mb']} MB allocated, "
                        f"{memory_info['cached_mb']} MB cached")
    else:
        logging.info("Intel GPU (XPU) not available")


def setup_intel_gpu_environment():
    """Set up environment variables for optimal Intel GPU performance."""
    import os
    
    intel_gpu_env_vars = {
        'INTEL_GPU_ENABLED': '1',
        'SYCL_CACHE_PERSISTENT': '1',
        'ZE_FLAT_DEVICE_HIERARCHY': 'COMPOSITE',
        'SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS': '1',
    }
    
    for key, value in intel_gpu_env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            logging.debug(f"Set Intel GPU environment variable: {key}={value}")


def get_optimal_batch_size_for_intel_gpu(default_batch_size: int) -> int:
    """Get optimal batch size for Intel GPU based on available memory."""
    if not is_intel_gpu_available():
        return default_batch_size
    
    memory_info = get_intel_gpu_memory_info()
    if not memory_info:
        return default_batch_size
    
    # Simple heuristic: adjust batch size based on available memory
    # This is a conservative approach to avoid OOM errors
    allocated_mb = memory_info.get('allocated_mb', 0)
    
    if allocated_mb > 8000:  # > 8GB allocated
        return max(1, default_batch_size // 2)
    elif allocated_mb > 4000:  # > 4GB allocated
        return max(1, int(default_batch_size * 0.75))
    else:
        return default_batch_size


def intel_gpu_autocast_context():
    """Get autocast context for Intel GPU mixed precision training."""
    if is_intel_gpu_available():
        try:
            return torch.autocast(device_type='xpu', dtype=torch.float16)
        except Exception:
            # Fallback to no autocast if Intel GPU autocast is not supported
            import contextlib
            return contextlib.nullcontext()
    else:
        import contextlib
        return contextlib.nullcontext()
