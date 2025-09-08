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

"""Device-specific utilities for LeRobot with support for CUDA, Intel GPU (XPU), MPS, and CPU."""

import logging
import torch


def synchronize_device(device: torch.device | str) -> None:
    """
    Synchronize device operations to ensure completion.
    
    Args:
        device: The device to synchronize. Can be torch.device or string.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        if hasattr(torch.xpu, 'synchronize'):
            torch.xpu.synchronize(device)
    # MPS and CPU don't need explicit synchronization


def empty_cache(device: torch.device | str) -> None:
    """
    Empty device cache to free memory.
    
    Args:
        device: The device whose cache should be emptied.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        if hasattr(torch.xpu, 'empty_cache'):
            torch.xpu.empty_cache()
    # MPS and CPU don't have explicit cache management


def get_memory_info(device: torch.device | str) -> dict[str, int]:
    """
    Get memory information for the device.
    
    Args:
        device: The device to query memory for.
        
    Returns:
        Dictionary with 'allocated' and 'reserved' memory in bytes.
        Returns zeros for devices that don't support memory querying.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if device.type == "cuda":
        return {
            'allocated': torch.cuda.memory_allocated(device),
            'reserved': torch.cuda.memory_reserved(device)
        }
    elif device.type == "xpu":
        if hasattr(torch.xpu, 'memory_allocated') and hasattr(torch.xpu, 'memory_reserved'):
            return {
                'allocated': torch.xpu.memory_allocated(device),
                'reserved': torch.xpu.memory_reserved(device)
            }
        else:
            logging.warning("XPU memory info not available in this PyTorch version")
            return {'allocated': 0, 'reserved': 0}
    else:
        return {'allocated': 0, 'reserved': 0}


def get_device_properties(device: torch.device | str) -> dict:
    """
    Get device properties and capabilities.
    
    Args:
        device: The device to query properties for.
        
    Returns:
        Dictionary with device properties.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return {
            'name': props.name,
            'major': props.major,
            'minor': props.minor,
            'total_memory': props.total_memory,
            'multi_processor_count': props.multi_processor_count
        }
    elif device.type == "xpu":
        if hasattr(torch.xpu, 'get_device_properties'):
            props = torch.xpu.get_device_properties(device)
            return {
                'name': getattr(props, 'name', 'Intel GPU'),
                'total_memory': getattr(props, 'total_memory', 0),
                'max_work_group_size': getattr(props, 'max_work_group_size', 0)
            }
        else:
            return {'name': 'Intel GPU (XPU)', 'total_memory': 0}
    else:
        return {'name': f'{device.type.upper()} Device', 'total_memory': 0}


def set_device_memory_fraction(device: torch.device | str, fraction: float) -> None:
    """
    Set memory fraction for the device.
    
    Args:
        device: The device to set memory fraction for.
        fraction: Memory fraction to set (0.0 to 1.0).
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    elif device.type == "xpu":
        if hasattr(torch.xpu, 'set_per_process_memory_fraction'):
            torch.xpu.set_per_process_memory_fraction(fraction, device)
        else:
            logging.warning("XPU memory fraction setting not available in this PyTorch version")
    else:
        logging.warning(f"Memory fraction setting not supported for {device.type}")


def get_optimal_batch_size_multiplier(device: torch.device | str) -> int:
    """
    Get optimal batch size multiplier for the device.
    
    Args:
        device: The device to get multiplier for.
        
    Returns:
        Recommended batch size multiplier for optimal performance.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if device.type == "cuda":
        # CUDA typically works well with multiples of 32 for tensor cores
        return 32
    elif device.type == "xpu":
        # Intel GPU typically works well with multiples of 16
        return 16
    elif device.type == "mps":
        # Apple Silicon works well with multiples of 8
        return 8
    else:
        # CPU doesn't have specific alignment requirements
        return 1


def supports_autocast(device: torch.device | str) -> bool:
    """
    Check if the device supports autocast for mixed precision.
    
    Args:
        device: The device to check.
        
    Returns:
        True if autocast is supported, False otherwise.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    return device.type in ["cuda", "xpu"]


def supports_compile(device: torch.device | str) -> bool:
    """
    Check if the device supports torch.compile.
    
    Args:
        device: The device to check.
        
    Returns:
        True if torch.compile is supported, False otherwise.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    # torch.compile support varies by device and PyTorch version
    # Generally supported on CUDA and newer versions support XPU
    return device.type in ["cuda", "xpu", "cpu"]


def get_optimal_dtype_for_device(device: torch.device | str, dtype: torch.dtype) -> torch.dtype:
    """
    Get optimal dtype for the device, considering device limitations.
    
    Args:
        device: The device to optimize for.
        dtype: The requested dtype.
        
    Returns:
        Optimal dtype for the device.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    # MPS doesn't support float64
    if device.type == "mps" and dtype == torch.float64:
        return torch.float32
    
    # Intel GPU typically works better with float32 for performance
    if device.type == "xpu" and dtype == torch.float64:
        return torch.float32
    
    return dtype


def log_device_info(device: torch.device | str) -> None:
    """
    Log comprehensive device information.
    
    Args:
        device: The device to log information for.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    logging.info(f"Using device: {device}")
    
    properties = get_device_properties(device)
    logging.info(f"Device properties: {properties}")
    
    memory_info = get_memory_info(device)
    if memory_info['allocated'] > 0 or memory_info['reserved'] > 0:
        logging.info(f"Memory allocated: {memory_info['allocated'] / 1024**3:.2f} GB")
        logging.info(f"Memory reserved: {memory_info['reserved'] / 1024**3:.2f} GB")
    
    logging.info(f"Supports autocast: {supports_autocast(device)}")
    logging.info(f"Supports compile: {supports_compile(device)}")
    logging.info(f"Optimal batch size multiplier: {get_optimal_batch_size_multiplier(device)}")
