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
DataLoader utilities optimized for Intel GPU compatibility.
"""

import logging
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from lerobot.utils.intel_gpu_utils import is_intel_gpu_available, optimize_dataloader_for_intel_gpu


def create_intel_gpu_dataloader(
    dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    timeout: float = 0,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader optimized for Intel GPU compatibility.
    
    This function automatically adjusts DataLoader parameters for Intel GPU environments
    to avoid common compatibility issues like multiprocessing problems and memory pinning.
    
    Args:
        dataset: The dataset to load from
        batch_size: Batch size for loading
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes (will be adjusted for Intel GPU)
        pin_memory: Whether to pin memory (will be disabled for Intel GPU)
        drop_last: Whether to drop the last incomplete batch
        timeout: Timeout for collecting a batch
        **kwargs: Additional arguments passed to DataLoader
        
    Returns:
        DataLoader: Optimized DataLoader instance
    """
    
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
        'timeout': timeout,
        **kwargs
    }
    
    # Optimize for Intel GPU
    dataloader_kwargs = optimize_dataloader_for_intel_gpu(dataloader_kwargs)
    
    if is_intel_gpu_available():
        logging.info("Creating DataLoader with Intel GPU optimizations")
    
    return DataLoader(dataset, **dataloader_kwargs)


def intel_gpu_collate_fn(batch):
    """
    Custom collate function for Intel GPU that handles tensor transfers efficiently.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched data with tensors moved to Intel GPU if available
    """
    # Use default collate function first
    from torch.utils.data.dataloader import default_collate
    
    try:
        collated = default_collate(batch)
    except Exception as e:
        logging.error(f"Default collate failed: {e}")
        raise e
    
    # Move tensors to Intel GPU if available
    if is_intel_gpu_available():
        collated = _move_to_intel_gpu_recursive(collated)
    
    return collated


def _move_to_intel_gpu_recursive(obj):
    """Recursively move tensors to Intel GPU."""
    if isinstance(obj, torch.Tensor):
        try:
            return obj.to('xpu', non_blocking=True)
        except Exception as e:
            logging.warning(f"Failed to move tensor to Intel GPU: {e}")
            return obj
    elif isinstance(obj, dict):
        return {key: _move_to_intel_gpu_recursive(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_intel_gpu_recursive(item) for item in obj)
    else:
        return obj


class IntelGPUDataLoader(DataLoader):
    """
    DataLoader subclass with Intel GPU specific optimizations.
    
    This class provides additional functionality for Intel GPU environments including:
    - Automatic parameter optimization
    - Error handling for Intel GPU specific issues
    - Memory management
    """
    
    def __init__(self, dataset, **kwargs):
        # Apply Intel GPU optimizations
        optimized_kwargs = optimize_dataloader_for_intel_gpu(kwargs)
        
        # Add Intel GPU specific collate function if not provided
        if 'collate_fn' not in optimized_kwargs and is_intel_gpu_available():
            optimized_kwargs['collate_fn'] = intel_gpu_collate_fn
        
        super().__init__(dataset, **optimized_kwargs)
        
        if is_intel_gpu_available():
            logging.info("Initialized IntelGPUDataLoader with optimizations")
    
    def __iter__(self):
        """Override iterator to handle Intel GPU specific errors."""
        try:
            return super().__iter__()
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "xpu" in error_msg or "intel" in error_msg:
                logging.error(
                    f"Intel GPU specific error in DataLoader: {e}\n"
                    "Try reducing num_workers or batch_size, or use video_backend='pyav'"
                )
            raise e
    
    def __len__(self):
        """Override len to handle potential issues."""
        try:
            return super().__len__()
        except Exception as e:
            logging.warning(f"Error getting DataLoader length: {e}")
            # Return dataset length as fallback
            return len(self.dataset)


def get_recommended_dataloader_config_for_intel_gpu(
    dataset_size: int,
    available_memory_mb: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get recommended DataLoader configuration for Intel GPU based on dataset size and memory.
    
    Args:
        dataset_size: Size of the dataset
        available_memory_mb: Available Intel GPU memory in MB (optional)
        
    Returns:
        Dict with recommended DataLoader configuration
    """
    if not is_intel_gpu_available():
        return {
            'num_workers': 4,
            'batch_size': 32,
            'pin_memory': True,
            'prefetch_factor': 2
        }
    
    # Intel GPU specific recommendations
    config = {
        'num_workers': 1,  # Conservative for Intel GPU
        'pin_memory': False,  # Intel GPU doesn't support pin_memory
        'prefetch_factor': 1,  # Reduce memory pressure
        'persistent_workers': False,  # Avoid worker persistence issues
    }
    
    # Adjust batch size based on memory and dataset size
    if available_memory_mb is not None:
        if available_memory_mb > 12000:  # > 12GB
            config['batch_size'] = 64
            config['num_workers'] = 2
        elif available_memory_mb > 8000:  # > 8GB
            config['batch_size'] = 32
            config['num_workers'] = 2
        elif available_memory_mb > 4000:  # > 4GB
            config['batch_size'] = 16
        else:
            config['batch_size'] = 8
    else:
        # Conservative default
        config['batch_size'] = 16
    
    # Adjust for small datasets
    if dataset_size < 1000:
        config['num_workers'] = 0
        config['batch_size'] = min(config['batch_size'], 8)
    
    logging.info(f"Recommended Intel GPU DataLoader config: {config}")
    return config
