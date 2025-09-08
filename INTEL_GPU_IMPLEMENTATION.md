# LeRobot Intel GPU Support Implementation Summary

## Overview

This implementation adds comprehensive Intel GPU support to LeRobot using PyTorch's XPU backend. The changes enable users to train and run robotic policies on Intel Arc GPUs and Intel Data Center GPUs.

## Key Changes Made

### 1. Core Device Utilities (`src/lerobot/utils/utils.py`)

**Enhanced device detection and management:**
- Added XPU device support in `get_safe_torch_device()`
- Updated `is_torch_device_available()` to check Intel GPU availability
- Enhanced `get_safe_dtype()` to handle Intel GPU dtype preferences
- Updated `print_cuda_memory_usage()` to support multiple GPU types

```python
# New functionality
def get_safe_torch_device(try_device: str | torch.device, log: bool = False) -> torch.device:
    # Now supports "xpu" for Intel GPU
    case "xpu":
        assert hasattr(torch, 'xpu') and torch.xpu.is_available()
        device = torch.device("xpu")
```

### 2. Device-Specific Operations (`src/lerobot/utils/device_utils.py`)

**New comprehensive device utility module:**
- `synchronize_device()` - Cross-platform device synchronization
- `empty_cache()` - Memory cache management for all devices
- `get_memory_info()` - Unified memory information retrieval
- `get_device_properties()` - Device capability detection
- Performance optimization helpers for different device types

### 3. Device Factory (`src/lerobot/utils/device_factory.py`)

**New high-level device management:**
- `DeviceManager` class for automatic device selection and optimization
- Auto-selection priority: Intel GPU > CUDA > MPS > CPU
- Device-specific configuration recommendations
- Simplified API for common device operations

### 4. Training Infrastructure Updates

**Updated training scripts for multi-device support:**

**`src/lerobot/scripts/train.py`:**
- Enhanced data loading with Intel GPU pinned memory support
- Updated non-blocking tensor transfers for XPU
- Compatible gradient scaling for Intel GPU

**`src/lerobot/scripts/eval.py`:**
- Added XPU support for evaluation pipelines
- Non-blocking transfers for Intel GPU during inference

### 5. Control and Inference Updates

**`src/lerobot/utils/control_utils.py`:**
- Extended autocast support to Intel GPU
- Mixed precision inference for XPU devices

**`src/lerobot/utils/transition.py`:**
- Non-blocking memory transfers for Intel GPU
- Optimized data movement patterns

### 6. Configuration Updates

**Policy configurations (`src/lerobot/configs/policies.py`):**
- Updated device options to include XPU
- Documentation improvements for device selection

**Environment configurations (`src/lerobot/envs/configs.py`):**
- Added Intel GPU device option with proper documentation
- Consistent device configuration across all environment types

### 7. Testing Infrastructure

**Enhanced testing framework:**

**`tests/utils.py`:**
- New `require_xpu()` decorator for Intel GPU tests
- `require_accelerator()` decorator for any GPU testing
- Automatic device priority in test environment selection

**`tests/test_intel_gpu.py`:**
- Comprehensive Intel GPU functionality tests
- Tensor operations, memory management, and LeRobot integration tests

### 8. Documentation and Guides

**Comprehensive documentation:**

**`INTEL_GPU_SETUP.md`:**
- Complete installation guide for Intel GPU support
- Driver installation instructions for Windows and Linux
- PyTorch XPU installation steps
- Troubleshooting guide

**`docs/source/intel_gpu.md`:**
- Detailed Intel GPU usage guide
- Performance optimization recommendations
- Code examples and best practices

**`docs/source/migration_intel_gpu.md`:**
- Migration guide from CUDA to Intel GPU
- Side-by-side code comparisons
- Common issues and solutions

### 9. Example Updates

**Updated example scripts:**
- `examples/2_evaluate_pretrained_policy.py` - Added XPU device option
- `examples/3_train_policy.py` - Enhanced device selection comments
- Policy conversion scripts with auto-device detection

### 10. Package Configuration

**`pyproject.toml`:**
- Added `intel-gpu` optional dependency group
- Integration with main package installation options

## Usage Examples

### Basic Intel GPU Usage

```python
import torch
from lerobot.utils.device_factory import create_device_manager

# Automatic device selection (prefers Intel GPU if available)
device_manager = create_device_manager()
device = device_manager.get_device()

# Move model to Intel GPU
model = model.to(device)

# Optimized training with Intel GPU
with device_manager.create_autocast_context():
    output = model(input_data)
```

### Training with Intel GPU

```bash
# Train with Intel GPU
lerobot-train \
    --dataset.repo_id=your/dataset \
    --policy.type=act \
    --policy.device=xpu \
    --policy.use_amp=true \
    --batch_size=16

# Evaluate with Intel GPU  
lerobot-eval \
    --policy.path=your/policy \
    --policy.device=xpu
```

### Programmatic Usage

```python
from lerobot.policies import DiffusionPolicy
from lerobot.utils.device_factory import auto_select_device

# Auto-select best device (Intel GPU preferred)
device = auto_select_device(log_info=True)

# Load and configure policy
policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")
policy = policy.to(device)

# Use with automatic mixed precision
with torch.autocast(device_type=device.type, dtype=torch.float16):
    action = policy.predict(observation)
```

## Performance Optimizations

### Intel GPU Specific Optimizations

1. **Optimal Batch Sizes**: Multiples of 16 work best for Intel GPU
2. **Memory Management**: Automatic cache management and synchronization
3. **Mixed Precision**: FP16 autocast support for better performance
4. **Data Loading**: Pinned memory and non-blocking transfers

### Automatic Configuration

The implementation includes automatic configuration based on device type:

```python
config = get_device_config("xpu")
# Returns:
# {
#     "device": device("xpu"),
#     "supports_autocast": True,
#     "optimal_batch_multiplier": 16,
#     "recommended_dtype": torch.float32,
#     "use_amp": True,
#     "max_batch_size_hint": 32
# }
```

## Compatibility

### Hardware Support
- Intel Arc A-Series Graphics (Alchemist)
- Intel Arc B-Series Graphics (Battlemage)
- Intel Core Ultra Processors with Intel Arc Graphics
- Intel Data Center GPU Max Series (Ponte Vecchio)

### Software Requirements
- PyTorch 2.5+ with XPU support
- Intel GPU drivers
- Windows 11, Ubuntu 24.04+, or WSL2

### Backward Compatibility
- All existing CUDA code continues to work unchanged
- Automatic fallback to CUDA or CPU when Intel GPU unavailable
- Legacy function names preserved (e.g., `print_cuda_memory_usage()`)

## Testing and Validation

### Test Coverage
- Basic tensor operations on Intel GPU
- Memory management and synchronization
- Training and inference workflows
- Device detection and fallback mechanisms
- Integration with existing LeRobot features

### Validation Scripts
- `simple_intel_gpu_test.py` - Basic functionality test
- `test_intel_gpu_support.py` - Comprehensive test suite
- CI/CD integration with device-specific test decorators

## Future Enhancements

1. **Performance Profiling**: Intel GPU specific performance monitoring
2. **Memory Optimization**: Advanced memory pooling for Intel GPU
3. **Multi-GPU**: Support for multiple Intel GPU training
4. **Quantization**: Intel GPU specific model quantization
5. **Advanced Compilation**: Intel GPU optimized torch.compile modes

This implementation provides a solid foundation for Intel GPU support in LeRobot while maintaining full backward compatibility and following established patterns in the codebase.
