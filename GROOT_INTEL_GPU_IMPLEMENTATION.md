# Groot Policy Intel GPU (XPU) Implementation Summary

This document summarizes the changes made to implement Intel GPU (XPU) support for the Groot policy in LeRobot.

## Overview

The implementation adds complete Intel GPU (XPU) support to the Groot policy, following the same patterns used for PI05. The key changes enable:
- Automatic device detection (xpu > cuda > cpu)
- Eager attention configuration (Flash Attention disabled for XPU)
- Proper device placement and synchronization
- BFloat16 precision support on XPU

## Files Modified

### 1. `src/lerobot/policies/groot/configuration_groot.py`

**Changes:**
- Added `import os` and `import torch` for device detection
- Added device configuration fields:
  - `device: str | None` - Auto-detect if None (xpu > cuda > cpu)
  - `use_xpu: bool | None` - Auto-detect Intel GPU availability
  - `use_eager_attention: bool = True` - Disable Flash Attention for XPU
  - `xpu_backend: str = "level_zero"` - Use Level Zero backend for Intel GPU

**New Methods:**
- `_auto_detect_device()` - Detects best available device (xpu > cuda > cpu)
- `_configure_xpu_backend()` - Configures Intel GPU backend (Level Zero) and disables Flash Attention

**Logic:**
```python
def _auto_detect_device(self) -> str:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
```

### 2. `src/lerobot/policies/groot/modeling_groot.py`

**Changes:**
- Added Intel GPU support documentation in module docstring
- Updated `__init__` to:
  - Initialize `self.device` from config
  - Move model to device with `self.to(self.device)`
  - Print device information

- Updated `_create_groot_model()` to:
  - Pass `device=self.config.device` to `GR00TN15.from_pretrained()`

- Updated `_handle_flash_attention_compatibility()` to:
  - Check for XPU/eager attention and disable Flash Attention
  - Set `TRANSFORMERS_NO_FLASH_ATTN=1` for Intel GPU
  - Print appropriate messages

- Updated `predict_action_chunk()` to:
  - Add XPU synchronization after inference: `torch.xpu.synchronize()`
  - Ensures accurate timing and completion

**Key Code:**
```python
# In predict_action_chunk()
if self.config.use_xpu and hasattr(torch, "xpu"):
    torch.xpu.synchronize()
```

### 3. `src/lerobot/policies/groot/groot_n1.py`

**Changes:**

#### EagleBackbone class:
- Added `device: str | None = None` parameter to `__init__`
- Auto-detects device if not specified (xpu > cuda > cpu)
- Stores device type in `self.device_type`
- Configures Eagle model config for XPU:
  - Sets `config.use_flash_attention = False`
  - Sets `config._attn_implementation = "eager"`
- Prints device and attention configuration

#### GR00TN15 class:
- Added `device: str | None = None` parameter to `__init__`
- Auto-detects device if not specified
- Stores device in `self._device`
- Passes device to EagleBackbone via `backbone_cfg["device"] = device`

- Updated `from_pretrained()` classmethod:
  - Extracts `device` from kwargs
  - Passes `device=device` to parent `from_pretrained()`

**Key Code:**
```python
# In EagleBackbone.__init__()
if device == "xpu":
    if hasattr(config, "use_flash_attention"):
        config.use_flash_attention = False
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
```

### 4. `benchmarks/groot/benchmark.py` (NEW)

**Purpose:** Benchmark script for Groot policy inference and training performance on CPU/XPU/CUDA.

**Features:**
- Multi-device support (cpu/xpu/cuda)
- Precision options (float32/bfloat16)
- Inference and training modes
- Synthetic batch generation matching Groot's input format
- Proper GPU synchronization for accurate timing
- Comprehensive error handling and reporting

**Key Functions:**
- `is_device_available(device)` - Check device availability
- `synchronize(device)` - Synchronize GPU operations
- `prepare_dummy_batch(...)` - Generate synthetic inputs
  - Random states (padded to max_state_dim)
  - Random images (Eagle pixel format)
  - Dummy attention masks and input IDs
  - Random embodiment IDs
  - Optional actions for training mode
- `run_iteration(...)` - Run single inference/training pass
- `run_benchmark(...)` - Full benchmark with warmup and timing

**Usage:**
```bash
# Benchmark on Intel GPU
ONEAPI_DEVICE_SELECTOR=level_zero:gpu python benchmarks/groot/benchmark.py --devices xpu --num-runs 10

# Benchmark all devices
python benchmarks/groot/benchmark.py --devices cpu xpu cuda --num-runs 10 --num-warmup 2

# Training mode
python benchmarks/groot/benchmark.py --devices xpu --mode training
```

### 5. `benchmarks/groot/README.md` (NEW)

**Purpose:** Documentation for the Groot benchmark script.

**Contents:**
- Features overview
- Requirements
- Usage examples
- Command-line arguments
- Intel GPU setup instructions
- Example output
- Implementation details
- Troubleshooting guide
- Performance notes

## Key Design Patterns

### 1. Device Auto-Detection

Priority order: XPU > CUDA > CPU
```python
if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = "xpu"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

### 2. Flash Attention Handling

For Intel GPU, Flash Attention is not supported:
```python
if self.config.use_xpu or self.config.use_eager_attention:
    os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
    # Configure model to use eager attention
```

### 3. Device Synchronization

Ensure operations complete before timing:
```python
if device == "cuda":
    torch.cuda.synchronize()
elif device == "xpu":
    torch.xpu.synchronize()
```

### 4. BFloat16 Autocast

Support mixed precision on compatible devices:
```python
with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
    outputs = self._groot_model.get_action(groot_inputs)
```

## Environment Variables

### For Intel GPU (XPU):

1. **Level Zero Backend** (recommended):
   ```bash
   export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
   ```

2. **Disable Flash Attention** (automatically set by config):
   ```bash
   export TRANSFORMERS_NO_FLASH_ATTN=1
   ```

3. **Debug Logging** (optional):
   ```bash
   export LOG_LEVEL=DEBUG
   ```

## Testing

### Quick Test (CPU)
```bash
python benchmarks/groot/benchmark.py --devices cpu --num-runs 1 --num-warmup 0
```

### Intel GPU Test
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu python benchmarks/groot/benchmark.py --devices xpu --num-runs 5 --num-warmup 1
```

### Full Comparison
```bash
python benchmarks/groot/benchmark.py --devices cpu xpu cuda --num-runs 10 --num-warmup 2
```

## Compatibility Notes

### Action Head
- Uses standard PyTorch device handling
- Automatically moves with parent module
- No XPU-specific changes needed

### Eagle Vision Model
- Loaded via `AutoModel.from_config()`
- Attention mechanism configured in EagleBackbone
- Supports eager attention for XPU

### Flow Matching Action Head
- Standard PyTorch modules
- Device placement handled automatically
- No Flash Attention dependencies

## Performance Considerations

1. **Eager Attention**: Slightly slower than Flash Attention but necessary for XPU
2. **BFloat16**: Recommended for performance on XPU and modern GPUs
3. **Level Zero Backend**: Preferred over OpenCL for Intel GPU
4. **Batch Size**: Start with 1 for benchmarking, increase as memory allows

## Migration from CUDA/CPU

The implementation is designed to be backward compatible:
- If no device is specified, auto-detects best available
- CUDA and CPU paths unchanged
- XPU support added as additional option

Existing code works without modification:
```python
# Old code (still works)
config = GrootConfig()
policy = GrootPolicy(config)

# New code (explicit device)
config = GrootConfig(device="xpu")
policy = GrootPolicy(config)
```

## Future Work

1. **Processor Integration**: Add Eagle processor support for real data
2. **Multi-GPU**: Support for distributed training on XPU
3. **Quantization**: INT8 inference optimization for Intel GPU
4. **Benchmark Extensions**: Add memory profiling and throughput metrics
5. **Real Dataset Testing**: Validate with actual robot datasets

## References

- PI05 Intel GPU implementation (similar patterns)
- Intel Extension for PyTorch documentation
- Transformers library attention mechanisms
- LeRobot policy architecture

## Change Summary

**Files Added:** 2
- `benchmarks/groot/benchmark.py`
- `benchmarks/groot/README.md`

**Files Modified:** 3
- `src/lerobot/policies/groot/configuration_groot.py`
- `src/lerobot/policies/groot/modeling_groot.py`
- `src/lerobot/policies/groot/groot_n1.py`

**Total Lines Added:** ~700
**Total Lines Modified:** ~100

**Backward Compatibility:** ✅ Maintained (auto-detection with fallback to CPU)
**Testing:** ⚠️ Requires Intel GPU hardware for full validation
