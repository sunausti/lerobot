# Migrating from NVIDIA CUDA to Intel GPU in LeRobot

This guide helps you migrate your LeRobot workflows from NVIDIA CUDA GPUs to Intel GPUs.

## Quick Migration Checklist

### 1. Install Intel GPU Support

Replace your PyTorch installation with Intel GPU support:

```bash
# Uninstall current PyTorch (optional)
pip uninstall torch torchvision torchaudio

# Install PyTorch with Intel GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

### 2. Update Device Configuration

Replace all instances of `device="cuda"` with `device="xpu"`:

**Before (CUDA):**
```bash
lerobot-train --policy.device=cuda
lerobot-eval --policy.device=cuda
```

**After (Intel GPU):**
```bash
lerobot-train --policy.device=xpu
lerobot-eval --policy.device=xpu
```

### 3. Update Python Code

**Before (CUDA):**
```python
device = "cuda"
policy = policy.to("cuda")
```

**After (Intel GPU):**
```python
device = "xpu"
policy = policy.to("xpu")
```

## Code Changes Required

### Configuration Files

Update any configuration files that specify device:

```yaml
# config.yaml
policy:
  device: "xpu"  # was "cuda"
  
env:
  device: "xpu"  # was "cuda"
```

### Training Scripts

**Before:**
```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**After:**
```python
import torch

def get_best_device():
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda") 
    else:
        return torch.device("cpu")

device = get_best_device()
```

### Memory Management

**Before (CUDA):**
```python
# CUDA memory management
torch.cuda.empty_cache()
torch.cuda.synchronize()
memory_allocated = torch.cuda.memory_allocated()
```

**After (Intel GPU):**
```python
# Intel GPU memory management  
if hasattr(torch.xpu, 'empty_cache'):
    torch.xpu.empty_cache()
if hasattr(torch.xpu, 'synchronize'):
    torch.xpu.synchronize()
if hasattr(torch.xpu, 'memory_allocated'):
    memory_allocated = torch.xpu.memory_allocated()
```

Or use LeRobot's device utilities:
```python
from lerobot.utils.device_utils import empty_cache, synchronize_device, get_memory_info

empty_cache(device)
synchronize_device(device)
memory_info = get_memory_info(device)
```

## Performance Considerations

### Batch Size Adjustments

Intel GPUs may have different memory constraints:

**CUDA (typical):**
```bash
--batch_size=32
```

**Intel GPU (may need adjustment):**
```bash
--batch_size=16  # Start with smaller batch size
```

### Mixed Precision

Both support mixed precision, but Intel GPU configuration:

```python
# Both CUDA and Intel GPU support autocast
with torch.autocast(device_type=device.type, dtype=torch.float16):
    output = model(input)
```

### Compilation

Intel GPU supports torch.compile (PyTorch 2.7+):

```python
# Works for both CUDA and Intel GPU
if device.type in ["cuda", "xpu"]:
    model = torch.compile(model)
```

## Environment Variables

Update environment variables for Intel GPU:

**Add for Intel GPU:**
```bash
export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

## Testing Your Migration

Use the Intel GPU test script:

```bash
python tests/test_intel_gpu.py
```

Or test manually:

```python
import torch
from lerobot.utils.device_utils import log_device_info

# Test Intel GPU availability
if torch.xpu.is_available():
    device = torch.device("xpu")
    log_device_info(device)
    print("✅ Intel GPU is ready!")
else:
    print("❌ Intel GPU not available")
```

## Common Migration Issues

### Issue: "RuntimeError: No XPU devices found"

**Solution:** Install Intel GPU drivers and PyTorch XPU support:
```bash
# Install Intel GPU drivers first
# Then install PyTorch with XPU support
pip install torch --index-url https://download.pytorch.org/whl/xpu
```

### Issue: "Out of memory" errors

**Solution:** Reduce batch size or enable memory optimizations:
```bash
--batch_size=8  # Reduce from 32
--policy.use_amp=true  # Enable mixed precision
```

### Issue: Slower training than expected

**Solutions:**
1. Use optimal batch sizes (multiples of 16)
2. Enable torch.compile:
   ```python
   model = torch.compile(model)
   ```
3. Use mixed precision:
   ```bash
   --policy.use_amp=true
   ```

### Issue: Import errors with torch.xpu

**Solution:** Update PyTorch version:
```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/xpu
```

## Performance Comparison

Expected performance differences:

| Aspect | NVIDIA CUDA | Intel GPU |
|--------|-------------|-----------|
| Memory | Up to 24GB+ | Typically 8-16GB |
| Precision | FP64/FP32/FP16/BF16 | FP32/FP16/BF16 (limited FP64) |
| Batch Size | 32-128 typical | 8-32 typical |
| Compilation | Mature support | Growing support |

## Validation Checklist

After migration, verify:

- [ ] Intel GPU is detected: `torch.xpu.is_available()`
- [ ] Models load on Intel GPU: `model.to("xpu")`
- [ ] Training runs without errors
- [ ] Evaluation produces expected results
- [ ] Memory usage is reasonable
- [ ] Performance meets expectations

## Rollback Plan

If you need to revert to CUDA:

```bash
# Reinstall CUDA PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Change device back to "cuda" in your configurations
```

## Getting Help

If you encounter issues:

1. Check [Intel GPU PyTorch documentation](https://pytorch.org/docs/stable/notes/get_start_xpu.html)
2. Review the [LeRobot Intel GPU guide](intel_gpu.md)
3. Run diagnostics: `python tests/test_intel_gpu.py`
4. Report issues on [LeRobot GitHub](https://github.com/huggingface/lerobot/issues)

## Example: Complete Migration

**Before (train_policy.py):**
```python
import torch
from lerobot.policies.diffusion import DiffusionPolicy

device = torch.device("cuda")
policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")
policy = policy.to(device)

# Training loop
for batch in dataloader:
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    with torch.autocast(device_type="cuda"):
        loss = policy.compute_loss(batch)
    loss.backward()
```

**After (train_policy.py):**
```python
import torch
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.utils.utils import get_safe_torch_device

device = get_safe_torch_device("xpu")  # Automatic fallback to CUDA/CPU
policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")
policy = policy.to(device)

# Training loop
for batch in dataloader:
    batch = {k: v.to(device, non_blocking=device.type in ["cuda", "xpu"]) 
             for k, v in batch.items()}
    with torch.autocast(device_type=device.type, enabled=device.type in ["cuda", "xpu"]):
        loss = policy.compute_loss(batch)
    loss.backward()
```

This migration maintains compatibility with both CUDA and Intel GPU environments.
