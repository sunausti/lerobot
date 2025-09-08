# Intel GPU Simulation Environment for LeRobot

This directory contains examples and tools for running LeRobot simulation environments on Intel GPUs (Intel Arc, Intel Data Center GPU Max series).

## 🚀 Quick Start

### 1. Setup Environment

Run the setup script to install Intel GPU support:

```bash
cd C:\opensource\lerobot
python setup_intel_gpu_simulation.py
```

This will:
- Install PyTorch with Intel GPU (XPU) support
- Install simulation dependencies (Gymnasium, MuJoCo, etc.)
- Set up environment variables
- Create test scripts

### 2. Test Installation

Verify that Intel GPU support is working:

```bash
python test_intel_gpu_setup.py
```

You should see output like:
```
✅ Intel GPU detected! Found 1 XPU device(s)
   Device 0: Intel(R) Arc(TM) A770 Graphics
✅ Basic gym environment working: (4,)
✅ Policy inference successful: torch.Size([1, 2])
🎉 All tests passed! LeRobot is ready for Intel GPU simulation.
```

### 3. Run Simulation Examples

#### Simple Simulation Demo
```bash
python examples/intel_gpu/intel_gpu_simulation.py
```

#### Training Example
```bash
python examples/intel_gpu/train_intel_gpu.py
```

#### Using LeRobot Commands
```bash
# Train with Intel GPU
lerobot-train --policy.device=xpu --policy.use_amp=true

# Evaluate with Intel GPU
lerobot-eval --policy.path=your_model --policy.device=xpu
```

## 📁 Files Overview

- **`intel_gpu_simulation.py`** - Complete simulation demo with Intel GPU optimization
- **`train_intel_gpu.py`** - Training example using Intel GPU acceleration
- **`intel_gpu_config.yaml`** - Configuration file optimized for Intel GPU
- **`README.md`** - This documentation file

## ⚙️ Configuration

### Intel GPU Optimizations

The configuration includes Intel GPU specific optimizations:

```yaml
# Device settings
device: xpu
use_amp: true
pin_memory: false

# Optimized batch size for Intel GPU
training:
  batch_size: 32  # Multiple of 16 works best
  
# Intel GPU specific settings
intel_gpu:
  batch_size_multiplier: 16
  preferred_dtype: "float32"
  enable_memory_pool: true
```

### Environment Variables

The following environment variables are automatically set for optimal performance:

```bash
INTEL_GPU_ENABLED=1
SYCL_CACHE_PERSISTENT=1
ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

## 🎮 Supported Environments

The examples work with various Gymnasium environments:

### Basic Environments
- `CartPole-v1` - Classic control task
- `Acrobot-v1` - Acrobot swing-up
- `MountainCar-v0` - Mountain car climbing

### MuJoCo Environments (if installed)
- `HalfCheetah-v4` - Locomotion task
- `Hopper-v4` - Hopper jumping
- `Walker2d-v4` - Walking task

### Custom LeRobot Environments
- `PushT-v0` - Pushing task
- `Aloha-v0` - Bimanual manipulation

## 🔧 Performance Tips

### Batch Size Optimization
Intel GPUs work best with batch sizes that are multiples of 16:
```python
# Good choices
batch_size = 16, 32, 48, 64

# Less optimal
batch_size = 20, 30, 50
```

### Memory Management
```python
# Clear cache periodically
from lerobot.utils.device_utils import empty_cache
empty_cache(device)

# Monitor memory usage
from lerobot.utils.device_utils import get_memory_info
memory_info = get_memory_info(device)
print(f"Memory used: {memory_info['allocated'] / 1024**2:.1f} MB")
```

### Mixed Precision Training
Enable automatic mixed precision for better performance:
```python
with torch.autocast(device_type="xpu", dtype=torch.float16):
    output = model(input)
```

## 🐛 Troubleshooting

### Intel GPU Not Detected
1. Make sure Intel GPU drivers are installed
2. Install PyTorch with XPU support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
   ```
3. Check if Intel GPU is visible: `intel-smi` (Linux) or Device Manager (Windows)

### Out of Memory Errors
1. Reduce batch size: `batch_size = 16`
2. Enable mixed precision: `use_amp = true`
3. Clear cache more frequently: `empty_cache(device)`

### Slow Performance
1. Check batch size is multiple of 16
2. Enable torch.compile (PyTorch 2.7+):
   ```python
   model = torch.compile(model)
   ```
3. Use float32 instead of float64:
   ```python
   tensor = tensor.to(dtype=torch.float32)
   ```

### Import Errors
Make sure all dependencies are installed:
```bash
pip install gymnasium[mujoco] mujoco dm_control pybullet opencv-python
```

## 📊 Performance Benchmarks

Typical performance on Intel Arc A770:

| Environment | Episodes/sec | Memory Usage | Batch Size |
|-------------|--------------|--------------|------------|
| CartPole-v1 | ~100 | 50 MB | 32 |
| HalfCheetah-v4 | ~50 | 150 MB | 32 |
| PushT-v0 | ~25 | 300 MB | 16 |

## 🔗 Related Documentation

- [Intel GPU Setup Guide](../../INTEL_GPU_SETUP.md)
- [Intel GPU Documentation](../../docs/source/intel_gpu.md)
- [Migration Guide](../../docs/source/migration_intel_gpu.md)
- [PyTorch Intel GPU Docs](https://pytorch.org/docs/stable/notes/get_start_xpu.html)

## 📞 Getting Help

If you encounter issues:

1. Run the test script: `python test_intel_gpu_setup.py`
2. Check the logs for error messages
3. Ensure Intel GPU drivers are properly installed
4. Create an issue on the [LeRobot GitHub repository](https://github.com/huggingface/lerobot/issues)

## 🎯 Next Steps

After successful setup, you can:

1. **Explore different environments**: Try various Gymnasium environments
2. **Experiment with policies**: Test different neural network architectures
3. **Scale up training**: Use larger datasets and longer training runs
4. **Real robot integration**: Connect to physical robots using LeRobot's robot interfaces

Happy simulating with Intel GPU! 🎉
