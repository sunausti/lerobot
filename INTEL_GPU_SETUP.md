# Intel GPU Setup for LeRobot

## Quick Start

### 1. Install Intel GPU Drivers

First, install the Intel GPU drivers for your system:

**Windows:**
- Download and install Intel Arc drivers from [Intel's website](https://www.intel.com/content/www/us/en/support/articles/000096551/graphics.html)

**Linux (Ubuntu 24.04+):**
```bash
# Add Intel GPU driver repository
sudo apt update
sudo apt install -y software-properties-common
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | sudo apt-key add -
sudo apt-add-repository 'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu jammy main'
sudo apt update
sudo apt install intel-opencl-icd
```

### 2. Install PyTorch with Intel GPU Support

Install PyTorch with XPU (Intel GPU) support:

```bash
# Create a new conda environment
conda create -n lerobot-intel python=3.12
conda activate lerobot-intel

# Install PyTorch with Intel GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

### 3. Install LeRobot

```bash
# Clone LeRobot repository
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Install LeRobot
pip install -e .
```

### 4. Verify Installation

Test that Intel GPU support is working:

```python
import torch

# Check if Intel GPU is available
print(f"Intel GPU available: {torch.xpu.is_available()}")
if torch.xpu.is_available():
    print(f"Device count: {torch.xpu.device_count()}")
    print(f"Device name: {torch.xpu.get_device_name()}")
    
    # Test basic operations
    x = torch.randn(10, 10, device='xpu')
    y = torch.randn(10, 10, device='xpu')
    z = torch.matmul(x, y)
    print(f"Test tensor operation successful: {z.shape}")
```

### 5. Use with LeRobot

Now you can use Intel GPU with LeRobot commands:

```bash
# Training with Intel GPU
lerobot-train \
    --dataset.type=your_dataset \
    --policy.type=act \
    --policy.device=xpu \
    --policy.use_amp=true

# Evaluation with Intel GPU
lerobot-eval \
    --policy.path=your_policy \
    --policy.device=xpu
```

## Troubleshooting

### "No module named 'torch'"
Make sure you activated the correct conda environment:
```bash
conda activate lerobot-intel
```

### "Intel GPU not available"
1. Check driver installation: `intel-smi` (Linux) or Device Manager (Windows)
2. Verify PyTorch XPU installation: `python -c "import torch; print(torch.__version__)"`
3. Try reinstalling PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/xpu --force-reinstall`

### Out of Memory Errors
Intel GPUs typically have less memory than high-end NVIDIA GPUs:
1. Reduce batch size: `--batch_size=8`
2. Enable mixed precision: `--policy.use_amp=true`
3. Use gradient accumulation for larger effective batch sizes

### Performance Issues
1. Use optimal batch sizes (multiples of 16 for Intel GPU)
2. Enable torch.compile: `--compile=true` (PyTorch 2.7+)
3. Check Intel GPU utilization with `intel-smi`

## Environment Variables

For optimal performance, set these environment variables:

```bash
# Enable Intel GPU optimizations
export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

# Optional: Set memory pool size (adjust based on GPU memory)
export SYCL_PI_LEVEL_ZERO_MAX_POOL_SIZE=4294967296  # 4GB
```

## Supported Hardware

**Intel Client GPUs:**
- Intel Arc A-Series Graphics (Alchemist)
- Intel Arc B-Series Graphics (Battlemage)
- Intel Core Ultra with Intel Arc Graphics

**Intel Data Center GPUs:**
- Intel Data Center GPU Max Series (Ponte Vecchio)

**Operating Systems:**
- Windows 11
- Ubuntu 24.04+
- WSL2 (Ubuntu 24.04)

For the latest compatibility information, see [PyTorch Intel GPU documentation](https://pytorch.org/docs/stable/notes/get_start_xpu.html).
