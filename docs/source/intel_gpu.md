# Intel GPU Support for LeRobot

LeRobot supports Intel GPUs through PyTorch's XPU backend, enabling you to train and run robotic policies on Intel Arc GPUs and Intel Data Center GPUs.

## Prerequisites

### Hardware Requirements

**Intel Client GPUs:**
- Intel Arc A-Series Graphics (Alchemist)
- Intel Arc B-Series Graphics (Battlemage) 
- Intel Core Ultra Processors with Intel Arc Graphics (Meteor Lake-H)
- Intel Core Ultra Desktop/Mobile Processors (Series 2) with Intel Arc Graphics

**Intel Data Center GPUs:**
- Intel Data Center GPU Max Series (Ponte Vecchio)

### Software Requirements

1. **Operating System**: 
   - Ubuntu 24.04 or later
   - Windows 11
   - WSL2 (Ubuntu 24.04)

2. **Intel GPU Driver**: Install the latest Intel GPU drivers following the [Intel GPU Driver Installation Guide](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html#driver-installation)

## Installation

### Step 1: Install PyTorch with Intel GPU Support

Install PyTorch with XPU support using the Intel GPU wheels:

```bash
# For stable release
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# For nightly release (latest features)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
```

### Step 2: Verify Intel GPU Detection

Check if PyTorch can detect your Intel GPU:

```python
import torch
print(f"Intel GPU available: {torch.xpu.is_available()}")
if torch.xpu.is_available():
    print(f"Intel GPU device count: {torch.xpu.device_count()}")
    print(f"Current Intel GPU: {torch.xpu.get_device_name()}")
```

### Step 3: Install LeRobot

Install LeRobot as usual:

```bash
pip install lerobot
```

Or for development:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

## Usage

### Training with Intel GPU

Use the `--policy.device=xpu` flag to train on Intel GPU:

```bash
lerobot-train \
    --robot.type=so100 \
    --robot.sensors.cameras.laptop.capture_timestep=1.0 \
    --robot.sensors.cameras.phone.capture_timestep=1.0 \
    --dataset.type=image_folder \
    --dataset.dir=data/example_folder \
    --training.save_checkpoint_freq=1000 \
    --training.log_freq=10 \
    --policy.device=xpu
```

### Evaluation with Intel GPU

Evaluate trained models on Intel GPU:

```bash
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.device=xpu
```

### Programming with Intel GPU

In your Python scripts, specify Intel GPU device:

```python
import torch
from lerobot.policies import DiffusionPolicy

# Check Intel GPU availability
if torch.xpu.is_available():
    device = "xpu"
    print("Using Intel GPU")
else:
    device = "cpu"
    print("Intel GPU not available, falling back to CPU")

# Load policy
policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")
policy = policy.to(device)

# Use with automatic mixed precision for better performance
with torch.autocast(device_type="xpu", dtype=torch.float16):
    output = policy(observation)
```

## Performance Optimization

### Mixed Precision Training

Intel GPUs support automatic mixed precision for faster training:

```bash
lerobot-train \
    --policy.device=xpu \
    --policy.use_amp=true \
    ...
```

### Memory Management

Intel GPUs typically have less memory than high-end NVIDIA GPUs. Consider:

- Reducing batch size: `--batch_size=8` (instead of 32)
- Using gradient accumulation: Split large batches across multiple steps
- Using FP16 precision: `--policy.use_amp=true`

### Batch Size Recommendations

Intel GPUs work optimally with batch sizes that are multiples of 16:

```bash
# Good batch sizes for Intel GPU
--batch_size=16    # Small models
--batch_size=32    # Medium models  
--batch_size=8     # Large models or limited memory
```

## Supported Features

✅ **Supported on Intel GPU:**
- Policy training and inference
- Automatic Mixed Precision (AMP)
- `torch.compile` optimization (PyTorch 2.7+)
- All LeRobot policy types (ACT, Diffusion, VQ-BeT, etc.)
- Data loading with pinned memory
- Multi-episode training

⚠️ **Limitations:**
- Some Intel Arc A-Series GPUs don't support FP64 operations (use FP32/FP16)
- Memory capacity may be lower than high-end NVIDIA GPUs
- Performance optimization is still evolving

## Troubleshooting

### Intel GPU Not Detected

1. **Check driver installation:**
   ```bash
   intel-smi  # Should show your Intel GPU
   ```

2. **Verify PyTorch XPU support:**
   ```python
   import torch
   print(hasattr(torch, 'xpu'))
   print(torch.xpu.is_available() if hasattr(torch, 'xpu') else False)
   ```

3. **Update PyTorch:**
   ```bash
   pip install --upgrade torch --index-url https://download.pytorch.org/whl/xpu
   ```

### Out of Memory Errors

1. **Reduce batch size:**
   ```bash
   --batch_size=8  # or smaller
   ```

2. **Enable gradient checkpointing** (if supported by the policy)

3. **Use mixed precision:**
   ```bash
   --policy.use_amp=true
   ```

### Performance Issues

1. **Use optimal batch sizes** (multiples of 16)
2. **Enable torch.compile** (PyTorch 2.7+):
   ```python
   policy = torch.compile(policy)
   ```
3. **Profile your code** to identify bottlenecks

## Environment Variables

Set these environment variables for optimal performance:

```bash
# Enable Intel GPU optimizations
export SYCL_CACHE_PERSISTENT=1

# Set memory pool settings (adjust based on available GPU memory)
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

## Examples

### Complete Training Example

```python
import torch
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.utils.device_utils import log_device_info

# Set device
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
log_device_info(device)

# Load dataset
dataset = make_dataset("pusht", train=True)

# Create policy
policy = make_policy("diffusion", device=device)

# Training loop with Intel GPU optimizations
for batch in dataset:
    # Move batch to Intel GPU with non-blocking transfer
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    
    # Use autocast for mixed precision
    with torch.autocast(device_type="xpu", dtype=torch.float16):
        loss = policy.compute_loss(batch)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Reporting Issues

If you encounter issues with Intel GPU support:

1. Check the [Intel GPU PyTorch documentation](https://pytorch.org/docs/stable/notes/get_start_xpu.html)
2. Report bugs to the [LeRobot GitHub repository](https://github.com/huggingface/lerobot/issues)
3. Include your system information:
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"XPU available: {torch.xpu.is_available()}")
   print(f"XPU device count: {torch.xpu.device_count()}")
   ```

## Further Reading

- [PyTorch Intel GPU Documentation](https://pytorch.org/docs/stable/notes/get_start_xpu.html)
- [Intel GPU Driver Installation](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html)
- [Intel AI Tools for PyTorch](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
