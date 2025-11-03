# Groot Policy Intel GPU (XPU) Benchmark

This benchmark script tests the Groot policy inference and training performance on different devices (CPU, Intel GPU/XPU, CUDA).

## Features

- **Multi-device support**: CPU, Intel GPU (XPU via Level Zero), CUDA
- **Precision options**: float32, bfloat16
- **Modes**: Inference (predict_action_chunk) and Training (forward pass)
- **Synthetic data generation**: Creates dummy batches matching Groot's expected input format

## Requirements

- PyTorch with Intel GPU support (for XPU benchmarking)
- Transformers library
- Groot policy implementation with Intel GPU support

## Usage

### Basic Inference Benchmark on All Devices

```bash
python benchmarks/groot/benchmark.py --devices cpu xpu cuda --num-runs 10 --num-warmup 2
```

### Intel GPU (XPU) Only

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu python benchmarks/groot/benchmark.py --devices xpu --num-runs 10
```

### CPU Benchmark with Float32

```bash
python benchmarks/groot/benchmark.py --devices cpu --precision float32 --num-runs 50
```

### Training Mode Benchmark

```bash
python benchmarks/groot/benchmark.py --devices xpu --mode training --num-runs 10
```

## Command-line Arguments

- `--devices`: Devices to benchmark (default: cpu xpu cuda)
- `--precision`: Computation precision (default: bfloat16, choices: float32, bfloat16)
- `--num-runs`: Number of timed runs (default: 10)
- `--num-warmup`: Number of warmup runs (default: 2)
- `--batch-size`: Batch size (default: 1)
- `--chunk-size`: Action horizon/chunk size (default: 50)
- `--state-dim`: Actual state dimension before padding (default: 14)
- `--max-state-dim`: Padded state dimension (default: 64)
- `--action-dim`: Actual action dimension before padding (default: 7)
- `--max-action-dim`: Padded action dimension (default: 32)
- `--task`: Task description (default: "Pick up the cube")
- `--mode`: Benchmark mode (default: inference, choices: inference, training)
- `--base-model-path`: HuggingFace model ID or local path (default: nvidia/GR00T-N1.5-3B)

## Intel GPU Setup

To use Intel GPU (XPU), ensure you have:

1. Intel GPU drivers installed
2. oneAPI Base Toolkit installed
3. PyTorch with Intel GPU support

Set the backend to Level Zero for optimal performance:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
```

## Example Output

```
----- Running Groot Benchmark on XPU -----
Precision: bfloat16, Mode: inference, Batch Size: 1, Runs: 10 (Warmup: 2), Chunk Size: 50
[GROOT Config] Intel GPU (XPU) detected and available
[GROOT Config] Configured Intel GPU to use Level Zero backend
[GROOT Config] Disabled Flash Attention for Intel GPU compatibility
[GROOT Policy] Using device: xpu
[EagleBackbone] Using device: xpu
[EagleBackbone] Configured eager attention for Intel GPU (XPU)
Running warmup...
Starting benchmark...

----- Benchmark Results -----
Device: XPU
Total time for 10 runs: 23.456 seconds
Average time per run: 2345.60 ms
Throughput: 0.43 samples/second
-----------------------------
```

## Implementation Details

### Device Detection

The benchmark automatically detects available devices in priority order:
1. Intel GPU (XPU) - if torch.xpu.is_available()
2. CUDA GPU - if torch.cuda.is_available()
3. CPU - fallback

### Attention Mechanism

For Intel GPU (XPU):
- Flash Attention is **disabled** (not supported on Intel GPU)
- Uses eager attention implementation
- Configured via `TRANSFORMERS_NO_FLASH_ATTN=1` environment variable

### Synchronization

The benchmark properly synchronizes GPU operations:
- `torch.cuda.synchronize()` for CUDA
- `torch.xpu.synchronize()` for Intel GPU
- Ensures accurate timing measurements

### Synthetic Data

The benchmark generates synthetic inputs matching Groot's expected format:
- **With Eagle Processor** (recommended): Uses the actual Eagle/Groot processor to format inputs correctly
  - Properly formatted vision inputs via transformers AutoProcessor
  - Correct attention masks and input IDs
  - Proper image size tracking
- **Fallback mode** (if processor unavailable): Simplified dummy tensors
- Random state vectors (padded to max_state_dim)
- Random embodiment IDs
- Optional action tensors for training mode

**Note**: The benchmark automatically tries to use the Eagle processor from `lerobot/eagle2hg-processor-groot-n1p5`. This ensures inputs match the model's expected format exactly.

## Troubleshooting

### XPU Not Available

If XPU is not detected:
1. Check Intel GPU drivers: `sycl-ls`
2. Verify PyTorch XPU support: `python -c "import torch; print(torch.xpu.is_available())"`
3. Set Level Zero backend: `export ONEAPI_DEVICE_SELECTOR=level_zero:gpu`

### Flash Attention Errors

Flash Attention is not supported on Intel GPU. The implementation automatically:
- Disables Flash Attention via environment variable
- Uses eager attention instead
- Configures transformers models accordingly

### Memory Issues

If you encounter OOM errors:
- Reduce `--batch-size`
- Reduce `--chunk-size`
- Use `--precision float32` instead of bfloat16 (may use less memory on some hardware)

## Performance Notes

- Intel GPU (XPU) performance depends on:
  - oneAPI version
  - PyTorch version with XPU support
  - Level Zero backend configuration
  - Model size and batch size

- BFloat16 precision:
  - Generally faster on modern hardware
  - May not be supported on all CPUs (will fallback to float32)
  - Supported on Intel GPU (XPU) and NVIDIA GPUs

## Related Documentation

- [Intel GPU Implementation Guide](../../docs/source/intel_gpu.md)
- [Groot Policy README](../../src/lerobot/policies/groot/README.md)
- [PI05 Benchmark](../pi05/benchmark.py)
