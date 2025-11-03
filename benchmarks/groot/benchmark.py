#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Benchmark script for Groot policy inference performance on CPU/XPU/CUDA."""

from __future__ import annotations

# CRITICAL: Set environment variables and block flash_attn IMMEDIATELY after __future__
# This must be done BEFORE any transformers/torch imports
import os
import sys

os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
os.environ["DISABLE_FLASH_ATTN"] = "1"
os.environ["FLASH_ATTENTION_SKIP"] = "1"

# Disable PyTorch SDPA (scaled_dot_product_attention) for Intel GPU
# SDPA has compatibility issues on XPU - "could not create a primitive" error
os.environ["PYTORCH_ENABLE_MHA_FUSED_ATTENTION"] = "0"

# Monkey-patch sys.modules to prevent flash_attn import
# This is a nuclear option but necessary for Intel GPU
sys.modules['flash_attn'] = None
sys.modules['flash_attn_interface'] = None
sys.modules['flash_attn.flash_attn_interface'] = None

# Now safe to import other modules
import argparse
import logging
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Monkey-patch torch.nn.functional.scaled_dot_product_attention for Intel GPU
# XPU has issues with SDPA primitive creation, so we replace it with basic attention
_original_sdpa = torch.nn.functional.scaled_dot_product_attention

def _safe_sdpa_for_xpu(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """Fallback implementation of scaled_dot_product_attention for Intel GPU."""
    # Check if we're on XPU
    if query.device.type == 'xpu':
        # Use basic attention implementation
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / (query.size(-1) ** 0.5) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)
        
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        
        if dropout_p > 0.0:
            attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)
        
        return attn_weight @ value
    else:
        # Use original SDPA for CUDA/CPU
        return _original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale)

# Replace the function
torch.nn.functional.scaled_dot_product_attention = _safe_sdpa_for_xpu

log_level = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

# Add the 'src' directory to the Python path so we can import lerobot modules
lerobot_src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(lerobot_src_path))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy

# Import Eagle processor for proper input formatting
try:
    from transformers import AutoProcessor
    PROCESSOR_AVAILABLE = True
except ImportError:
    PROCESSOR_AVAILABLE = False
    print("Warning: transformers AutoProcessor not available, will use simplified batch generation")

DEFAULT_TASK = "Pick up the cube"
DEFAULT_TOKENIZER_ASSETS_REPO = "lerobot/eagle2hg-processor-groot-n1p5"


def is_device_available(device: str) -> bool:
    """Check if a device is available for benchmarking."""
    if device == "cpu":
        return True
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    raise ValueError(f"Unsupported device: {device}")


def synchronize(device: str) -> None:
    """Synchronize GPU operations to ensure accurate timing."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def prepare_dummy_batch_with_processor(
    *,
    config: GrootConfig,
    batch_size: int,
    task: str,
    device: str,
    state_dim: int,
    action_dim: int,
    mode: str,
) -> dict[str, torch.Tensor]:
    """
    Build synthetic inputs using the Eagle processor.
    
    This uses the actual Eagle/Groot processor to generate properly formatted inputs.
    """
    if not PROCESSOR_AVAILABLE:
        raise RuntimeError("AutoProcessor not available. Install transformers library.")
    
    height, width = config.image_size
    
    # Create random state (B, T=1, state_dim) and pad to max_state_dim
    # Groot expects 3D state tensor with time dimension
    states = torch.rand(batch_size, 1, state_dim, dtype=torch.float32) * 2.0 - 1.0
    if state_dim < config.max_state_dim:
        padding = torch.zeros(batch_size, 1, config.max_state_dim - state_dim, dtype=torch.float32)
        states = torch.cat([states, padding], dim=2)
    
    # State mask: (B, T=1)
    state_mask = torch.ones(batch_size, 1, dtype=torch.bool)
    
    # Create random embodiment IDs
    embodiment_ids = torch.randint(0, 1, (batch_size,), dtype=torch.long)
    
    # Create synthetic PIL images
    images = []
    for _ in range(batch_size):
        array = (np.random.rand(height, width, 3) * 255).astype("uint8")
        images.append(Image.fromarray(array))
    
    # Use Eagle processor to format images properly
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME
        cache_dir = HF_LEROBOT_HOME / DEFAULT_TOKENIZER_ASSETS_REPO
        processor = AutoProcessor.from_pretrained(str(cache_dir), trust_remote_code=True)
    except Exception:
        # Fallback to loading from HF hub
        processor = AutoProcessor.from_pretrained(DEFAULT_TOKENIZER_ASSETS_REPO, trust_remote_code=True)
    
    # Process images with text prompts using Eagle processor
    texts = [task] * batch_size
    processor_output = processor(
        text=texts,
        images=images,
        images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False},
        return_tensors="pt",
        padding=True,
    )
    
    # Debug: print available keys
    print(f"[DEBUG] Processor output keys: {processor_output.keys()}")
    
    # Build batch with processed Eagle inputs
    batch = {
        "state": states.to(device),
        "state_mask": state_mask.to(device),
        "embodiment_id": embodiment_ids.to(device),
    }
    
    # Add processor outputs with eagle_ prefix
    # Handle different possible key names from the processor
    for key in processor_output.keys():
        if key in ["input_ids", "attention_mask", "pixel_values", "image_sizes", "image_grid_thw"]:
            batch[f"eagle_{key}"] = processor_output[key].to(device)
    
    # Verify required keys are present
    required_keys = ["eagle_input_ids", "eagle_attention_mask", "eagle_pixel_values"]
    missing_keys = [k for k in required_keys if k not in batch]
    if missing_keys:
        raise RuntimeError(f"Processor did not generate required keys: {missing_keys}")
    
    # Add actions and action_mask for training mode
    if mode == "training":
        actions = torch.randn(batch_size, config.chunk_size, action_dim, dtype=torch.float32)
        if action_dim < config.max_action_dim:
            padding = torch.zeros(batch_size, config.chunk_size, config.max_action_dim - action_dim, dtype=torch.float32)
            actions = torch.cat([actions, padding], dim=2)
        
        action_mask = torch.ones(batch_size, config.chunk_size, dtype=torch.bool)
        
        batch["action"] = actions.to(device)
        batch["action_mask"] = action_mask.to(device)
    
    return batch


def prepare_dummy_batch(
    *,
    config: GrootConfig,
    batch_size: int,
    task: str,
    device: str,
    state_dim: int,
    action_dim: int,
    mode: str,
) -> dict[str, torch.Tensor]:
    """
    Build synthetic inputs that mimic the Groot policy pipeline.
    
    Tries to use Eagle processor if available, otherwise falls back to simplified generation.
    """
    if PROCESSOR_AVAILABLE:
        try:
            return prepare_dummy_batch_with_processor(
                config=config,
                batch_size=batch_size,
                task=task,
                device=device,
                state_dim=state_dim,
                action_dim=action_dim,
                mode=mode,
            )
        except Exception as e:
            print(f"Warning: Failed to use Eagle processor: {e}")
            print("Falling back to simplified batch generation")
    
    # Simplified fallback (may not work with actual model)
    height, width = config.image_size
    
    # Create random state with time dimension (batch_size, 1, state_dim)
    states = torch.rand(batch_size, 1, state_dim, dtype=torch.float32) * 2.0 - 1.0
    if state_dim < config.max_state_dim:
        padding = torch.zeros(batch_size, 1, config.max_state_dim - state_dim, dtype=torch.float32)
        states = torch.cat([states, padding], dim=2)
    
    state_mask = torch.ones(batch_size, 1, dtype=torch.bool)
    embodiment_ids = torch.randint(0, 1, (batch_size,), dtype=torch.long)
    
    # Create dummy Eagle inputs (simplified - may not work)
    seq_len = 256
    batch = {
        "state": states.to(device),
        "state_mask": state_mask.to(device),
        "embodiment_id": embodiment_ids.to(device),
        "eagle_input_ids": torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long).to(device),
        "eagle_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long).to(device),
        "eagle_pixel_values": torch.randn(batch_size, 3, height, width).to(device),
    }
    
    if mode == "training":
        actions = torch.randn(batch_size, config.chunk_size, action_dim, dtype=torch.float32)
        if action_dim < config.max_action_dim:
            padding = torch.zeros(batch_size, config.chunk_size, config.max_action_dim - action_dim, dtype=torch.float32)
            actions = torch.cat([actions, padding], dim=2)
        
        action_mask = torch.ones(batch_size, config.chunk_size, dtype=torch.bool)
        batch["action"] = actions.to(device)
        batch["action_mask"] = action_mask.to(device)
    
    return batch


def run_iteration(
    *,
    model: GrootPolicy,
    batch: dict[str, torch.Tensor],
    mode: str,
) -> torch.Tensor:
    """Run a single forward pass (inference or training)."""
    if mode == "inference":
        return model.predict_action_chunk(batch)
    else:
        loss, loss_dict = model.forward(batch)
        return loss


def run_benchmark(
    *,
    device: str,
    precision: str,
    num_runs: int,
    num_warmup: int,
    batch_size: int,
    chunk_size: int,
    state_dim: int,
    max_state_dim: int,
    action_dim: int,
    max_action_dim: int,
    task: str,
    mode: str,
    base_model_path: str,
) -> None:
    """Run benchmark for Groot policy on specified device."""
    print(f"\n----- Running Groot Benchmark on {device.upper()} -----")
    print(
        f"Precision: {precision}, Mode: {mode}, Batch Size: {batch_size}, "
        f"Runs: {num_runs} (Warmup: {num_warmup}), Chunk Size: {chunk_size}"
    )

    if not is_device_available(device):
        print(f"{device.upper()} not available. Skipping benchmark.\n")
        return

    if state_dim > max_state_dim:
        raise ValueError(f"state_dim ({state_dim}) cannot exceed max_state_dim ({max_state_dim})")
    if action_dim > max_action_dim:
        raise ValueError(f"action_dim ({action_dim}) cannot exceed max_action_dim ({max_action_dim})")

    # Create Groot configuration
    config = GrootConfig(
        chunk_size=chunk_size,
        n_action_steps=min(chunk_size, 50),
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        use_bf16=(precision == "bfloat16"),
        device=device,
        base_model_path=base_model_path,
        # Tuning settings for inference benchmark (all frozen)
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
    )

    config.input_features = {
        "observation.images.base_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, *config.image_size),
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(max_state_dim,),
        ),
    }
    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(max_action_dim,),
        )
    }

    print(f"Creating Groot policy with config device={config.device}")
    try:
        model = GrootPolicy(config)
        model.eval()
    except Exception as e:
        print(f"Failed to create Groot policy: {e}")
        import traceback
        traceback.print_exc()
        return

    # Prepare dummy batch
    try:
        batch = prepare_dummy_batch(
            config=config,
            batch_size=batch_size,
            task=task,
            device=device,
            state_dim=state_dim,
            action_dim=action_dim,
            mode=mode,
        )
    except Exception as e:
        print(f"Failed to prepare dummy batch: {e}")
        import traceback
        traceback.print_exc()
        return

    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32

    # Autocast context for bfloat16
    ctx_manager = (
        torch.autocast(device_type=device, dtype=dtype)
        if precision == "bfloat16" and device != "cpu"
        else nullcontext()
    )

    # Warmup runs
    print("Running warmup...")
    with torch.no_grad(), ctx_manager:
        for i in range(num_warmup):
            try:
                _ = run_iteration(model=model, batch=batch, mode=mode)
                print(f"Warmup {i+1}/{num_warmup}", end="\r")
            except Exception as e:
                print(f"\nWarmup iteration {i+1} failed: {e}")
                import traceback
                traceback.print_exc()
                return

    if device in {"cuda", "xpu"}:
        synchronize(device)

    # Benchmark runs
    print("\nStarting benchmark...")
    start_time = time.perf_counter()

    with torch.no_grad(), ctx_manager:
        for run_idx in range(num_runs):
            try:
                _ = run_iteration(model=model, batch=batch, mode=mode)
                print(f"Run {run_idx + 1}/{num_runs}", end="\r")
            except Exception as e:
                print(f"\nRun {run_idx + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                return

    if device in {"cuda", "xpu"}:
        synchronize(device)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    avg_time = total_time / max(1, num_runs)
    throughput = (batch_size * num_runs) / total_time if total_time > 0 else float("inf")

    print("\n\n----- Benchmark Results -----")
    print(f"Device: {device.upper()}")
    print(f"Total time for {num_runs} runs: {total_time:.3f} seconds")
    print(f"Average time per run: {avg_time * 1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/second")
    print("-----------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Groot policy performance across devices.")
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cpu", "xpu", "cuda"],
        help="Devices to benchmark (cpu, xpu, cuda).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16"],
        help="Computation precision.",
    )
    parser.add_argument("--num-runs", type=int, default=10, help="Number of timed runs.")
    parser.add_argument("--num-warmup", type=int, default=2, help="Number of warmup runs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for synthetic inputs.")
    parser.add_argument("--chunk-size", type=int, default=50, help="Number of action steps in each chunk.")
    parser.add_argument("--state-dim", type=int, default=14, help="Actual state dimension before padding.")
    parser.add_argument("--max-state-dim", type=int, default=64, help="Padded state dimension used in config.")
    parser.add_argument("--action-dim", type=int, default=7, help="Actual action dimension before padding.")
    parser.add_argument("--max-action-dim", type=int, default=32, help="Padded action dimension used in config.")
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Synthetic task prompt used in benchmarking.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="inference",
        choices=["inference", "training"],
        help="Benchmark inference (predict_action_chunk) or training forward pass.",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="nvidia/GR00T-N1.5-3B",
        help="HuggingFace model ID or local path for pretrained Groot model.",
    )

    args = parser.parse_args()

    for device in args.devices:
        try:
            run_benchmark(
                device=device,
                precision=args.precision,
                num_runs=args.num_runs,
                num_warmup=args.num_warmup,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                state_dim=args.state_dim,
                max_state_dim=args.max_state_dim,
                action_dim=args.action_dim,
                max_action_dim=args.max_action_dim,
                task=args.task,
                mode=args.mode,
                base_model_path=args.base_model_path,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"An error occurred while benchmarking on {device}: {exc}")
            import traceback
            traceback.print_exc()
            print("Skipping to next device.\n")
