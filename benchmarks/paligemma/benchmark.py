#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import argparse
import time
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING

# Add lerobot to Python path
import sys
from pathlib import Path

# Add the 'src' directory to the Python path to allow direct imports
# from 'lerobot'
lerobot_src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(lerobot_src_path))

from lerobot.policies.pi05.modeling_pi05 import get_gemma_config


PROCESSOR = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224", trust_remote_code=True)
DEFAULT_PROMPT = "Describe the image."


def get_pi05_paligemma_config(variant: str = "gemma_2b"):
    """
    Creates a PaliGemma configuration identical to the one used in the PI05 policy.
    """
    vlm_config = get_gemma_config(variant)

    # Create a Hugging Face PaliGemmaConfig object
    vlm_config_hf = CONFIG_MAPPING["paligemma"]()

    # --- Populate the config with values from PI05 ---
    # General VLM settings
    vlm_config_hf._vocab_size = 257216
    vlm_config_hf.image_token_index = 257152

    # Text model config
    vlm_config_hf.text_config.hidden_size = vlm_config.width
    vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
    vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
    vlm_config_hf.text_config.head_dim = vlm_config.head_dim
    vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
    vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
    vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
    vlm_config_hf.text_config.vocab_size = 257216

    # Vision model config (SigLIP)
    vlm_config_hf.vision_config.intermediate_size = 4304
    vlm_config_hf.vision_config.projection_dim = vlm_config.width
    vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
    # Force eager attention for vision tower to avoid potential backend errors on some hardware
    vlm_config_hf.vision_config._attn_implementation = "eager"

    return vlm_config_hf


def run_benchmark(
    device: str,
    paligemma_variant: str,
    num_runs: int,
    num_warmup: int,
    batch_size: int,
    precision: str,
):
    """
    Runs the performance benchmark for PaliGemma on a specified device.
    """
    print(f"----- Running PaliGemma Benchmark on {device.upper()} -----")
    print(
        f"Variant: {paligemma_variant}, Precision: {precision}, Batch Size: {batch_size}, "
        f"Runs: {num_runs} (Warmup: {num_warmup})"
    )

    # 1. Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return
    if device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("XPU (Intel GPU) not available. Skipping benchmark.")
        return

    # 2. Get model configuration and initialize model
    config = get_pi05_paligemma_config(paligemma_variant)
    model = PaliGemmaForConditionalGeneration(config)
    model.vision_tower.config._attn_implementation = "eager"
    model.language_model.config._attn_implementation = "eager"
    model.eval()

    # 3. Set data type and move model to device
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    try:
        model.to(device=device, dtype=dtype)
    except RuntimeError as e:
        print(f"Error moving model to {device} with dtype {dtype}: {e}")
        print(f"This may happen if the device does not support {precision}.")
        print("Skipping benchmark.")
        return

    # 4. Create dummy input data using the official processor to ensure
    # proper placement of the special image tokens.
    processor = PROCESSOR
    processor_size = processor.image_processor.size
    if isinstance(processor_size, dict):
        height = processor_size.get("height") or processor_size.get("shortest_edge") or 224
        width = processor_size.get("width") or processor_size.get("shortest_edge") or height
    elif isinstance(processor_size, (tuple, list)):
        if len(processor_size) == 2:
            height, width = processor_size
        else:
            height = width = processor_size[0]
    else:
        height = width = int(processor_size)

    height = int(height)
    width = int(width)

    texts = [DEFAULT_PROMPT] * batch_size
    images = [
        Image.fromarray((np.random.rand(height, width, 3) * 255).astype("uint8"))
        for _ in range(batch_size)
    ]

    processor_outputs = processor(text=texts, images=images, return_tensors="pt")
    input_ids = processor_outputs["input_ids"].to(device)
    attention_mask = processor_outputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    pixel_values = processor_outputs["pixel_values"].to(device=device, dtype=dtype)

    generate_kwargs = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "max_new_tokens": 50,
        "do_sample": False,
    }
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    # 5. Warmup runs
    print("Warming up...")
    # Use torch.no_grad() for inference and autocast for mixed precision
    ctx_manager = torch.autocast(device_type=device, dtype=dtype) if precision == "bfloat16" else nullcontext()
    with torch.no_grad(), ctx_manager:
        for _ in range(num_warmup):
            _ = model.generate(**generate_kwargs)
    
    # Synchronize if on a GPU device to ensure warmup is complete
    if device in ["cuda", "xpu"]:
        torch.cuda.synchronize() if device == "cuda" else torch.xpu.synchronize()

    # 6. Benchmark runs
    print("Starting benchmark...")
    total_time = 0
    start_time = time.perf_counter()

    with torch.no_grad(), ctx_manager:
        for i in range(num_runs):
            _ = model.generate(**generate_kwargs)
            # Progress indicator
            print(f"Run {i+1}/{num_runs}", end="\r")

    # Synchronize if on a GPU device to get accurate timing
    if device in ["cuda", "xpu"]:
        torch.cuda.synchronize() if device == "cuda" else torch.xpu.synchronize()

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # 7. Report results
    avg_time_per_run = total_time / num_runs
    throughput = num_runs * batch_size / total_time

    print("\n----- Benchmark Results -----")
    print(f"Device: {device.upper()}")
    print(f"Total time for {num_runs} runs: {total_time:.3f} seconds")
    print(f"Average time per run: {avg_time_per_run * 1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} inferences/second")
    print("-----------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PaliGemma performance.")
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cpu", "xpu", "cuda"],
        help="List of devices to benchmark (cpu, xpu, cuda).",
    )
    parser.add_argument(
        "--paligemma-variant",
        type=str,
        default="gemma_2b",
        choices=["gemma_300m", "gemma_2b"],
        help="PaliGemma variant to benchmark (matches PI05 configuration).",
    )
    parser.add_argument("--num-runs", type=int, default=50, help="Number of benchmark runs.")
    parser.add_argument("--num-warmup", type=int, default=5, help="Number of warmup runs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16"],
        help="Precision to use for the model (float32 or bfloat16).",
    )
    args = parser.parse_args()

    for device in args.devices:
        try:
            run_benchmark(
                device,
                args.paligemma_variant,
                args.num_runs,
                args.num_warmup,
                args.batch_size,
                args.precision,
            )
        except Exception as e:
            print(f"An error occurred while benchmarking on {device}: {e}")
            print("Skipping to next device.")
