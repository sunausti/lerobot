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

"""Benchmark script for SmolVLA inference and training performance."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

log_level = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)
# Add the 'src' directory to the Python path so we can import lerobot modules
lerobot_src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(lerobot_src_path))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE

DEFAULT_TASK = "Pick up the cube"


def is_device_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    raise ValueError(f"Unsupported device: {device}")


def synchronize(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def prepare_dummy_batch(
    *,
    config: SmolVLAConfig,
    processor,
    batch_size: int,
    task: str,
    device: str,
    state_dim: int,
    action_dim: int,
    mode: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Build synthetic inputs that mimic the policy pipeline.
    """
    height, width = config.resize_imgs_with_padding

    # Prepare state
    states = torch.rand(batch_size, state_dim, dtype=torch.float32, device=device) * 2.0 - 1.0
    # Note: SmolVLA policy handles padding internally in prepare_state if needed, 
    # but here we simulate the input batch which usually has the raw state.
    # However, the policy expects specific keys.
    
    # Prepare images
    images = []
    for _ in range(batch_size):
        array = (np.random.rand(height, width, 3) * 255).astype("uint8")
        images.append(Image.fromarray(array))

    # Prepare text prompts
    # SmolVLM expects specific tokens in the prompt to indicate where the image goes.
    # The error "The number of images in the text [0] and images [1] should be the same"
    # indicates we need to include <image> tokens in the prompt.
    prompts = [f"<image>{task}" for _ in range(batch_size)]
    
    # Use processor to get language tokens
    # SmolVLA expects OBS_LANGUAGE_TOKENS and OBS_LANGUAGE_ATTENTION_MASK in batch
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    
    batch = {}
    
    # Map processor outputs to batch keys expected by SmolVLA
    # Note: The exact keys depend on how the dataset/processor produces them.
    # Looking at modeling_smolvla.py:
    # lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    # lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
    
    batch[OBS_LANGUAGE_TOKENS] = inputs["input_ids"].to(device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = inputs["attention_mask"].to(device)
    
    # Images
    # Use the image keys defined in the config
    if hasattr(config, "image_features") and config.image_features:
        for image_key in config.image_features.keys():
            batch[image_key] = torch.rand(batch_size, 3, height, width, device=device)
    else:
        # Fallback if no image features found (shouldn't happen with proper config)
        image_key = "observation.images.top"
        batch[image_key] = torch.rand(batch_size, 3, height, width, device=device)
    
    # State
    batch[OBS_STATE] = states
    
    # Action (for training)
    if mode == "train":
        actions = torch.rand(batch_size, action_dim, dtype=torch.float32, device=device)
        batch[ACTION] = actions
    
    return batch


def run_benchmark(
    args: argparse.Namespace,
    device: str,
) -> None:
    if not is_device_available(device):
        logging.warning(f"Device {device} is not available. Skipping benchmark.")
        return

    logging.info(f"Running benchmark on {device}...")
    
    if args.policy_path:
        logging.info(f"Loading policy from {args.policy_path}...")
        policy = SmolVLAPolicy.from_pretrained(args.policy_path)
        config = policy.config
    else:
        logging.info("Initializing fresh policy from default config...")
        config = SmolVLAConfig()
        # Update config device
        # config.device is used in SmolVLMWithExpertModel
        config.device = device
        
        # Also set image features in config so prepare_images works
        config.image_features = {"observation.images.top": FeatureType.VISUAL}
        config.max_state_dim = args.state_dim
        config.max_action_dim = args.action_dim
        
        # Initialize policy
        # We use 'meta' device to avoid loading weights if possible, but SmolVLA loads VLM weights by default.
        # We should probably disable loading weights for benchmark to be faster/lighter if we just want to test throughput.
        config.load_vlm_weights = False 
        
        policy = SmolVLAPolicy(config)

    policy.to(device)
    policy.eval()

    # Get processor from policy
    processor = policy.model.vlm_with_expert.processor

    # Prepare batch
    batch = prepare_dummy_batch(
        config=config,
        processor=processor,
        batch_size=args.batch_size,
        task=DEFAULT_TASK,
        device=device,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        mode="inference",
    )

    # Warmup
    logging.info("Warming up...")
    for _ in range(args.warmup_steps):
        with torch.no_grad():
            policy.select_action(batch)
    synchronize(device)

    # Benchmark Inference
    logging.info("Benchmarking inference...")
    start_time = time.time()
    for _ in range(args.num_steps):
        with torch.no_grad():
            policy.select_action(batch)
    synchronize(device)
    end_time = time.time()

    avg_time = (end_time - start_time) / args.num_steps
    fps = args.batch_size / avg_time
    logging.info(f"Inference - Batch Size: {args.batch_size}, Latency: {avg_time*1000:.2f} ms, FPS: {fps:.2f}")

    # Benchmark Training (Forward pass)
    if args.include_training:
        policy.train()
        batch_train = prepare_dummy_batch(
            config=config,
            processor=processor,
            batch_size=args.batch_size,
            task=DEFAULT_TASK,
            device=device,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            mode="train",
        )
        
        # Warmup
        for _ in range(args.warmup_steps):
            policy(batch_train)
        synchronize(device)

        logging.info("Benchmarking training (forward pass)...")
        start_time = time.time()
        for _ in range(args.num_steps):
            policy(batch_train)
        synchronize(device)
        end_time = time.time()

        avg_time = (end_time - start_time) / args.num_steps
        fps = args.batch_size / avg_time
        logging.info(f"Training - Batch Size: {args.batch_size}, Latency: {avg_time*1000:.2f} ms, FPS: {fps:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SmolVLA")
    parser.add_argument("--policy-path", type=str, default="lerobot/smolvla_base", help="Path to pretrained policy (default: lerobot/smolvla_base)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--state-dim", type=int, default=14, help="State dimension")
    parser.add_argument("--action-dim", type=int, default=14, help="Action dimension")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of steps for benchmark")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "xpu"], help="Device to run on")
    parser.add_argument("--include-training", action="store_true", help="Include training benchmark")
    
    args = parser.parse_args()

    run_benchmark(args, args.device)


if __name__ == "__main__":
    main()
