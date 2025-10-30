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

"""Benchmark script for PI05Pytorch inference and training performance."""

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
from transformers import AutoProcessor

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
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch, pad_vector

PROCESSOR = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224", trust_remote_code=True)
DEFAULT_TASK = "Pick up the cube"
BINS = np.linspace(-1, 1, 256 + 1)[:-1]


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
    config: PI05Config,
    batch_size: int,
    task: str,
    device: str,
    state_dim: int,
    action_dim: int,
    mode: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Build synthetic inputs that mimic the policy pipeline.
    """
    height, width = config.image_resolution

    states = torch.rand(batch_size, state_dim, dtype=torch.float32) * 2.0 - 1.0
    states = pad_vector(states, config.max_state_dim)
    discretized_states = np.digitize(states.cpu().numpy(), bins=BINS) - 1

    prompts = []
    for state_row in discretized_states:
        state_str = " ".join(map(str, state_row.tolist()))
        prompts.append(f"Task: {task}, State: {state_str};\nAction: ")

    images = []
    for _ in range(batch_size):
        array = (np.random.rand(height, width, 3) * 255).astype("uint8")
        images.append(Image.fromarray(array))

    processor_outputs = PROCESSOR(text=prompts, images=images, return_tensors="pt")

    tokens = processor_outputs["input_ids"].to(device)
    attention_mask = processor_outputs["attention_mask"].to(device=device).to(dtype=torch.bool)
    pixel_values = processor_outputs["pixel_values"]

    if pixel_values.ndim == 5:
        image_tensors = [pixel_values[:, i].to(device=device, dtype=torch.float32) for i in range(pixel_values.shape[1])]
    else:
        image_tensors = [pixel_values.to(device=device, dtype=torch.float32)]

    img_masks = [torch.ones(batch_size, dtype=torch.bool, device=device) for _ in image_tensors]

    actions = None
    if mode == "training":
        actions = torch.randn(batch_size, config.chunk_size, action_dim, dtype=torch.float32, device=device)
        actions = pad_vector(actions, config.max_action_dim)

    return image_tensors, img_masks, tokens, attention_mask, actions


def run_iteration(
    *,
    model: PI05Pytorch,
    images: list[torch.Tensor],
    img_masks: list[torch.Tensor],
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    actions: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    if mode == "inference":
        return model.sample_actions(images, img_masks, tokens, attention_mask)
    return model.forward(images, img_masks, tokens, attention_mask, actions)


def run_benchmark(
    *,
    device: str,
    precision: str,
    paligemma_variant: str,
    action_expert_variant: str,
    num_runs: int,
    num_warmup: int,
    batch_size: int,
    chunk_size: int,
    state_dim: int,
    max_state_dim: int,
    action_dim: int,
    max_action_dim: int,
    num_inference_steps: int,
    task: str,
    compile_model: bool,
    compile_mode: str,
    mode: str,
) -> None:
    print(f"----- Running PI05 Benchmark on {device.upper()} -----")
    print(
        f"Precision: {precision}, Mode: {mode}, Batch Size: {batch_size}, Runs: {num_runs} (Warmup: {num_warmup}), "
        f"Chunk Size: {chunk_size}, Inference Steps: {num_inference_steps}"
    )

    if not is_device_available(device):
        print(f"{device.upper()} not available. Skipping benchmark.\n")
        return

    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32

    if state_dim > max_state_dim:
        raise ValueError(f"state_dim ({state_dim}) cannot exceed max_state_dim ({max_state_dim})")
    if action_dim > max_action_dim:
        raise ValueError(f"action_dim ({action_dim}) cannot exceed max_action_dim ({max_action_dim})")

    config = PI05Config(
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
        dtype=precision,
        chunk_size=chunk_size,
        n_action_steps=min(chunk_size, 1),
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        num_inference_steps=num_inference_steps,
        compile_model=compile_model,
        compile_mode=compile_mode,
        device=device,
    )

    config.input_features = {
        "observation.images.base_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, *config.image_resolution),
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
    config.validate_features()

    model = PI05Pytorch(config)
    model.eval()
    model.to(device=device)

    images, img_masks, tokens, attention_mask, actions = prepare_dummy_batch(
        config=config,
        batch_size=batch_size,
        task=task,
        device=device,
        state_dim=state_dim,
        action_dim=action_dim,
        mode=mode,
    )

    # Clamp token ids to the embedding vocabulary supported by the policy to avoid OOB indices.
    vocab_size = model.paligemma_with_expert.paligemma.language_model.embed_tokens.num_embeddings
    tokens = tokens.clamp(max=vocab_size - 1)

    # Autocast helps when testing bfloat16 on supported devices.
    ctx_manager = (
        torch.autocast(device_type=device, dtype=dtype) if precision == "bfloat16" and device != "cpu" else nullcontext()
    )

    with torch.no_grad(), ctx_manager:
        for _ in range(num_warmup):
            _ = run_iteration(
                model=model,
                images=images,
                img_masks=img_masks,
                tokens=tokens,
                attention_mask=attention_mask,
                actions=actions,
                mode=mode,
            )

    if device in {"cuda", "xpu"}:
        synchronize(device)

    print("Starting benchmark...")
    start_time = time.perf_counter()

    with torch.no_grad(), ctx_manager:
        for run_idx in range(num_runs):
            _ = run_iteration(
                model=model,
                images=images,
                img_masks=img_masks,
                tokens=tokens,
                attention_mask=attention_mask,
                actions=actions,
                mode=mode,
            )
            print(f"Run {run_idx + 1}/{num_runs}", end="\r")

    if device in {"cuda", "xpu"}:
        synchronize(device)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    avg_time = total_time / max(1, num_runs)
    throughput = (batch_size * num_runs) / total_time if total_time > 0 else float("inf")

    print("\n----- Benchmark Results -----")
    print(f"Device: {device.upper()}")
    print(f"Total time for {num_runs} runs: {total_time:.3f} seconds")
    print(f"Average time per run: {avg_time * 1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/second")
    print("-----------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PI05Pytorch performance across devices.")
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
        default="float32",
        choices=["float32", "bfloat16"],
        help="Computation precision.",
    )
    parser.add_argument("--num-runs", type=int, default=50, help="Number of timed runs.")
    parser.add_argument("--num-warmup", type=int, default=5, help="Number of warmup runs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for synthetic inputs.")
    parser.add_argument("--chunk-size", type=int, default=50, help="Number of action steps in each chunk.")
    parser.add_argument("--state-dim", type=int, default=14, help="Actual state dimension before padding.")
    parser.add_argument("--max-state-dim", type=int, default=32, help="Padded state dimension used in config.")
    parser.add_argument("--action-dim", type=int, default=7, help="Actual action dimension before padding.")
    parser.add_argument("--max-action-dim", type=int, default=32, help="Padded action dimension used in config.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=10,
        help="Number of denoising steps for inference.",
    )
    parser.add_argument(
        "--paligemma-variant",
        type=str,
        default="gemma_2b",
        choices=["gemma_300m", "gemma_2b"],
        help="Variant for the VLM encoder.",
    )
    parser.add_argument(
        "--action-expert-variant",
        type=str,
        default="gemma_300m",
        choices=["gemma_300m", "gemma_2b"],
        help="Variant for the action expert decoder.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Synthetic task prompt used in benchmarking.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the sample_actions method.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune",
        help="torch.compile mode (only used when --compile is set).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="inference",
        choices=["inference", "training"],
        help="Benchmark inference (sample_actions) or training forward pass.",
    )

    args = parser.parse_args()

    for device in args.devices:
        try:
            run_benchmark(
                device=device,
                precision=args.precision,
                paligemma_variant=args.paligemma_variant,
                action_expert_variant=args.action_expert_variant,
                num_runs=args.num_runs,
                num_warmup=args.num_warmup,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                state_dim=args.state_dim,
                max_state_dim=args.max_state_dim,
                action_dim=args.action_dim,
                max_action_dim=args.max_action_dim,
                num_inference_steps=args.num_inference_steps,
                task=args.task,
                compile_model=args.compile,
                compile_mode=args.compile_mode,
                mode=args.mode,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"An error occurred while benchmarking on {device}: {exc}")
            print("Skipping to next device.\n")
