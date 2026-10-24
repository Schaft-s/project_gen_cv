#!/usr/bin/env python3
"""S1: Diffusers DDIM baseline on google/ddpm-cifar10-32."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from PIL import Image

from gencv.device import get_device, sync
from gencv.paths import OUTPUTS, load_default_config


def save_grid(arr: np.ndarray, path: Path, ncol: int = 4) -> None:
    n, h, w, c = arr.shape
    ncol = min(ncol, n)
    nrow = (n + ncol - 1) // ncol
    grid = np.ones((nrow * h, ncol * w, c), dtype=np.uint8) * 255
    for i, img in enumerate(arr):
        r, col = divmod(i, ncol)
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = img
    Image.fromarray(grid).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    cfg = load_default_config()
    out = OUTPUTS / "s1_baseline"
    out.mkdir(parents=True, exist_ok=True)

    device = get_device()
    from diffusers import DDPMPipeline, DDIMScheduler

    pipe = DDPMPipeline.from_pretrained(
        cfg["model_id"],
        use_safetensors=False,  # official checkpoint is pickle, not safetensors
    ).to(device)
    pipe.unet.eval()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Diffusers randn_tensor requires a CPU generator on MPS.
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    with torch.inference_mode():
        # DDPMPipeline does not take eta; DDIMScheduler.step defaults to 0.
        result = pipe(
            batch_size=args.n,
            num_inference_steps=args.steps,
            generator=generator,
            output_type="np",
        )
    images = result.images  # numpy float [0,1] or PIL depending on version
    if isinstance(images, list):
        arr = np.stack([np.asarray(im) for im in images], axis=0)
        if arr.dtype != np.uint8:
            arr = (arr * 255).round().clip(0, 255).astype(np.uint8)
    else:
        arr = (images * 255).round().clip(0, 255).astype(np.uint8)

    for i, img in enumerate(arr):
        Image.fromarray(img).save(out / f"sample_{i:02d}.png")
    save_grid(arr, out / "grid.png")

    # throughput: second pass after warmup
    generator_w = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    with torch.inference_mode():
        pipe(
            batch_size=min(args.batch_size, args.n),
            num_inference_steps=args.steps,
            generator=generator_w,
            output_type="np",
        )
        sync(device)
        t0 = time.perf_counter()
        generator_t = torch.Generator(device="cpu").manual_seed(args.seed + 2)
        pipe(
            batch_size=min(args.batch_size, args.n),
            num_inference_steps=args.steps,
            generator=generator_t,
            output_type="np",
        )
        sync(device)
        elapsed = time.perf_counter() - t0
    n_timed = min(args.batch_size, args.n)
    throughput = {
        "device": str(device),
        "batch_size": n_timed,
        "steps": args.steps,
        "seconds": elapsed,
        "images_per_sec": n_timed / elapsed,
        "nfe_per_image": args.steps,
        "image_nfe_per_sec": n_timed * args.steps / elapsed,
    }

    import diffusers
    import transformers

    metadata = {
        "model_id": cfg["model_id"],
        "device": str(device),
        "seed": args.seed,
        "n": args.n,
        "num_inference_steps": args.steps,
        "eta": cfg["ddim"]["eta"],
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "scheduler_class": pipe.scheduler.__class__.__name__,
        "scheduler_config": dict(pipe.scheduler.config),
        "unet_config": {
            "sample_size": pipe.unet.config.sample_size,
            "in_channels": pipe.unet.config.in_channels,
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (out / "throughput.json").write_text(json.dumps(throughput, indent=2))
    print(json.dumps({"out": str(out), **throughput}, indent=2))


if __name__ == "__main__":
    main()
