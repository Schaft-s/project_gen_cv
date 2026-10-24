#!/usr/bin/env python3
"""FID for 6 configs at NFE 19. Intended for Colab (CUDA Inception)."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml
from PIL import Image
from cleanfid import fid as clean_fid

from gencv.denoiser import CountedUNet
from gencv.device import get_device
from gencv.generate import generate_batched
from gencv.images import to_uint8
from gencv.model import load_unet
from gencv.noise import VPSchedule
from gencv.noise_bank import load_bank
from gencv.paths import DATA, OUTPUTS, load_default_config
from gencv.schedules import timesteps_for_nfe
from gencv.search import apply_rule_timesteps
from gencv.timing import write_json


def dump_pngs(images: torch.Tensor, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    arr = to_uint8(images)
    for i, img in enumerate(arr):
        Image.fromarray(img).save(folder / f"{i:05d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--nfe", type=int, default=19)
    args = parser.parse_args()

    t_all = time.perf_counter()
    cfg = load_default_config()
    device = get_device()
    if device.type != "cuda":
        print("WARNING: clean-fid Inception is slow off CUDA. Prefer Colab GPU.")
    vp = VPSchedule.from_config()
    unet = CountedUNet(load_unet(device))
    batch = int(cfg["batch_size_colab"] if device.type == "cuda" else cfg["batch_size_local"])
    z_all = load_bank("fid")[: args.n].to(device)
    found = yaml.safe_load((DATA / "schedules" / "found.yaml").read_text())
    deltas = found["deltas"]

    jobs = []
    for sampler in ("ddim", "dpm_pp_2m", "heun"):
        jobs.append((sampler, "karras", timesteps_for_nfe(args.nfe, sampler, "karras", vp)))
        k = 10 if sampler == "heun" else args.nfe
        jobs.append((sampler, "found_rule", apply_rule_timesteps(deltas, k, vp)))

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for sampler, kind, ts in jobs:
            pred = generate_batched(sampler, unet, z_all, ts, vp, batch_size=batch, clip_sample=False)
            folder = tmp / f"{sampler}_{kind}"
            dump_pngs(pred, folder)
            score = clean_fid.compute_fid(str(folder), dataset_name="cifar10", dataset_res=32, mode="clean", dataset_split="test")
            results.append(
                {
                    "sampler": sampler,
                    "schedule": kind,
                    "nfe": args.nfe,
                    "n": args.n,
                    "fid": float(score),
                }
            )
            print(results[-1])

    payload = {
        "protocol": "clean-fid cifar10 test 32",
        "n": args.n,
        "nfe": args.nfe,
        "device": str(device),
        "results": results,
        "seconds": time.perf_counter() - t_all,
    }
    out = OUTPUTS / "s8" / "fid.json"
    write_json(out, payload)
    print("wrote", out)


if __name__ == "__main__":
    main()
