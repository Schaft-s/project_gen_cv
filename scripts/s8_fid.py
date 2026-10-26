#!/usr/bin/env python3
"""FID for 6 configs at NFE 19. MPS generation + CPU Inception is fine."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml
from PIL import Image
from cleanfid import fid as clean_fid
from scipy import linalg


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """clean-fid's Frechet Distance without scipy's removed sqrtm(disp=...)."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(np.asarray(covmean)).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


clean_fid.frechet_distance = _frechet_distance
import cleanfid.fid as _cleanfid_fid_mod

_cleanfid_fid_mod.frechet_distance = _frechet_distance

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
    fid_device = torch.device("cuda" if device.type == "cuda" else "cpu")
    print(f"sample device={device}  inception device={fid_device}", flush=True)
    vp = VPSchedule.from_config()
    batch = int(cfg["batch_size_colab"] if device.type == "cuda" else cfg["batch_size_local"])
    found = yaml.safe_load((DATA / "schedules" / "found.yaml").read_text())
    deltas = found["deltas"]
    unet = None
    z_all = None

    jobs = []
    for sampler in ("ddim", "dpm_pp_2m", "heun"):
        jobs.append((sampler, "karras", timesteps_for_nfe(args.nfe, sampler, "karras", vp)))
        k = 10 if sampler == "heun" else args.nfe
        jobs.append((sampler, "found_rule", apply_rule_timesteps(deltas, k, vp)))

    img_root = OUTPUTS / "s8" / "fid_images"
    img_root.mkdir(parents=True, exist_ok=True)
    results = []
    for sampler, kind, ts in jobs:
        folder = img_root / f"{sampler}_{kind}"
        t0 = time.perf_counter()
        n_existing = len(list(folder.glob("*.png"))) if folder.exists() else 0
        if n_existing >= args.n:
            print(f"skip gen {sampler}/{kind}: {n_existing} pngs", flush=True)
            dt_gen = 0.0
        else:
            if unet is None:
                unet = CountedUNet(load_unet(device))
                z_all = load_bank("fid")[: args.n].to(device)
            pred = generate_batched(sampler, unet, z_all, ts, vp, batch_size=batch, clip_sample=False)
            dump_pngs(pred, folder)
            dt_gen = time.perf_counter() - t0
        t1 = time.perf_counter()
        score = clean_fid.compute_fid(
            str(folder),
            dataset_name="cifar10",
            dataset_res=32,
            mode="clean",
            dataset_split="test",
            device=fid_device,
            num_workers=0,
        )
        row = {
            "sampler": sampler,
            "schedule": kind,
            "nfe": args.nfe,
            "n": args.n,
            "fid": float(score),
            "seconds_gen": dt_gen,
            "seconds_fid": time.perf_counter() - t1,
        }
        results.append(row)
        print(row, flush=True)
        write_json(OUTPUTS / "s8" / "fid_partial.json", {"results": results, "elapsed": time.perf_counter() - t_all})

    payload = {
        "protocol": "clean-fid cifar10 test 32",
        "n": args.n,
        "nfe": args.nfe,
        "device": str(device),
        "inception_device": str(fid_device),
        "results": results,
        "seconds": time.perf_counter() - t_all,
    }
    out = OUTPUTS / "s8" / "fid.json"
    write_json(out, payload)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
