#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml

from gencv.denoiser import CountedUNet
from gencv.device import get_device
from gencv.generate import generate
from gencv.images import save_grid
from gencv.model import load_unet
from gencv.noise import VPSchedule
from gencv.noise_bank import load_bank
from gencv.paths import DATA, OUTPUTS
from gencv.schedules import timesteps_for_nfe
from gencv.search import apply_rule_timesteps


def main() -> None:
    device = get_device()
    vp = VPSchedule.from_config()
    unet = CountedUNet(load_unet(device))
    z = load_bank("mse_test")[:4].to(device)
    found = yaml.safe_load((DATA / "schedules" / "found.yaml").read_text())
    out = OUTPUTS / "s9_demo"
    out.mkdir(parents=True, exist_ok=True)
    grids = []
    for sampler in ("ddim", "dpm_pp_2m", "heun"):
        ts_k = timesteps_for_nfe(19, sampler, "karras", vp)
        k = 10 if sampler == "heun" else 19
        ts_f = apply_rule_timesteps(found["deltas"], k, vp)
        a = generate(sampler, unet, z, ts_k, vp, clip_sample=False)
        b = generate(sampler, unet, z, ts_f, vp, clip_sample=False)
        save_grid(a, out / f"{sampler}_karras.png", ncol=4)
        save_grid(b, out / f"{sampler}_found.png", ncol=4)
        grids.append(a)
        grids.append(b)
    save_grid(torch.cat(grids, dim=0), out / "all.png", ncol=4)
    print("wrote", out)


if __name__ == "__main__":
    main()
