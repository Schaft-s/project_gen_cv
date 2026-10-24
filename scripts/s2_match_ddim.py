#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from diffusers import DDIMScheduler

from gencv.denoiser import CountedUNet
from gencv.device import get_device
from gencv.diffusers_loops import diffusers_ddim_loop, ours_ddim_from_scheduler
from gencv.model import load_unet
from gencv.noise import VPSchedule
from gencv.paths import OUTPUTS
from gencv.samplers.ddim import ddim_step
from gencv.timing import write_json


def main() -> None:
    t_all = time.perf_counter()
    device = get_device()
    vp = VPSchedule.from_config()
    sch = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=True,
        set_alpha_to_one=True,
        prediction_type="epsilon",
        timestep_spacing="leading",
    )
    sch.set_timesteps(20)
    t = int(sch.timesteps[0].item())
    t_next = int(sch.timesteps[1].item())
    x = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(1))
    eps = torch.randn_like(x)
    ours_step = ddim_step(x, eps, t, t_next, vp, clip_sample=True)
    ref_step = sch.step(eps, t, x, eta=0.0).prev_sample
    max_abs_step = float((ours_step - ref_step).abs().max())

    unet = load_unet(device)
    counted = CountedUNet(unet)
    z = torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(0)).to(device)
    vp_d = vp.to(device)
    ours = ours_ddim_from_scheduler(counted, z.clone(), sch, vp_d)
    counted.reset()
    ref = diffusers_ddim_loop(counted, z.clone(), sch)
    max_abs_traj = float((ours - ref).abs().max())

    payload = {
        "device": str(device),
        "max_abs_step": max_abs_step,
        "max_abs_traj": max_abs_traj,
        "atol_step": 1e-5,
        "atol_traj": 1e-4,
        "ok_step": max_abs_step < 1e-5,
        "ok_traj": max_abs_traj < 1e-4,
        "seconds": time.perf_counter() - t_all,
    }
    write_json(OUTPUTS / "s2_match_ddim.json", payload)
    print(payload)
    if not payload["ok_step"] or not payload["ok_traj"]:
        raise SystemExit("DDIM match failed")


if __name__ == "__main__":
    main()
