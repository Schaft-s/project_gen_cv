#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from diffusers import HeunDiscreteScheduler

from gencv.denoiser import CountedUNet
from gencv.device import get_device
from gencv.diffusers_loops import diffusers_heun_loop, ours_heun_from_scheduler
from gencv.model import load_unet
from gencv.noise import VPSchedule
from gencv.paths import OUTPUTS
from gencv.timing import write_json


def make_sch(n: int = 10) -> HeunDiscreteScheduler:
    sch = HeunDiscreteScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=True,
        prediction_type="epsilon",
        timestep_spacing="linspace",
    )
    sch.set_timesteps(n)
    return sch


def main() -> None:
    t_all = time.perf_counter()
    device = get_device()
    vp = VPSchedule.from_config().to(device)
    unet = load_unet(device)
    counted = CountedUNet(unet)
    sch = make_sch(10)
    z = torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    y = (z * float(sch.init_noise_sigma)).to(device)
    ours = ours_heun_from_scheduler(counted, y.clone(), sch, vp)
    nfe = counted.nfe
    counted.reset()
    sch = make_sch(10)
    ref = diffusers_heun_loop(counted, y.clone(), sch)
    max_abs_traj = float((ours - ref).abs().max())
    payload = {
        "device": str(device),
        "max_abs_traj": max_abs_traj,
        "nfe": nfe,
        "expected_nfe": 19,
        "ok_traj": max_abs_traj < 1e-4,
        "ok_nfe": nfe == 19,
        "seconds": time.perf_counter() - t_all,
    }
    write_json(OUTPUTS / "s4_match_heun.json", payload)
    print(payload)
    if not payload["ok_traj"] or not payload["ok_nfe"]:
        raise SystemExit("Heun match failed")


if __name__ == "__main__":
    main()
