#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from diffusers import DPMSolverMultistepScheduler

from gencv.denoiser import CountedUNet
from gencv.device import get_device
from gencv.diffusers_loops import diffusers_dpm_loop, ours_dpm_from_scheduler
from gencv.model import load_unet
from gencv.paths import OUTPUTS
from gencv.timing import write_json


def make_sch(n: int = 20) -> DPMSolverMultistepScheduler:
    sch = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        algorithm_type="dpmsolver++",
        solver_order=2,
        solver_type="midpoint",
        prediction_type="epsilon",
        lower_order_final=True,
        final_sigmas_type="zero",
        thresholding=False,
        timestep_spacing="linspace",
    )
    sch.set_timesteps(n)
    return sch


def main() -> None:
    t_all = time.perf_counter()
    device = get_device()
    unet = load_unet(device)
    counted = CountedUNet(unet)
    sch = make_sch(20)
    z = torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(0)).to(device)
    ours = ours_dpm_from_scheduler(counted, z.clone(), sch)
    nfe = counted.nfe
    counted.reset()
    sch = make_sch(20)
    ref = diffusers_dpm_loop(counted, z.clone(), sch)
    max_abs_traj = float((ours - ref).abs().max())
    payload = {
        "device": str(device),
        "max_abs_traj": max_abs_traj,
        "nfe": nfe,
        "expected_nfe": 20,
        "ok_traj": max_abs_traj < 1e-4,
        "ok_nfe": nfe == 20,
        "seconds": time.perf_counter() - t_all,
    }
    write_json(OUTPUTS / "s5_match_dpm.json", payload)
    print(payload)
    if not payload["ok_traj"] or not payload["ok_nfe"]:
        raise SystemExit("DPM match failed")


if __name__ == "__main__":
    main()
