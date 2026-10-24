#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from gencv.denoiser import CountedUNet
from gencv.device import get_device
from gencv.generate import generate
from gencv.model import load_unet
from gencv.noise import VPSchedule
from gencv.paths import OUTPUTS
from gencv.schedules import expected_nfe, timesteps_for_nfe
from gencv.timing import write_json


def main() -> None:
    t_all = time.perf_counter()
    device = get_device()
    vp = VPSchedule.from_config()
    unet = CountedUNet(load_unet(device))
    z = torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(0)).to(device)
    rows = []
    ok = True
    for sampler in ("ddim", "dpm_pp_2m", "heun"):
        for kind in ("uniform", "karras"):
            for nfe in (9, 19, 39):
                ts = timesteps_for_nfe(nfe, sampler, kind, vp)
                k = len(ts)
                exp = expected_nfe(sampler, k)
                _ = generate(sampler, unet, z.clone(), ts, vp, clip_sample=False)
                got = unet.nfe
                row = {
                    "sampler": sampler,
                    "schedule": kind,
                    "nfe_budget": nfe,
                    "k": k,
                    "expected": exp,
                    "actual": got,
                    "ok": got == exp == nfe,
                }
                rows.append(row)
                ok = ok and row["ok"]
                print(row)
    payload = {"rows": rows, "ok": ok, "seconds": time.perf_counter() - t_all, "device": str(device)}
    write_json(OUTPUTS / "s6_nfe.json", payload)
    if not ok:
        raise SystemExit("NFE check failed")


if __name__ == "__main__":
    main()
