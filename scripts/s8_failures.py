#!/usr/bin/env python3
"""Save the 4 mse_test images with the largest DPM-found vs DDIM-1000 error."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml

from gencv.denoiser import CountedUNet
from gencv.device import get_device
from gencv.generate import generate_batched
from gencv.images import save_grid
from gencv.model import load_unet
from gencv.noise import VPSchedule
from gencv.noise_bank import load_bank
from gencv.paths import DATA, OUTPUTS, REFERENCES, load_default_config
from gencv.schedules import timesteps_for_nfe
from gencv.search import apply_rule_timesteps


def per_image_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).pow(2).mean(dim=(1, 2, 3))


def main() -> None:
    cfg = load_default_config()
    device = get_device()
    vp = VPSchedule.from_config()
    unet = CountedUNet(load_unet(device))
    batch = int(cfg["batch_size_local"])
    z = load_bank("mse_test")
    ref = torch.load(REFERENCES / "mse_test_ddim1000.pt", map_location="cpu", weights_only=True)
    found = yaml.safe_load((DATA / "schedules" / "found.yaml").read_text())
    ts_k = timesteps_for_nfe(19, "dpm_pp_2m", "karras", vp)
    ts_f = apply_rule_timesteps(found["deltas"], 19, vp)
    pred_k = generate_batched("dpm_pp_2m", unet, z.to(device), ts_k, vp, batch, clip_sample=False)
    pred_f = generate_batched("dpm_pp_2m", unet, z.to(device), ts_f, vp, batch, clip_sample=False)
    err = per_image_mse(pred_f, ref)
    idx = torch.topk(err, 4).indices.tolist()
    out = OUTPUTS / "s8"
    save_grid(ref[idx], out / "fail_ref.png", ncol=4)
    save_grid(pred_k[idx], out / "fail_dpm_karras.png", ncol=4)
    save_grid(pred_f[idx], out / "fail_dpm_found.png", ncol=4)
    print("worst indices", idx, "mse", [float(err[i]) for i in idx])


if __name__ == "__main__":
    main()
