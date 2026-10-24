#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml

from gencv.denoiser import CountedUNet
from gencv.device import get_device, sync
from gencv.generate import generate_batched
from gencv.images import save_grid
from gencv.model import load_unet
from gencv.noise import VPSchedule
from gencv.noise_bank import load_bank
from gencv.paths import DATA, OUTPUTS, REFERENCES, load_default_config
from gencv.schedules import timesteps_for_nfe
from gencv.search import apply_rule_timesteps, mse_to_ref
from gencv.timing import write_json


def timed_gen(unet, sampler, z, ts, vp, batch, device):
    sync(device)
    t0 = time.perf_counter()
    out = generate_batched(sampler, unet, z, ts, vp, batch_size=batch, clip_sample=False)
    sync(device)
    dt = time.perf_counter() - t0
    nfe = unet.nfe
    return out, nfe, dt


def main() -> None:
    t_all = time.perf_counter()
    cfg = load_default_config()
    device = get_device()
    vp = VPSchedule.from_config()
    unet = CountedUNet(load_unet(device))
    batch = int(cfg["batch_size_local"])
    z = load_bank("mse_test").to(device)
    ref = torch.load(REFERENCES / "mse_test_ddim1000.pt", map_location="cpu", weights_only=True)
    vis = z[:8]
    out_dir = OUTPUTS / "s8"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sampler in ("ddim", "dpm_pp_2m", "heun"):
        for kind in ("uniform", "karras"):
            for nfe in (9, 19, 39):
                ts = timesteps_for_nfe(nfe, sampler, kind, vp)
                pred, nfe_act, dt = timed_gen(unet, sampler, z, ts, vp, batch, device)
                row = {
                    "group": "standard",
                    "sampler": sampler,
                    "schedule": kind,
                    "nfe_budget": nfe,
                    "nfe_actual": nfe_act,
                    "mse": mse_to_ref(pred, ref),
                    "seconds": dt,
                    "k": len(ts),
                }
                rows.append(row)
                print(row)
                if nfe == 19 and kind == "karras":
                    save_grid(pred[:8], out_dir / f"grid_{sampler}_{kind}_{nfe}.png")

    found = yaml.safe_load((DATA / "schedules" / "found.yaml").read_text())
    deltas = found["deltas"]
    for sampler, k in (("dpm_pp_2m", 19), ("ddim", 19), ("heun", 10)):
        ts = apply_rule_timesteps(deltas, k, vp)
        pred, nfe_act, dt = timed_gen(unet, sampler, z, ts, vp, batch, device)
        row = {
            "group": "found_transfer",
            "sampler": sampler,
            "schedule": "found_rule",
            "nfe_budget": 19,
            "nfe_actual": nfe_act,
            "mse": mse_to_ref(pred, ref),
            "seconds": dt,
            "k": len(ts),
        }
        rows.append(row)
        print(row)
        save_grid(pred[:8], out_dir / f"grid_{sampler}_found_19.png")

    for name in ("early", "mid", "late"):
        ctrl = yaml.safe_load((DATA / "schedules" / f"control_{name}.yaml").read_text())
        ts = ctrl["timesteps"]
        pred, nfe_act, dt = timed_gen(unet, "dpm_pp_2m", z, ts, vp, batch, device)
        row = {
            "group": "control",
            "sampler": "dpm_pp_2m",
            "schedule": f"anchor_{name}",
            "nfe_budget": 19,
            "nfe_actual": nfe_act,
            "mse": mse_to_ref(pred, ref),
            "seconds": dt,
            "k": len(ts),
        }
        rows.append(row)
        print(row)

    csv_path = out_dir / "mse_table.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    save_grid(vis.cpu(), out_dir / "grid_noise.png")
    save_grid(ref[:8], out_dir / "grid_ref_ddim1000.png")
    write_json(
        out_dir / "mse_table.json",
        {"rows": rows, "seconds": time.perf_counter() - t_all, "device": str(device)},
    )


if __name__ == "__main__":
    main()
