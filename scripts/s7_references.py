#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from gencv.denoiser import CountedUNet
from gencv.device import get_device, sync
from gencv.generate import generate_batched
from gencv.model import load_unet
from gencv.noise import VPSchedule
from gencv.noise_bank import load_bank
from gencv.paths import OUTPUTS, REFERENCES, load_default_config
from gencv.schedules import uniform_timesteps
from gencv.search import mse_to_ref
from gencv.timing import write_json

MAX_HOURS = 1.0


def run_ddim(unet, vp, z, nfe, batch, device):
    ts = uniform_timesteps(nfe, t_max=vp.T)
    z = z.to(device)
    return generate_batched("ddim", unet, z, ts, vp, batch_size=batch, clip_sample=False)


def main() -> None:
    t_all = time.perf_counter()
    cfg = load_default_config()
    device = get_device()
    vp = VPSchedule.from_config()
    unet = CountedUNet(load_unet(device))
    batch = int(cfg["batch_size_local"])
    REFERENCES.mkdir(parents=True, exist_ok=True)

    # probe
    z_probe = load_bank("stability")[:2]
    sync(device)
    t0 = time.perf_counter()
    _ = run_ddim(unet, vp, z_probe, 50, batch=2, device=device)
    sync(device)
    dt = time.perf_counter() - t0
    nfe_probe = 2 * 50
    rate = nfe_probe / dt
    n_ref = 32 + 64 + 256
    est = n_ref * 1000 / rate
    print(f"probe rate {rate:.1f} image-NFE/s, est 1000-NFE refs {est/60:.1f} min")
    if est > MAX_HOURS * 3600:
        write_json(
            OUTPUTS / "s7_stop_colab.json",
            {"reason": "references too slow", "est_seconds": est, "rate": rate},
        )
        raise SystemExit(f"STOP for Colab: estimated {est/3600:.2f} h")

    stab = load_bank("stability")
    stability_mse = {}
    imgs = {}
    for nfe in cfg["reference"]["stability_nfes"]:
        imgs[nfe] = run_ddim(unet, vp, stab, int(nfe), batch=batch, device=device)
        torch.save(imgs[nfe], REFERENCES / f"stability_{nfe}.pt")
    stability_mse["250_vs_500"] = mse_to_ref(imgs[250], imgs[500])
    stability_mse["500_vs_1000"] = mse_to_ref(imgs[500], imgs[1000])
    stability_mse["250_vs_1000"] = mse_to_ref(imgs[250], imgs[1000])
    print("stability mse", stability_mse)

    for name in ("search", "valid", "mse_test"):
        path = REFERENCES / f"{name}_ddim1000.pt"
        if path.exists():
            print("skip existing", path)
            continue
        z = load_bank(name)
        t1 = time.perf_counter()
        ref = run_ddim(unet, vp, z, 1000, batch=batch, device=device)
        torch.save(ref, path)
        print(name, "done", time.perf_counter() - t1)

    payload = {
        "device": str(device),
        "probe_image_nfe_per_sec": rate,
        "est_ref_seconds": est,
        "stability_mse": stability_mse,
        "seconds": time.perf_counter() - t_all,
    }
    write_json(OUTPUTS / "s7_references.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
