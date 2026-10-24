#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml

from gencv.denoiser import CountedUNet
from gencv.device import get_device
from gencv.generate import generate_batched
from gencv.model import load_unet
from gencv.noise import VPSchedule
from gencv.noise_bank import load_bank
from gencv.paths import DATA, OUTPUTS, REFERENCES, load_default_config
from gencv.search import apply_rule_timesteps, coordinate_search, mse_to_ref, one_anchor_grids
from gencv.timing import write_json


def main() -> None:
    t_all = time.perf_counter()
    cfg = load_default_config()
    device = get_device()
    vp = VPSchedule.from_config()
    unet = CountedUNet(load_unet(device))
    batch = int(cfg["batch_size_local"])
    z_s = load_bank("search")
    ref_s = torch.load(REFERENCES / "search_ddim1000.pt", map_location="cpu", weights_only=True)
    found = coordinate_search(
        unet,
        vp,
        z_s,
        ref_s,
        device,
        batch_size=batch,
        log_path=OUTPUTS / "s7_search_log.csv",
    )

    z_v = load_bank("valid").to(device)
    ref_v = torch.load(REFERENCES / "valid_ddim1000.pt", map_location="cpu", weights_only=True)
    pred_v = generate_batched(
        "dpm_pp_2m", unet, z_v, found["timesteps"], vp, batch_size=batch, clip_sample=False
    )
    found["mse_valid"] = mse_to_ref(pred_v, ref_v)

    # control grids: pick the better of ± on search MSE for each single anchor
    controls = one_anchor_grids(vp)
    z_s_dev = z_s.to(device)
    chosen = {}
    for name, pack in controls.items():
        best = None
        for var in pack["variants"]:
            pred = generate_batched(
                "dpm_pp_2m",
                unet,
                z_s_dev,
                var["timesteps"],
                vp,
                batch_size=batch,
                clip_sample=False,
            )
            mse = mse_to_ref(pred, ref_s)
            if best is None or mse < best["mse_search"]:
                best = {**var, "mse_search": mse, "anchor": name}
        if best is None:
            best = {"timesteps": pack["base_timesteps"], "deltas": pack["best_d"], "mse_search": None, "anchor": name}
        chosen[name] = best
        path = DATA / "schedules" / f"control_{name}.yaml"
        path.write_text(yaml.safe_dump(best, sort_keys=False))

    (DATA / "schedules" / "found.yaml").write_text(yaml.safe_dump(found, sort_keys=False))
    found["seconds_total"] = time.perf_counter() - t_all
    found["controls"] = {k: {"mse_search": v.get("mse_search"), "deltas": v.get("deltas")} for k, v in chosen.items()}
    write_json(OUTPUTS / "s7_search.json", found)
    print(found)


if __name__ == "__main__":
    main()
