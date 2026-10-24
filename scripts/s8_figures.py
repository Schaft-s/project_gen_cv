#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import yaml

from gencv.noise import VPSchedule
from gencv.paths import DATA, OUTPUTS
from gencv.schedules import karras_timesteps
from gencv.search import apply_rule_timesteps


def main() -> None:
    out = OUTPUTS / "s8"
    rows = list(csv.DictReader((out / "mse_table.csv").open()))
    for r in rows:
        r["nfe_budget"] = int(float(r["nfe_budget"]))
        r["mse"] = float(r["mse"])
        r["seconds"] = float(r["seconds"])

    fig, ax = plt.subplots(figsize=(6, 4))
    for sampler in ("ddim", "dpm_pp_2m", "heun"):
        for kind, ls in (("uniform", "--"), ("karras", "-")):
            pts = [r for r in rows if r["group"] == "standard" and r["sampler"] == sampler and r["schedule"] == kind]
            pts = sorted(pts, key=lambda r: r["nfe_budget"])
            ax.plot(
                [r["nfe_budget"] for r in pts],
                [r["mse"] for r in pts],
                ls,
                marker="o",
                label=f"{sampler} {kind}",
            )
    ax.set_xlabel("NFE")
    ax.set_ylabel("MSE to DDIM-1000")
    ax.legend(fontsize=7)
    ax.set_title("Short-run MSE vs budget")
    fig.tight_layout()
    fig.savefig(out / "mse_vs_nfe.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for sampler in ("ddim", "dpm_pp_2m", "heun"):
        pts = [
            r
            for r in rows
            if r["group"] == "standard" and r["sampler"] == sampler and r["schedule"] == "karras"
        ]
        pts = sorted(pts, key=lambda r: r["nfe_budget"])
        ax.plot([r["nfe_budget"] for r in pts], [r["seconds"] for r in pts], marker="o", label=sampler)
    ax.set_xlabel("NFE")
    ax.set_ylabel("seconds (256 images)")
    ax.legend()
    ax.set_title("Wall time vs budget")
    fig.tight_layout()
    fig.savefig(out / "time_vs_nfe.png", dpi=140)
    plt.close(fig)

    vp = VPSchedule.from_config()
    found = yaml.safe_load((DATA / "schedules" / "found.yaml").read_text())
    base_ts = karras_timesteps(19, vp)
    found_ts = apply_rule_timesteps(found["deltas"], 19, vp)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(base_ts)), [float(vp.sigma_from_t(t)) for t in base_ts], label="Karras")
    ax.plot(range(len(found_ts)), [float(vp.sigma_from_t(t)) for t in found_ts], label="found")
    ax.set_yscale("log")
    ax.set_xlabel("step index")
    ax.set_ylabel("sigma")
    ax.legend()
    ax.set_title("Noise schedule (DPM NFE 19)")
    fig.tight_layout()
    fig.savefig(out / "sigma_karras_vs_found.png", dpi=140)
    plt.close(fig)
    print("wrote figures in", out)


if __name__ == "__main__":
    main()
