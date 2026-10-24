from __future__ import annotations

import copy
import csv
import time
from pathlib import Path

import torch
import yaml

from gencv.denoiser import CountedUNet
from gencv.generate import generate_batched
from gencv.noise import VPSchedule
from gencv.paths import DATA, load_default_config
from gencv.schedules import karras_sigmas, snap_sigmas_to_timesteps


ANCHOR_POS = (0.25, 0.50, 0.75)


def shift_sigmas(base: torch.Tensor, deltas: list[float]) -> torch.Tensor:
    """Piecewise-linear log-s shifts; endpoints stay 0."""
    k = base.numel()
    u = torch.linspace(0.0, 1.0, k)
    nodes_u = torch.tensor([0.0, *ANCHOR_POS, 1.0])
    nodes_d = torch.tensor([0.0, *deltas, 0.0])
    delta_u = torch.zeros(k)
    for i in range(k):
        delta_u[i] = _interp(float(u[i]), nodes_u, nodes_d)
    return base * torch.exp(delta_u)


def _interp(x: float, xp: torch.Tensor, fp: torch.Tensor) -> float:
    for i in range(len(xp) - 1):
        if xp[i] <= x <= xp[i + 1]:
            w = 0.0 if xp[i + 1] == xp[i] else (x - float(xp[i])) / float(xp[i + 1] - xp[i])
            return float(fp[i] + w * (fp[i + 1] - fp[i]))
    return float(fp[-1])


def try_timesteps(sigmas: torch.Tensor, vp: VPSchedule, k: int) -> list[int] | None:
    try:
        ts = snap_sigmas_to_timesteps(sigmas, vp)
    except ValueError:
        return None
    if len(ts) != k:
        return None
    return ts


def mse_to_ref(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return float((pred.float() - ref.float()).pow(2).mean().item())


def coordinate_search(
    unet: CountedUNet,
    vp: VPSchedule,
    z_search: torch.Tensor,
    ref_search: torch.Tensor,
    device: torch.device,
    batch_size: int,
    log_path: Path,
) -> dict:
    cfg = load_default_config()
    k = int(cfg["search"]["nfe"])
    rho = float(cfg["karras_rho"])
    s_max = float(vp.sigmas[-1].cpu())
    s_min = float(vp.sigmas[0].cpu())
    base = karras_sigmas(k, s_min=s_min, s_max=s_max, rho=rho)
    shifts = [float(x) for x in cfg["search"]["log_sigma_shifts"]]
    deltas = [0.0, 0.0, 0.0]
    z = z_search.to(device)
    ref = ref_search

    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()

    def eval_deltas(d: list[float], tag: str) -> dict | None:
        sig = shift_sigmas(base, d)
        ts = try_timesteps(sig, vp, k)
        if ts is None:
            return None
        pred = generate_batched(
            "dpm_pp_2m",
            unet,
            z,
            ts,
            vp,
            batch_size=batch_size,
            clip_sample=False,
        )
        mse = mse_to_ref(pred, ref)
        row = {
            "tag": tag,
            "deltas": copy.deepcopy(d),
            "timesteps": ts,
            "mse": mse,
            "nfe": unet.nfe / z.shape[0],
        }
        rows.append(row)
        return row

    current = eval_deltas(deltas, "base")
    if current is None:
        raise RuntimeError("base Karras did not snap")
    best = current

    for pass_i, step in enumerate(shifts):
        for j in range(3):
            local_best = best
            local_d = deltas
            for sign, name in ((+1, "plus"), (-1, "minus")):
                cand = deltas.copy()
                cand[j] = cand[j] + sign * step
                row = eval_deltas(cand, f"pass{pass_i}_a{j}_{name}")
                if row is None:
                    continue
                if row["mse"] < local_best["mse"]:
                    local_best = row
                    local_d = cand
            best = local_best
            deltas = local_d

    elapsed = time.perf_counter() - t0
    with log_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "mse", "nfe", "deltas", "timesteps"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "tag": r["tag"],
                    "mse": f"{r['mse']:.8e}",
                    "nfe": r["nfe"],
                    "deltas": r["deltas"],
                    "timesteps": r["timesteps"],
                }
            )

    found = {
        "sampler": "dpm_pp_2m",
        "nfe": k,
        "base": "karras",
        "deltas": deltas,
        "timesteps": best["timesteps"],
        "mse_search": best["mse"],
        "n_candidates": len(rows),
        "seconds": elapsed,
    }
    out_yaml = DATA / "schedules" / "found.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(found, sort_keys=False))
    return found


def one_anchor_grids(vp: VPSchedule) -> dict[str, dict]:
    """Control grids: move only early / mid / late anchor with the same budget."""
    cfg = load_default_config()
    k = int(cfg["search"]["nfe"])
    rho = float(cfg["karras_rho"])
    s_max = float(vp.sigmas[-1].cpu())
    s_min = float(vp.sigmas[0].cpu())
    base = karras_sigmas(k, s_min=s_min, s_max=s_max, rho=rho)
    step = float(cfg["search"]["log_sigma_shifts"][0])
    out = {}
    names = ("early", "mid", "late")
    for j, name in enumerate(names):
        best_ts = snap_sigmas_to_timesteps(base, vp)
        best_d = [0.0, 0.0, 0.0]
        # store both directions; caller evaluates MSE
        variants = []
        for sign in (+1, -1):
            d = [0.0, 0.0, 0.0]
            d[j] = sign * step
            sig = shift_sigmas(base, d)
            ts = try_timesteps(sig, vp, k)
            if ts is not None:
                variants.append({"deltas": d, "timesteps": ts})
        out[name] = {"variants": variants, "base_timesteps": best_ts, "best_d": best_d}
    return out


def apply_rule_timesteps(deltas: list[float], k: int, vp: VPSchedule, rho: float = 7.0) -> list[int]:
    s_max = float(vp.sigmas[-1].cpu())
    s_min = float(vp.sigmas[0].cpu())
    base = karras_sigmas(k, s_min=s_min, s_max=s_max, rho=rho)
    sig = shift_sigmas(base, deltas)
    ts = try_timesteps(sig, vp, k)
    if ts is None:
        raise RuntimeError("rule did not snap")
    return ts
