from __future__ import annotations

import torch

from gencv.denoiser import Denoiser


def heun_sample(
    denoiser: Denoiser,
    y: torch.Tensor,
    sigmas: torch.Tensor,
    ts: list | torch.Tensor,
    clip_sample: bool = False,
    clip_range: float = 1.0,
) -> torch.Tensor:
    """Deterministic Heun (S_churn=0) in y-space.

    `sigmas` is decreasing and must end with 0. `ts` has one time label per
    positive sigma (same length as sigmas minus one).
    """
    sigmas = sigmas.to(device=y.device, dtype=y.dtype)
    ts = [t for t in ts]
    assert sigmas[-1] == 0 or float(sigmas[-1]) == 0.0
    n_pos = sigmas.numel() - 1
    if len(ts) != n_pos:
        raise ValueError(f"expected {n_pos} timesteps, got {len(ts)}")

    def predict_x0(y_cur: torch.Tensor, s: torch.Tensor, t) -> torch.Tensor:
        x0 = denoiser.D(y_cur, s, t)
        if clip_sample:
            x0 = x0.clamp(-clip_range, clip_range)
        return x0

    for i in range(n_pos):
        s = sigmas[i]
        s_next = sigmas[i + 1]
        t = ts[i]
        x0 = predict_x0(y, s, t)
        d = (y - x0) / s
        y_e = y + (s_next - s) * d
        if float(s_next) == 0.0:
            y = y_e
            continue
        t_next = ts[i + 1]
        x0_e = predict_x0(y_e, s_next, t_next)
        d_prime = (y_e - x0_e) / s_next
        y = y + (s_next - s) * 0.5 * (d + d_prime)
    return y
