from __future__ import annotations

import torch

from gencv.denoiser import CountedUNet
from gencv.noise import VPSchedule


def ddim_step(
    x: torch.Tensor,
    eps: torch.Tensor,
    t: int,
    t_next: int,
    vp: VPSchedule,
    clip_sample: bool = False,
    clip_range: float = 1.0,
) -> torch.Tensor:
    """One DDIM η=0 step with an explicit (t, t_next) pair. t_next < 0 → ᾱ=1."""
    abar_t = vp.abar(t).to(device=x.device, dtype=x.dtype)
    abar_next = vp.abar(t_next).to(device=x.device, dtype=x.dtype)
    x0 = (x - (1.0 - abar_t).sqrt() * eps) / abar_t.sqrt()
    if clip_sample:
        x0 = x0.clamp(-clip_range, clip_range)
    return abar_next.sqrt() * x0 + (1.0 - abar_next).sqrt() * eps


def ddim_sample(
    unet: CountedUNet,
    x: torch.Tensor,
    timesteps: list[int] | torch.Tensor,
    vp: VPSchedule,
    clip_sample: bool = False,
    clip_range: float = 1.0,
) -> torch.Tensor:
    ts = [int(t) for t in timesteps]
    for i, t in enumerate(ts):
        t_next = ts[i + 1] if i + 1 < len(ts) else -1
        eps = unet(x, t)
        x = ddim_step(x, eps, t, t_next, vp, clip_sample=clip_sample, clip_range=clip_range)
    return x
