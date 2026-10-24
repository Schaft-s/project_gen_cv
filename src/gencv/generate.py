from __future__ import annotations

import torch

from gencv.denoiser import CountedUNet, Denoiser
from gencv.noise import VPSchedule
from gencv.samplers.ddim import ddim_sample
from gencv.samplers.dpm_solver_pp import dpm_pp_2m_sample
from gencv.samplers.heun import heun_sample
from gencv.schedules import sigmas_from_timesteps


def generate(
    sampler: str,
    unet: CountedUNet,
    z: torch.Tensor,
    timesteps: list[int],
    vp: VPSchedule,
    clip_sample: bool = False,
    clip_range: float = 1.0,
) -> torch.Tensor:
    """z is DDPM x-space N(0,I). Returns x in [-1,1] float (unclamped unless clip inside)."""
    vp = vp.to(z.device)
    unet.reset()
    if sampler == "ddim":
        return ddim_sample(unet, z, timesteps, vp, clip_sample=clip_sample, clip_range=clip_range)
    if sampler == "dpm_pp_2m":
        sigmas = sigmas_from_timesteps(timesteps, vp, append_zero=True).to(z.device)
        return dpm_pp_2m_sample(unet, z, sigmas, timesteps, clip_sample=clip_sample, clip_range=clip_range)
    if sampler == "heun":
        t0 = timesteps[0]
        y = vp.x_to_y(z, t0)
        sigmas = sigmas_from_timesteps(timesteps, vp, append_zero=True).to(z.device)
        denoiser = Denoiser(unet, vp)
        y = heun_sample(denoiser, y, sigmas, timesteps, clip_sample=clip_sample, clip_range=clip_range)
        return y  # at s=0, y = x0
    raise ValueError(sampler)


def generate_batched(
    sampler: str,
    unet: CountedUNet,
    z: torch.Tensor,
    timesteps: list[int],
    vp: VPSchedule,
    batch_size: int,
    clip_sample: bool = False,
    clip_range: float = 1.0,
) -> torch.Tensor:
    outs = []
    nfe_per_image = 0
    for i in range(0, z.shape[0], batch_size):
        chunk = z[i : i + batch_size]
        out = generate(
            sampler,
            unet,
            chunk,
            timesteps,
            vp,
            clip_sample=clip_sample,
            clip_range=clip_range,
        )
        # One forward processes the whole batch, so NFE/image = number of forwards.
        nfe_per_image = unet.nfe
        outs.append(out.detach().cpu())
    unet.nfe = nfe_per_image
    return torch.cat(outs, dim=0)
