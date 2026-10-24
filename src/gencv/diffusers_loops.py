from __future__ import annotations

import torch

from gencv.denoiser import CountedUNet, Denoiser
from gencv.noise import VPSchedule
from gencv.samplers.ddim import ddim_sample
from gencv.samplers.dpm_solver_pp import dpm_pp_2m_sample
from gencv.samplers.heun import heun_sample


@torch.no_grad()
def diffusers_ddim_loop(unet: CountedUNet, x: torch.Tensor, scheduler) -> torch.Tensor:
    for t in scheduler.timesteps:
        eps = unet(x, t)
        x = scheduler.step(eps, t, x, eta=0.0, return_dict=True).prev_sample
    return x


@torch.no_grad()
def ours_ddim_from_scheduler(unet: CountedUNet, x: torch.Tensor, scheduler, vp: VPSchedule) -> torch.Tensor:
    ts = [int(t) for t in scheduler.timesteps.tolist()]
    clip = bool(scheduler.config.clip_sample)
    rng = float(getattr(scheduler.config, "clip_sample_range", 1.0))
    return ddim_sample(unet, x, ts, vp, clip_sample=clip, clip_range=rng)


@torch.no_grad()
def diffusers_heun_loop(unet: CountedUNet, y: torch.Tensor, scheduler) -> torch.Tensor:
    for t in scheduler.timesteps:
        y_in = scheduler.scale_model_input(y, t)
        eps = unet(y_in, t)
        y = scheduler.step(eps, t, y, return_dict=True).prev_sample
    return y


def heun_unique_sigmas_ts(scheduler) -> tuple[torch.Tensor, list]:
    sigmas = torch.unique_consecutive(scheduler.sigmas.cpu().float())
    ts = torch.unique_consecutive(scheduler.timesteps.cpu().float()).tolist()
    if float(sigmas[-1]) != 0.0:
        sigmas = torch.cat([sigmas, torch.zeros(1)])
    return sigmas, ts


@torch.no_grad()
def ours_heun_from_scheduler(unet: CountedUNet, y: torch.Tensor, scheduler, vp: VPSchedule) -> torch.Tensor:
    sigmas, ts = heun_unique_sigmas_ts(scheduler)
    clip = bool(scheduler.config.clip_sample)
    rng = float(getattr(scheduler.config, "clip_sample_range", 1.0))
    denoiser = Denoiser(unet, vp)
    return heun_sample(
        denoiser,
        y,
        sigmas.to(y.device),
        ts,
        clip_sample=clip,
        clip_range=rng,
    )


@torch.no_grad()
def diffusers_dpm_loop(unet: CountedUNet, x: torch.Tensor, scheduler) -> torch.Tensor:
    for t in scheduler.timesteps:
        eps = unet(x, t)
        x = scheduler.step(eps, t, x, return_dict=True).prev_sample
    return x


@torch.no_grad()
def ours_dpm_from_scheduler(unet: CountedUNet, x: torch.Tensor, scheduler) -> torch.Tensor:
    sigmas = scheduler.sigmas.cpu().float()
    ts = scheduler.timesteps.tolist()
    clip = bool(getattr(scheduler.config, "thresholding", False))
    return dpm_pp_2m_sample(unet, x, sigmas.to(x.device), ts, clip_sample=clip)
