import torch
from diffusers import DDIMScheduler

from gencv.denoiser import CountedUNet
from gencv.diffusers_loops import diffusers_ddim_loop, ours_ddim_from_scheduler
from gencv.samplers.ddim import ddim_step


def _scheduler(n: int = 20) -> DDIMScheduler:
    sch = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=True,
        set_alpha_to_one=True,
        prediction_type="epsilon",
        timestep_spacing="leading",
    )
    sch.set_timesteps(n)
    return sch


def test_ddim_frozen_step(vp):
    sch = _scheduler(20)
    t = int(sch.timesteps[0].item())
    t_next = int(sch.timesteps[1].item())
    x = torch.randn(2, 3, 32, 32)
    eps = torch.randn_like(x)
    ours = ddim_step(x, eps, t, t_next, vp, clip_sample=True)
    ref = sch.step(eps, t, x, eta=0.0).prev_sample
    err = (ours - ref).abs().max().item()
    assert err < 1e-5, err


def test_ddim_trajectory(unet, vp, device):
    sch = _scheduler(20)
    z = torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    z = z.to(device)
    vp_d = vp.to(device)
    counted = CountedUNet(unet)
    ours = ours_ddim_from_scheduler(counted, z.clone(), sch, vp_d)
    counted.reset()
    ref = diffusers_ddim_loop(counted, z.clone(), sch)
    err = (ours - ref).abs().max().item()
    assert err < 1e-4, err
