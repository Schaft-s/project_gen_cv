import torch
from diffusers import HeunDiscreteScheduler

from gencv.denoiser import CountedUNet
from gencv.diffusers_loops import diffusers_heun_loop, ours_heun_from_scheduler
from gencv.schedules import expected_nfe


def _scheduler(n: int = 10) -> HeunDiscreteScheduler:
    sch = HeunDiscreteScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=True,
        prediction_type="epsilon",
        timestep_spacing="linspace",
    )
    sch.set_timesteps(n)
    return sch


def test_heun_nfe_formula():
    assert expected_nfe("heun", 10) == 19
    assert expected_nfe("heun", 5) == 9


def test_heun_trajectory(unet, vp, device):
    sch = _scheduler(10)
    g = torch.Generator().manual_seed(0)
    z = torch.randn(4, 3, 32, 32, generator=g)
    y = z * float(sch.init_noise_sigma)
    y = y.to(device)
    vp_d = vp.to(device)
    counted = CountedUNet(unet)
    ours = ours_heun_from_scheduler(counted, y.clone(), sch, vp_d)
    nfe_ours = counted.nfe
    counted.reset()
    # Diffusers scheduler keeps state; rebuild
    sch = _scheduler(10)
    ref = diffusers_heun_loop(counted, y.clone(), sch)
    err = (ours - ref).abs().max().item()
    assert nfe_ours == 2 * 10 - 1
    assert err < 1e-4, err
