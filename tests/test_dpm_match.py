import torch
from diffusers import DPMSolverMultistepScheduler

from gencv.denoiser import CountedUNet
from gencv.diffusers_loops import diffusers_dpm_loop, ours_dpm_from_scheduler


def _scheduler(n: int = 20) -> DPMSolverMultistepScheduler:
    sch = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        algorithm_type="dpmsolver++",
        solver_order=2,
        solver_type="midpoint",
        prediction_type="epsilon",
        lower_order_final=True,
        final_sigmas_type="zero",
        thresholding=False,
        timestep_spacing="linspace",
    )
    sch.set_timesteps(n)
    return sch


def test_dpm_trajectory(unet, device):
    sch = _scheduler(20)
    z = torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(0)).to(device)
    counted = CountedUNet(unet)
    ours = ours_dpm_from_scheduler(counted, z.clone(), sch)
    nfe = counted.nfe
    counted.reset()
    sch = _scheduler(20)
    ref = diffusers_dpm_loop(counted, z.clone(), sch)
    err = (ours - ref).abs().max().item()
    assert nfe == 20
    assert err < 1e-4, err
