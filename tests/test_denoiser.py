import torch

from gencv.denoiser import CountedUNet, Denoiser


def test_nfe_increments(unet, device, vp):
    x = torch.randn(2, 3, 32, 32, device=device)
    counted = CountedUNet(unet)
    counted.reset()
    counted(x, 50)
    counted(x, 10)
    assert counted.nfe == 2


def test_D_matches_x0(unet, device, vp):
    vp_d = vp.to(device)
    counted = CountedUNet(unet)
    den = Denoiser(counted, vp_d)
    t = 300
    x = torch.randn(1, 3, 32, 32, device=device)
    y = vp_d.x_to_y(x, t)
    s = vp_d.sigma_from_t(t).to(device)
    x0_d = den.D(y, s, t)
    eps = counted.unet(x, torch.tensor([t], device=device)).sample
    x0 = vp_d.x0_from_eps(x, eps, t)
    assert torch.allclose(x0_d, x0, atol=1e-5)


def test_no_call_at_zero(unet, device, vp):
    counted = CountedUNet(unet)
    den = Denoiser(counted, vp.to(device))
    y = torch.randn(1, 3, 32, 32, device=device)
    counted.reset()
    out = den.D(y, torch.tensor(0.0, device=device), 0)
    assert counted.nfe == 0
    assert torch.equal(out, y)
