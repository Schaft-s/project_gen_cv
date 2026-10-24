import torch

from gencv.noise import VPSchedule


def test_beta_bounds():
    vp = VPSchedule.from_config()
    assert vp.T == 1000
    assert abs(float(vp.betas[0]) - 1e-4) < 1e-10
    assert abs(float(vp.betas[-1]) - 0.02) < 1e-6


def test_xy_roundtrip():
    vp = VPSchedule.from_config()
    x = torch.randn(2, 3, 32, 32)
    for t in (0, 10, 500, 999):
        y = vp.x_to_y(x, t)
        x2 = vp.y_to_x(y, t)
        assert torch.allclose(x, x2, atol=1e-6)


def test_x0_formula_consistency():
    vp = VPSchedule.from_config()
    x0 = torch.randn(1, 3, 32, 32)
    eps = torch.randn_like(x0)
    t = 400
    abar = vp.abar(t)
    x = abar.sqrt() * x0 + (1 - abar).sqrt() * eps
    x0_hat = vp.x0_from_eps(x, eps, t)
    assert torch.allclose(x0, x0_hat, atol=1e-5)


def test_nearest_t_hits_table():
    vp = VPSchedule.from_config()
    for t in (0, 17, 999):
        s = float(vp.sigma_from_t(t))
        assert vp.nearest_t(s) == t
