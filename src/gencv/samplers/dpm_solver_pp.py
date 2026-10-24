from __future__ import annotations

import torch

from gencv.denoiser import CountedUNet
from gencv.noise import edm_alpha_sigma


def _x0_from_eps(x: torch.Tensor, eps: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    alpha, sigma = edm_alpha_sigma(s)
    return (x - sigma * eps) / alpha


def _first_order(x: torch.Tensor, x0: torch.Tensor, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    if float(s_next) == 0.0:
        return x0
    alpha, sigma = edm_alpha_sigma(s)
    alpha_n, sigma_n = edm_alpha_sigma(s_next)
    lam = torch.log(alpha) - torch.log(sigma)
    lam_n = torch.log(alpha_n) - torch.log(sigma_n)
    h = lam_n - lam
    return (sigma_n / sigma) * x - alpha_n * torch.expm1(-h) * x0


def _second_order(
    x: torch.Tensor,
    x0: torch.Tensor,
    x0_prev: torch.Tensor,
    s: torch.Tensor,
    s_next: torch.Tensor,
    s_prev: torch.Tensor,
) -> torch.Tensor:
    if float(s_next) == 0.0:
        return x0
    alpha, sigma = edm_alpha_sigma(s)
    alpha_n, sigma_n = edm_alpha_sigma(s_next)
    alpha_p, sigma_p = edm_alpha_sigma(s_prev)
    lam = torch.log(alpha) - torch.log(sigma)
    lam_n = torch.log(alpha_n) - torch.log(sigma_n)
    lam_p = torch.log(alpha_p) - torch.log(sigma_p)
    h = lam_n - lam
    h0 = lam - lam_p
    r0 = h0 / h
    d1 = (x0 - x0_prev) / r0
    coeff = alpha_n * torch.expm1(-h)
    return (sigma_n / sigma) * x - coeff * x0 - 0.5 * coeff * d1


def dpm_pp_2m_sample(
    unet: CountedUNet,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    ts: list | torch.Tensor,
    clip_sample: bool = False,
    clip_range: float = 1.0,
) -> torch.Tensor:
    """Multistep DPM-Solver++ order-2 midpoint in x-space.

    `sigmas` decreasing, last value 0. Last step is first-order
    (`final_sigmas_type=zero`). History is local to this call.
    """
    sigmas = sigmas.to(device=x.device, dtype=x.dtype)
    ts = [t for t in ts]
    n_pos = sigmas.numel() - 1
    if len(ts) != n_pos:
        raise ValueError(f"expected {n_pos} timesteps, got {len(ts)}")

    x0_prev = None
    s_prev = None
    for i in range(n_pos):
        s = sigmas[i]
        s_next = sigmas[i + 1]
        eps = unet(x, ts[i])
        x0 = _x0_from_eps(x, eps, s)
        if clip_sample:
            x0 = x0.clamp(-clip_range, clip_range)
        is_last = i == n_pos - 1
        if x0_prev is None or is_last:
            x = _first_order(x, x0, s, s_next)
        else:
            x = _second_order(x, x0, x0_prev, s, s_next, s_prev)
        x0_prev = x0
        s_prev = s
    return x
