from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VPSchedule:
    """Discrete VP noise table of google/ddpm-cifar10-32."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sigmas: torch.Tensor  # EDM s_t = sqrt((1-ᾱ)/ᾱ)

    @classmethod
    def from_config(
        cls,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> "VPSchedule":
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod).sqrt()
        return cls(betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod, sigmas=sigmas)

    def to(self, device: torch.device | str) -> "VPSchedule":
        return VPSchedule(
            betas=self.betas.to(device),
            alphas=self.alphas.to(device),
            alphas_cumprod=self.alphas_cumprod.to(device),
            sigmas=self.sigmas.to(device),
        )

    @property
    def T(self) -> int:
        return int(self.betas.numel())

    def abar(self, t: int) -> torch.Tensor:
        if t < 0:
            return self.alphas_cumprod.new_tensor(1.0)
        return self.alphas_cumprod[t]

    def sigma_from_t(self, t: int) -> torch.Tensor:
        if t < 0:
            return self.sigmas.new_tensor(0.0)
        return self.sigmas[t]

    def nearest_t(self, sigma: float) -> int:
        s = self.sigmas.detach().float().cpu()
        return int(torch.argmin((s - float(sigma)).abs()).item())

    def x_to_y(self, x: torch.Tensor, t: int) -> torch.Tensor:
        s = self.sigma_from_t(t).to(device=x.device, dtype=x.dtype)
        return x * torch.sqrt(1.0 + s * s)

    def y_to_x(self, y: torch.Tensor, t: int) -> torch.Tensor:
        s = self.sigma_from_t(t).to(device=y.device, dtype=y.dtype)
        return y / torch.sqrt(1.0 + s * s)

    def x0_from_eps(self, x: torch.Tensor, eps: torch.Tensor, t: int) -> torch.Tensor:
        abar = self.abar(t).to(device=x.device, dtype=x.dtype)
        return (x - (1.0 - abar).sqrt() * eps) / abar.sqrt()


def edm_alpha_sigma(s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Diffusers mapping: α = 1/√(1+s²), σ = s·α."""
    alpha = 1.0 / torch.sqrt(s * s + 1.0)
    sigma = s * alpha
    return alpha, sigma
