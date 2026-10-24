from __future__ import annotations

import torch

from gencv.noise import VPSchedule


class CountedUNet:
    """U-Net wrapper: integer/float t, NFE += 1 per forward."""

    def __init__(self, unet: torch.nn.Module):
        self.unet = unet
        self.nfe = 0

    def reset(self) -> None:
        self.nfe = 0

    def __call__(self, x: torch.Tensor, t) -> torch.Tensor:
        self.nfe += 1
        b = x.shape[0]
        if not torch.is_tensor(t):
            t_b = torch.full((b,), t, device=x.device, dtype=torch.long)
        else:
            t_b = t.to(device=x.device)
            if t_b.ndim == 0:
                t_b = t_b.repeat(b)
            elif t_b.shape[0] == 1 and b > 1:
                t_b = t_b.repeat(b)
        return self.unet(x, t_b).sample


class Denoiser:
    def __init__(self, unet: CountedUNet, vp: VPSchedule):
        self.unet = unet
        self.vp = vp

    def eps_x(self, x: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        return self.unet(x, t)

    def D(self, y: torch.Tensor, s: torch.Tensor, t) -> torch.Tensor:
        """x̂₀ from EDM state y at level s. Does not call the net if s=0."""
        if float(s) == 0.0:
            return y
        scale = torch.sqrt(1.0 + s * s)
        x = y / scale
        eps = self.unet(x, t)
        return y - s * eps
