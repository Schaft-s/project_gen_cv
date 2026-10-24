from __future__ import annotations

import torch

from gencv.noise import VPSchedule

NFE_TO_K = {
    "ddim": {9: 9, 19: 19, 39: 39},
    "dpm_pp_2m": {9: 9, 19: 19, 39: 39},
    "heun": {9: 5, 19: 10, 39: 20},
}


def uniform_timesteps(k: int, t_max: int = 1000) -> list[int]:
    """Leading DDIM-style grid: K decreasing labels from the train table."""
    if k < 1:
        raise ValueError("k must be >= 1")
    step_ratio = t_max // k
    ts = (torch.arange(0, k) * step_ratio).round().to(torch.int64).flip(0)
    out = [int(t) for t in ts.tolist()]
    _validate_timesteps(out, t_max)
    return out


def karras_sigmas(k: int, s_min: float, s_max: float, rho: float = 7.0) -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, k)
    min_inv = s_min ** (1.0 / rho)
    max_inv = s_max ** (1.0 / rho)
    return (max_inv + ramp * (min_inv - max_inv)) ** rho


def snap_sigmas_to_timesteps(sigmas: torch.Tensor, vp: VPSchedule) -> list[int]:
    """Nearest unused t, strictly decreasing, length preserved."""
    k = int(sigmas.numel())
    t_max = vp.T
    ts: list[int] = []
    last = t_max
    for i, s in enumerate(sigmas.tolist()):
        remaining = k - i
        t = vp.nearest_t(float(s))
        t = min(t, last - 1)
        t = max(t, remaining - 1)
        ts.append(int(t))
        last = t
    _validate_timesteps(ts, t_max)
    return ts


def karras_timesteps(k: int, vp: VPSchedule, rho: float = 7.0) -> list[int]:
    s_max = float(vp.sigmas[-1].cpu())
    s_min = float(vp.sigmas[0].cpu())
    sigmas = karras_sigmas(k, s_min=s_min, s_max=s_max, rho=rho)
    return snap_sigmas_to_timesteps(sigmas, vp)


def timesteps_for_nfe(
    nfe: int,
    sampler: str,
    kind: str,
    vp: VPSchedule,
    rho: float = 7.0,
) -> list[int]:
    k = NFE_TO_K[sampler][nfe]
    if kind == "uniform":
        return uniform_timesteps(k, t_max=vp.T)
    if kind == "karras":
        return karras_timesteps(k, vp, rho=rho)
    raise ValueError(kind)


def expected_nfe(sampler: str, k: int) -> int:
    if sampler in ("ddim", "dpm_pp_2m"):
        return k
    if sampler == "heun":
        return 2 * k - 1
    raise ValueError(sampler)


def sigmas_from_timesteps(ts: list[int], vp: VPSchedule, append_zero: bool = True) -> torch.Tensor:
    s = torch.stack([vp.sigma_from_t(t).cpu() for t in ts])
    if append_zero:
        s = torch.cat([s, torch.zeros(1, dtype=s.dtype)])
    return s


def _validate_timesteps(ts: list[int], t_max: int) -> None:
    if len(ts) != len(set(ts)):
        raise ValueError(f"duplicate timesteps: {ts}")
    if any(t < 0 or t >= t_max for t in ts):
        raise ValueError(f"timestep out of range: {ts}")
    if any(ts[i] <= ts[i + 1] for i in range(len(ts) - 1)):
        raise ValueError(f"timesteps must be strictly decreasing: {ts}")
