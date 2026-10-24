from __future__ import annotations

from pathlib import Path

import torch

from gencv.paths import NOISE_BANKS, load_default_config

SPLIT_ORDER = ("stability", "search", "valid", "mse_test", "fid")


def make_noise_banks(directory: Path | None = None) -> dict[str, torch.Tensor]:
    cfg = load_default_config()
    directory = directory or NOISE_BANKS
    directory.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(int(cfg["seeds"]["noise_bank"]))
    ch, h, w = cfg["channels"], cfg["image_size"], cfg["image_size"]
    banks = {}
    for name in SPLIT_ORDER:
        n = int(cfg["splits"][name] if name != "stability" else cfg["splits"]["ref_stability"])
        z = torch.randn((n, ch, h, w), generator=g, dtype=torch.float32)
        path = directory / f"{name}.pt"
        torch.save(z, path)
        banks[name] = z
    return banks


def load_bank(name: str, directory: Path | None = None) -> torch.Tensor:
    directory = directory or NOISE_BANKS
    path = directory / f"{name}.pt"
    if not path.exists():
        make_noise_banks(directory)
    return torch.load(path, map_location="cpu", weights_only=True)
