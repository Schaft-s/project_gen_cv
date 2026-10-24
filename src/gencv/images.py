from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def to_uint8(images: torch.Tensor) -> np.ndarray:
    x = images.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).numpy()


def save_grid(images: torch.Tensor, path: Path, ncol: int = 8) -> None:
    arr = to_uint8(images)
    n, h, w, c = arr.shape
    ncol = min(ncol, n)
    nrow = (n + ncol - 1) // ncol
    grid = np.ones((nrow * h, ncol * w, c), dtype=np.uint8) * 255
    for i, img in enumerate(arr):
        r, col = divmod(i, ncol)
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = img
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)


def save_pngs(images: torch.Tensor, directory: Path, prefix: str = "sample") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    arr = to_uint8(images)
    for i, img in enumerate(arr):
        Image.fromarray(img).save(directory / f"{prefix}_{i:04d}.png")
