from __future__ import annotations

import torch
from diffusers import DDPMPipeline

from gencv.device import get_device
from gencv.paths import load_default_config


def load_unet(device: torch.device | None = None) -> torch.nn.Module:
    cfg = load_default_config()
    device = device or get_device()
    pipe = DDPMPipeline.from_pretrained(cfg["model_id"], use_safetensors=False)
    unet = pipe.unet
    unet.eval()
    unet.to(device)
    for p in unet.parameters():
        p.requires_grad_(False)
    return unet
