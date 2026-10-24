import pytest
import torch

from gencv.device import get_device
from gencv.model import load_unet
from gencv.noise import VPSchedule


@pytest.fixture(scope="session")
def device():
    return get_device()


@pytest.fixture(scope="session")
def vp():
    return VPSchedule.from_config()


@pytest.fixture(scope="session")
def unet(device):
    return load_unet(device)
