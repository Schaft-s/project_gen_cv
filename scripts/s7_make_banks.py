#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gencv.noise_bank import make_noise_banks


def main() -> None:
    banks = make_noise_banks()
    for k, v in banks.items():
        print(k, tuple(v.shape))


if __name__ == "__main__":
    main()
