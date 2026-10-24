from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

from gencv.device import sync


@contextmanager
def timed(device, label: str = ""):
    sync(device)
    t0 = time.perf_counter()
    yield
    sync(device)
    dt = time.perf_counter() - t0
    if label:
        print(f"[time] {label}: {dt:.2f}s")
    timed.last = dt  # type: ignore[attr-defined]


timed.last = 0.0  # type: ignore[attr-defined]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
