"""Line images: read once per kernel, resized to a fixed height, kept as uint8 arrays (ink dark on light)."""

from __future__ import annotations

import numpy as np
from PIL import Image


def load_line(path: str, height: int, max_width: int) -> np.ndarray:
    """Greyscale, resized to `height` keeping the aspect ratio, width capped at `max_width`."""
    im = Image.open(path).convert("L")
    w = max(8, round(im.width * height / im.height))
    w = min(w, max_width)
    return np.asarray(im.resize((w, height), Image.BICUBIC), dtype=np.uint8)


def load_lines(paths, height: int, max_width: int, cache: dict | None = None) -> list[np.ndarray]:
    """Arrays for every path; cached by (path, height, max_width) so later folds and runs reuse them."""
    out = []
    for p in paths:
        key = ("line", p.rsplit("/", 1)[-1], height, max_width)   # by file name: the same image under any checkout
        if cache is not None and key in cache:
            out.append(cache[key])
            continue
        a = load_line(p, height, max_width)
        if cache is not None:
            cache[key] = a
        out.append(a)
    return out
