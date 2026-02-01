from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import imageio.v2 as imageio
import numpy as np


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).astype(np.uint8)
    return frame


def save_video_mp4(frames: Iterable[np.ndarray], path: str | Path, fps: int = 30) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(_to_uint8(frame))


def save_gif(frames: Iterable[np.ndarray], path: str | Path, fps: int = 30) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / max(1, fps)
    imageio.mimsave(path, [_to_uint8(f) for f in frames], duration=duration)


def side_by_side(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Stack two frames horizontally. Frames are HxW or HxWxC in [0,1] or uint8."""
    if gt.ndim == 2:
        gt = np.repeat(gt[:, :, None], 3, axis=2)
    if pred.ndim == 2:
        pred = np.repeat(pred[:, :, None], 3, axis=2)
    if gt.shape[2] == 1:
        gt = np.repeat(gt, 3, axis=2)
    if pred.shape[2] == 1:
        pred = np.repeat(pred, 3, axis=2)
    return np.concatenate([gt, pred], axis=1)
