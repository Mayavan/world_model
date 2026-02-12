from __future__ import annotations

"""Video helpers for saving rollouts and creating side-by-side frames."""

from pathlib import Path
from typing import Iterable, List

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert a frame to uint8 for writing."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).astype(np.uint8)
    return frame


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    """Ensure frame is HxWx3 uint8."""
    frame = _to_uint8(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    return frame


def save_video_mp4(frames: Iterable[np.ndarray], path: str | Path, fps: int = 30) -> None:
    """Save a sequence of frames to an MP4 file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_list = [_to_rgb(f) for f in frames]
    if not frame_list:
        raise ValueError("save_video_mp4 requires at least one frame.")

    h, w = frame_list[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size=(w, h),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
    )
    writer.send(None)
    for frame in frame_list:
        if frame.shape[:2] != (h, w):
            raise ValueError("All video frames must have the same height and width.")
        writer.send(frame)
    writer.close()


def save_gif(frames: Iterable[np.ndarray], path: str | Path, fps: int = 30) -> None:
    """Save a sequence of frames to a GIF file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / max(1, fps)
    imageio.mimsave(path, [_to_uint8(f) for f in frames], duration=duration)


def _add_border(frame: np.ndarray, border: int = 1) -> np.ndarray:
    """Add a thin border around a frame to preserve edge details in video codecs."""
    if border <= 0:
        return frame
    if frame.dtype == np.uint8:
        value = 128
    else:
        value = 0.5
    pad_width = ((border, border), (border, border), (0, 0))
    return np.pad(frame, pad_width, mode="constant", constant_values=value)


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
    gt = _add_border(gt, border=1)
    pred = _add_border(pred, border=1)
    return np.concatenate([gt, pred], axis=1)
