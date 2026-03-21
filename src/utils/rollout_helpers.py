from __future__ import annotations

"""Shared helpers for rollout packing, unpacking, and stack updates."""

from typing import Sequence

import numpy as np
import torch


def stack_to_model_obs(
    pred_stack: np.ndarray,
    *,
    frame_channels: int,
    device: torch.device,
) -> torch.Tensor:
    """Pack a frame-major stack into the BCHW tensor expected by the model."""
    if frame_channels == 3:
        if pred_stack.ndim != 4 or pred_stack.shape[-1] != 3:
            raise ValueError(f"Expected RGB stack shape (T,H,W,3), got {pred_stack.shape}")
        packed = np.transpose(pred_stack, (0, 3, 1, 2)).reshape(
            pred_stack.shape[0] * 3,
            pred_stack.shape[1],
            pred_stack.shape[2],
        )
    elif frame_channels == 1:
        if pred_stack.ndim != 3:
            raise ValueError(f"Expected grayscale stack shape (T,H,W), got {pred_stack.shape}")
        packed = pred_stack
    else:
        raise ValueError(f"Unsupported frame_channels={frame_channels}")
    return torch.from_numpy(packed).unsqueeze(0).to(device=device, dtype=torch.float32)


def first_frame_from_prediction(pred: torch.Tensor, *, frame_channels: int) -> np.ndarray:
    """Extract the first predicted frame from a BCHW packed model output."""
    pred_np = pred.detach().cpu().float().clamp(0.0, 1.0).numpy()
    if pred_np.ndim != 4:
        raise ValueError(f"Expected rank-4 BCHW prediction, got shape={pred_np.shape}")
    first = pred_np[0, :frame_channels]
    if frame_channels == 1:
        return first[0]
    if frame_channels == 3:
        return np.transpose(first, (1, 2, 0))
    raise ValueError(f"Unsupported frame_channels={frame_channels}")


def latest_frame_from_env_stack(stack: np.ndarray, *, frame_channels: int) -> np.ndarray:
    """Return the newest frame from an environment stack."""
    frame = stack[-1]
    if frame_channels == 3:
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame (H,W,3), got {frame.shape}")
        return frame
    if frame_channels == 1:
        if frame.ndim != 2:
            raise ValueError(f"Expected grayscale frame (H,W), got {frame.shape}")
        return frame
    raise ValueError(f"Unsupported frame_channels={frame_channels}")


def packed_channels_to_frames(x: np.ndarray, *, frame_channels: int) -> list[np.ndarray]:
    """Unpack a channel-packed CHW tensor into a list of display-ready frames."""
    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor for frame unpacking, got shape={x.shape}")
    channels, height, width = x.shape
    if channels % frame_channels != 0:
        raise ValueError(
            f"Expected channel count divisible by frame_channels={frame_channels}, got {channels}"
        )
    frame_count = channels // frame_channels
    frames: list[np.ndarray] = []
    for idx in range(frame_count):
        frame_chw = x[idx * frame_channels : (idx + 1) * frame_channels]
        if frame_channels == 1:
            frame = np.repeat(frame_chw[0][:, :, None], 3, axis=2)
        elif frame_channels == 3:
            frame = np.transpose(frame_chw, (1, 2, 0))
        else:
            raise ValueError(f"Unsupported frame_channels={frame_channels}")
        if frame.shape[:2] != (height, width):
            raise ValueError(f"Unexpected frame shape={frame.shape}")
        frames.append(np.clip(frame, 0.0, 1.0))
    return frames


def update_rollout_stack(pred_stack: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
    """Append a new frame to the rolling context stack, dropping the oldest one."""
    if pred_stack.ndim not in {3, 4}:
        raise ValueError(f"Expected rank-3/4 pred_stack, got shape={pred_stack.shape}")
    expected_shape = pred_stack.shape[1:]
    if next_frame.shape != expected_shape:
        raise ValueError(
            f"next_frame shape mismatch: expected {expected_shape}, got {next_frame.shape}"
        )
    return np.concatenate([pred_stack[1:], next_frame[None, ...]], axis=0)


def build_future_action_tensor(
    action: int | Sequence[int],
    *,
    n_future_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize a single action or action plan into a (1, T) int64 tensor."""
    if isinstance(action, int):
        actions = [int(action)] * n_future_frames
    else:
        actions = [int(a) for a in action]
        if len(actions) == 1:
            actions = actions * n_future_frames
        elif len(actions) != n_future_frames:
            raise ValueError(
                f"Expected 1 or {n_future_frames} actions, got {len(actions)}"
            )
    return torch.tensor([actions], device=device, dtype=torch.int64)
