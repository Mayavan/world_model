from __future__ import annotations

"""Helpers for training and validation utilities."""

from collections import deque
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from src.utils.video import save_video_mp4, side_by_side


def parse_train_cli(argv: list[str]) -> tuple[Path, list[str]]:
    """Parse CLI args into (config_path, overrides)."""
    if len(argv) >= 2 and argv[1].endswith((".yaml", ".yml")):
        return Path(argv[1]), argv[2:]
    return Path("config.yaml"), argv[1:]


def run_validation(
    *,
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    motion_tau: float,
    motion_weight: float,
    motion_dilate_px: int,
) -> tuple[float, np.ndarray | None]:
    """Evaluate the model on the validation loader and return average loss and viz."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    viz_image = None
    with torch.no_grad():
        for obs, past_actions, future_actions, next_obs, _ in loader:
            obs = obs.to(device)
            past_actions = past_actions.to(device)
            future_actions = future_actions.to(device)
            next_obs = next_obs.to(device)
            logits = model(obs, future_actions, past_actions)
            last_frame = obs[:, -1:, :, :]
            assert logits.shape == next_obs.shape, "logits must match next_obs shape"
            motion = (next_obs - last_frame).abs() > motion_tau
            motion = motion.float()
            motion = F.max_pool2d(
                motion,
                kernel_size=2 * motion_dilate_px + 1,
                stride=1,
                padding=motion_dilate_px,
            )
            weights = 1.0 + motion_weight * motion
            weights = weights / weights.mean().clamp_min(1e-6)
            loss_map = F.binary_cross_entropy_with_logits(
                logits,
                next_obs,
                reduction="none",
            )
            loss = (weights * loss_map).mean()
            if viz_image is None:
                pred = torch.sigmoid(logits)
                viz_image = build_viz_image(obs, pred)
            batch_size = int(obs.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    model.train()
    if total_samples == 0:
        return 0.0, viz_image
    return total_loss / total_samples, viz_image


def build_viz_image(obs: torch.Tensor, next_pred: torch.Tensor) -> np.ndarray:
    """Create a grayscale strip of input frames plus predicted frames."""
    obs_np = obs.detach().cpu().numpy()
    pred_np = next_pred.detach().cpu().numpy()
    frames = [np.clip(f, 0.0, 1.0) for f in obs_np[0]]
    if pred_np.ndim == 3:
        pred_frames = [pred_np[0]]
    else:
        pred_frames = list(pred_np[0])
    frames.extend([np.clip(f, 0.0, 1.0) for f in pred_frames])
    return np.concatenate(frames, axis=1)


def log_prediction_image(
    *,
    tag: str,
    step: int,
    obs: torch.Tensor,
    next_pred: torch.Tensor,
    image_dir: Path,
    wandb_run,
) -> None:
    """Save and optionally log a visualization image."""
    with torch.no_grad():
        # next_pred is expected to be in [0,1] for visualization.
        image = build_viz_image(obs, next_pred)
    save_image(tag=tag, step=step, image=image, image_dir=image_dir, wandb_run=wandb_run)


def save_image(
    *,
    tag: str,
    step: int,
    image: np.ndarray,
    image_dir: Path,
    wandb_run,
) -> None:
    """Write a grayscale PNG and log to W&B if enabled."""
    path = image_dir / f"{tag}_step_{step:08d}.png"
    imageio.imwrite(path, (image * 255.0).astype(np.uint8))
    if wandb_run is not None:
        wandb.log({f"{tag}_viz": wandb.Image(str(path))}, step=step)


def run_rollout_video(
    *,
    model: torch.nn.Module,
    env,
    device: torch.device,
    horizon: int,
    fps: int,
    step: int,
    video_dir: Path,
    wandb_run,
    tag: str = "val_rollout",
) -> Path | None:
    """Run an open-loop rollout and save a side-by-side video."""
    if horizon <= 0:
        return None
    was_training = model.training
    model.eval()
    obs, _ = env.reset()
    n_past_frames = int(getattr(model, "n_past_frames", obs.shape[0]))
    n_past_actions = int(getattr(model, "n_past_actions", 0))
    n_future_frames = int(getattr(model, "n_future_frames", 1))

    # Build a real input stack with actual actions (no zero padding).
    pred_stack = obs[-n_past_frames:].copy()
    action_history: list[int] = []
    warmup_needed = max(0, n_past_frames - 1)
    while len(action_history) < warmup_needed:
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        action_history.append(int(action))
        pred_stack = next_obs[-n_past_frames:].copy()
        if terminated or truncated:
            obs, _ = env.reset()
            pred_stack = obs[-n_past_frames:].copy()
            action_history.clear()
    past_actions = deque(action_history[-n_past_actions:], maxlen=n_past_actions)

    frames: list[np.ndarray] = []
    # First frame: last input frame on both sides (gt|pred format).
    last_input = pred_stack[-1]
    frames.append(side_by_side(last_input, last_input))
    with torch.no_grad():
        for _ in range(horizon):
            future_actions = [env.action_space.sample() for _ in range(n_future_frames)]
            action = future_actions[0]
            obs_t = torch.from_numpy(pred_stack).unsqueeze(0).to(device)
            future_t = torch.tensor([future_actions], device=device, dtype=torch.int64)
            past_t = torch.tensor([list(past_actions)], device=device, dtype=torch.int64)
            logits = model(obs_t, future_t, past_t)
            pred = torch.sigmoid(logits)
            next_frame = pred[:, 0].squeeze(0).cpu().numpy()
            pred_frame = next_frame

            next_obs, _, terminated, truncated, _ = env.step(action)
            gt_frame = next_obs[-1]
            frames.append(side_by_side(gt_frame, pred_frame))

            pred_stack = np.concatenate([pred_stack[1:], pred_frame[None, ...]], axis=0)
            if n_past_actions > 0:
                past_actions.append(action)
            if terminated or truncated:
                break

    if was_training:
        model.train()

    if not frames:
        return None
    path = video_dir / f"{tag}_step_{step:08d}.mp4"
    save_video_mp4(frames, path, fps=fps)
    if wandb_run is not None:
        wandb.log({tag: wandb.Video(str(path), fps=fps, format="mp4")}, step=step)
    return path
