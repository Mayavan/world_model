from __future__ import annotations

"""Helpers for training and validation utilities."""

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
    loss_name: str,
    delta: float,
    motion_weight: float,
) -> tuple[float, np.ndarray | None]:
    """Evaluate the model on the validation loader and return average loss and viz."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    viz_image = None
    with torch.no_grad():
        for obs, action, next_obs, _ in loader:
            obs = obs.to(device)
            action = action.to(device)
            next_obs = next_obs.to(device)
            next_pred = model(obs, action)
            last_frame = obs[:, -1:, :, :]
            delta_gt = next_obs - last_frame
            assert next_pred.shape == next_obs.shape, "next_pred must match next_obs shape"
            abs_delta = delta_gt.abs()
            motion_mask = (abs_delta > 0.02).float()
            weights = 1.0 + motion_weight * motion_mask
            if loss_name == "huber":
                base_loss = F.huber_loss(
                    next_pred,
                    next_obs,
                    delta=delta,
                    reduction="none",
                )
            else:
                base_loss = (next_pred - next_obs) ** 2
            loss = (weights * base_loss).mean()
            if viz_image is None:
                next_pred = torch.clamp(next_pred, 0.0, 1.0)
                viz_image = build_viz_image(obs, next_pred)
            batch_size = int(obs.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    model.train()
    if total_samples == 0:
        return 0.0, viz_image
    return total_loss / total_samples, viz_image


def build_viz_image(obs: torch.Tensor, next_pred: torch.Tensor) -> np.ndarray:
    """Create a grayscale strip of input frames plus predicted frame."""
    obs_np = obs.detach().cpu().numpy()
    pred_np = next_pred.detach().cpu().numpy()
    frames = [np.clip(f, 0.0, 1.0) for f in obs_np[0]]
    frames.append(np.clip(pred_np[0, 0], 0.0, 1.0))
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
        # Clamp is only for visualization; next_pred remains unconstrained.
        next_pred = torch.clamp(next_pred, 0.0, 1.0)
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
    pred_stack = obs.copy()

    frames: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(horizon):
            action = env.action_space.sample()
            obs_t = torch.from_numpy(pred_stack).unsqueeze(0).to(device)
            action_t = torch.tensor([action], device=device, dtype=torch.int64)
            next_pred = model(obs_t, action_t)
            next_frame = next_pred.squeeze(0).squeeze(0).cpu().numpy()
            # Clamp only when forming image frames/stack; next_frame is unconstrained.
            pred_frame = np.clip(next_frame, 0.0, 1.0)

            next_obs, _, terminated, truncated, _ = env.step(action)
            gt_frame = next_obs[-1]
            frames.append(side_by_side(gt_frame, pred_frame))

            pred_stack = np.concatenate([pred_stack[1:], pred_frame[None, ...]], axis=0)
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
