from __future__ import annotations

"""Helpers for training and validation utilities."""

from collections import deque
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch
import wandb

from src.metrics.image_quality import compute_image_quality_metrics, compute_mse_psnr
from src.metrics.rollout import save_rollout_metric_plot
from src.utils.contracts import (
    validate_model_output,
    validate_rollout_prediction,
    validate_rollout_stack,
    validate_supervised_batch,
)
from src.utils.rollout_helpers import (
    first_frame_from_prediction,
    latest_frame_from_env_stack,
    packed_channels_to_frames,
    stack_to_model_obs,
    update_rollout_stack,
)
from src.utils.video import save_video_mp4, side_by_side


def parse_train_cli(
    argv: list[str],
    *,
    default_config: str | Path = "config.yaml",
) -> tuple[Path, list[str]]:
    """Parse CLI args into (config_path, overrides)."""
    if len(argv) >= 2 and argv[1].endswith((".yaml", ".yml")):
        return Path(argv[1]), argv[2:]
    return Path(default_config), argv[1:]


def _frame_channels_for_model(model: torch.nn.Module) -> int:
    return int(getattr(model, "frame_channels", 1))



def run_validation(
    *,
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    sampling_steps: int,
) -> tuple[dict[str, float], np.ndarray | None]:
    """Evaluate validation flow loss and image-quality metrics."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0
    viz_image = None
    frame_channels = _frame_channels_for_model(model)

    with torch.no_grad():
        for obs, past_actions, future_actions, next_obs, _ in loader:
            obs = obs.to(device)
            past_actions = past_actions.to(device)
            future_actions = future_actions.to(device)
            next_obs = next_obs.to(device)
            validate_supervised_batch(
                obs=obs,
                past_actions=past_actions,
                future_actions=future_actions,
                next_obs=next_obs,
                n_past_frames=int(getattr(model, "n_past_frames", obs.shape[1])),
                n_past_actions=int(getattr(model, "n_past_actions", past_actions.shape[1])),
                n_future_frames=int(getattr(model, "n_future_frames", next_obs.shape[1])),
                frame_channels=frame_channels,
            )
            losses = model.compute_flow_matching_loss(obs, future_actions, past_actions, next_obs)
            pred = model.sample_future(
                obs,
                future_actions,
                past_actions,
                sampling_steps=sampling_steps,
            )
            validate_model_output(pred=pred, next_obs=next_obs)
            batch_metrics = compute_image_quality_metrics(pred=pred, target=next_obs, data_range=1.0)
            if viz_image is None:
                viz_image = build_viz_image(obs, pred, frame_channels=frame_channels)
            batch_size = int(obs.shape[0])
            total_loss += float(losses["loss"].item()) * batch_size
            total_mse += batch_metrics["mse"] * batch_size
            total_psnr += batch_metrics["psnr"] * batch_size
            total_ssim += batch_metrics["ssim"] * batch_size
            total_samples += batch_size
    model.train()
    if total_samples == 0:
        return {"loss": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0}, viz_image
    return {
        "loss": total_loss / total_samples,
        "mse": total_mse / total_samples,
        "psnr": total_psnr / total_samples,
        "ssim": total_ssim / total_samples,
    }, viz_image


def build_viz_image(obs: torch.Tensor, next_pred: torch.Tensor, *, frame_channels: int) -> np.ndarray:
    """Create a strip of input frames plus predicted frames."""
    obs_np = obs[0].detach().cpu().float().clamp(0.0, 1.0).numpy()
    pred_np = next_pred[0].detach().cpu().float().clamp(0.0, 1.0).numpy()
    frames = packed_channels_to_frames(obs_np, frame_channels=frame_channels)
    frames.extend(packed_channels_to_frames(pred_np, frame_channels=frame_channels))
    return np.concatenate(frames, axis=1)


def log_prediction_image(
    *,
    tag: str,
    step: int,
    obs: torch.Tensor,
    next_pred: torch.Tensor,
    image_dir: Path,
    wandb_run,
    frame_channels: int,
) -> None:
    """Save and optionally log a visualization image."""
    with torch.no_grad():
        image = build_viz_image(obs, next_pred, frame_channels=frame_channels)
    save_image(tag=tag, step=step, image=image, image_dir=image_dir, wandb_run=wandb_run)


def save_image(
    *,
    tag: str,
    step: int,
    image: np.ndarray,
    image_dir: Path,
    wandb_run,
) -> None:
    """Write a PNG and log to W&B if enabled."""
    path = image_dir / f"{tag}_step_{step:08d}.png"
    imageio.imwrite(path, (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))
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
    plot_dir: Path,
    wandb_run,
    sampling_steps: int,
    tag: str = "val_rollout",
) -> dict[str, object] | None:
    """Run an open-loop rollout, save video/plot, and return rollout metrics."""
    if horizon <= 0:
        return None
    was_training = model.training
    model.eval()
    obs, _ = env.reset()
    n_past_frames = int(getattr(model, "n_past_frames", obs.shape[0]))
    n_past_actions = int(getattr(model, "n_past_actions", 0))
    n_future_frames = int(getattr(model, "n_future_frames", 1))
    frame_channels = _frame_channels_for_model(model)

    pred_stack = obs[-n_past_frames:].copy()
    validate_rollout_stack(
        pred_stack=pred_stack,
        n_past_frames=n_past_frames,
        frame_channels=frame_channels,
    )
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
    horizons: list[int] = []
    mse_by_horizon: list[float] = []
    psnr_by_horizon: list[float] = []
    last_input = latest_frame_from_env_stack(pred_stack, frame_channels=frame_channels)
    frames.append(side_by_side(last_input, last_input))
    with torch.no_grad():
        for _ in range(horizon):
            future_actions = [env.action_space.sample() for _ in range(n_future_frames)]
            action = future_actions[0]
            obs_t = stack_to_model_obs(pred_stack, frame_channels=frame_channels, device=device)
            future_t = torch.tensor([future_actions], device=device, dtype=torch.int64)
            past_t = torch.tensor([list(past_actions)], device=device, dtype=torch.int64)
            pred = model.sample_future(
                obs_t,
                future_t,
                past_t,
                sampling_steps=sampling_steps,
            )
            validate_rollout_prediction(
                pred=pred,
                expected_channels=n_future_frames * frame_channels,
                height=pred_stack.shape[1],
                width=pred_stack.shape[2],
            )
            pred_frame = first_frame_from_prediction(pred, frame_channels=frame_channels)

            next_obs, _, terminated, truncated, _ = env.step(action)
            gt_frame = latest_frame_from_env_stack(next_obs, frame_channels=frame_channels)
            frames.append(side_by_side(gt_frame, pred_frame))
            if frame_channels == 3:
                pred_metric = torch.from_numpy(pred_frame).permute(2, 0, 1)
                gt_metric = torch.from_numpy(gt_frame).permute(2, 0, 1)
            else:
                pred_metric = torch.from_numpy(pred_frame)
                gt_metric = torch.from_numpy(gt_frame)
            mse, psnr = compute_mse_psnr(
                pred=pred_metric.to(device=device, dtype=torch.float32),
                target=gt_metric.to(device=device, dtype=torch.float32),
                data_range=1.0,
            )
            horizons.append(len(horizons) + 1)
            mse_by_horizon.append(mse)
            psnr_by_horizon.append(psnr)

            pred_stack = update_rollout_stack(pred_stack, pred_frame)
            if n_past_actions > 0:
                past_actions.append(action)
            if terminated or truncated:
                break

    if was_training:
        model.train()

    if not frames:
        return None
    video_path = video_dir / f"{tag}_step_{step:08d}.mp4"
    save_video_mp4(frames, video_path, fps=fps)
    plot_path = plot_dir / f"{tag}_metrics_step_{step:08d}.png"
    save_rollout_metric_plot(
        horizons=horizons,
        mse_values=mse_by_horizon,
        psnr_values=psnr_by_horizon,
        out_path=plot_path,
        title=f"Validation rollout metrics (step {step})",
    )
    if wandb_run is not None:
        wandb.log({tag: wandb.Video(str(video_path), fps=fps, format="mp4")}, step=step)
        wandb.log({f"{tag}_metrics": wandb.Image(str(plot_path))}, step=step)
    return {
        "video_path": video_path,
        "plot_path": plot_path,
        "horizons": horizons,
        "mse_by_horizon": mse_by_horizon,
        "psnr_by_horizon": psnr_by_horizon,
        "final_mse": mse_by_horizon[-1] if mse_by_horizon else 0.0,
        "final_psnr": psnr_by_horizon[-1] if psnr_by_horizon else 0.0,
        "mean_mse": float(np.mean(mse_by_horizon)) if mse_by_horizon else 0.0,
        "mean_psnr": float(np.mean(psnr_by_horizon)) if psnr_by_horizon else 0.0,
    }
