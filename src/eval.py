from __future__ import annotations

"""Evaluation utilities for open-loop rollouts and visualization."""

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
import wandb

from src.envs.procgen_wrappers import make_procgen_env
from src.metrics.image_quality import compute_mse_psnr
from src.metrics.rollout import save_rollout_metric_plot
from src.models.registry import build_model_from_checkpoint_cfg
from src.utils.contracts import validate_rollout_prediction, validate_rollout_stack
from src.utils.io import append_csv, ensure_dir, init_csv, timestamp_dir
from src.utils.rollout_helpers import (
    first_frame_from_prediction,
    latest_frame_from_env_stack,
    stack_to_model_obs,
    update_rollout_stack,
)
from src.utils.seed import set_seed
from src.utils.video import save_gif, save_video_mp4, side_by_side



def rollout_open_loop(
    model: torch.nn.Module,
    env,
    horizon: int,
    device: torch.device,
    sampling_steps: int,
    capture_video: bool = False,
) -> Tuple[float, float, List[np.ndarray]]:
    """Roll out the model in open-loop for a fixed horizon."""
    obs, _ = env.reset()
    n_past_frames = int(getattr(model, "n_past_frames", obs.shape[0]))
    n_past_actions = int(getattr(model, "n_past_actions", 0))
    n_future_frames = int(getattr(model, "n_future_frames", 1))
    frame_channels = int(getattr(model, "frame_channels", 1))

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

    frames: List[np.ndarray] = []
    if capture_video:
        last_input = latest_frame_from_env_stack(pred_stack, frame_channels=frame_channels)
        frames.append(side_by_side(last_input, last_input))

    last_mse = 0.0
    last_psnr = 0.0
    for _ in range(horizon):
        future_actions = [env.action_space.sample() for _ in range(n_future_frames)]
        action = future_actions[0]
        obs_t = stack_to_model_obs(pred_stack, frame_channels=frame_channels, device=device)
        future_t = torch.tensor([future_actions], device=device, dtype=torch.int64)
        past_t = torch.tensor([list(past_actions)], device=device, dtype=torch.int64)

        with torch.no_grad():
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

        if frame_channels == 3:
            pred_metric = torch.from_numpy(pred_frame).permute(2, 0, 1)
            gt_metric = torch.from_numpy(gt_frame).permute(2, 0, 1)
        else:
            pred_metric = torch.from_numpy(pred_frame)
            gt_metric = torch.from_numpy(gt_frame)
        last_mse, last_psnr = compute_mse_psnr(
            pred=pred_metric.to(device=device, dtype=torch.float32),
            target=gt_metric.to(device=device, dtype=torch.float32),
            data_range=1.0,
        )

        if capture_video:
            frames.append(side_by_side(gt_frame, pred_frame))

        pred_stack = update_rollout_stack(pred_stack, pred_frame)
        if n_past_actions > 0:
            past_actions.append(action)

        if terminated or truncated:
            break

    return last_mse, last_psnr, frames


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate horizons, save videos, and plot MSE/PSNR vs horizon."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    wandb_run = None
    if args.wandb_mode != "disabled":
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
                mode=args.wandb_mode,
            )
        except Exception as e:  # noqa: BLE001
            print(f"W&B init failed, continuing without logging: {e}")
            wandb_run = None

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt.get("model_cfg", {})
    n_past_frames = int(model_cfg.get("n_past_frames", 4))
    sampling_steps = int(model_cfg.get("sampling_steps", 16))

    env = make_procgen_env(
        args.game,
        seed=args.seed,
        frame_stack=n_past_frames,
        obs_mode="rgb",
        normalize=True,
    )
    num_actions = env.action_space.n

    model = build_model_from_checkpoint_cfg(model_cfg=model_cfg, num_actions=num_actions)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    horizons = [int(h) for h in args.horizons]
    run_dir = timestamp_dir("runs", name=f"{args.game.lower()}_eval")
    video_dir = ensure_dir(Path(run_dir) / "videos")
    plot_dir = ensure_dir(Path(run_dir) / "plots")
    metrics_path = Path(run_dir) / "metrics.csv"
    init_csv(metrics_path, ["horizon", "mse", "psnr"])

    horizon_mse: Dict[int, float] = {}
    horizon_psnr: Dict[int, float] = {}

    max_horizon = max(horizons)
    saved_video = False

    for horizon in horizons:
        mses: List[float] = []
        psnrs: List[float] = []
        for ep in tqdm(range(args.episodes), desc=f"horizon {horizon}"):
            capture = horizon == max_horizon and ep == 0 and not saved_video
            mse, psnr, frames = rollout_open_loop(
                model=model,
                env=env,
                horizon=horizon,
                device=device,
                sampling_steps=sampling_steps,
                capture_video=capture,
            )
            mses.append(mse)
            psnrs.append(psnr)
            if capture and frames:
                mp4_path = video_dir / f"rollout_h{horizon}.mp4"
                gif_path = video_dir / f"rollout_h{horizon}.gif"
                save_video_mp4(frames, mp4_path, fps=args.fps)
                save_gif(frames, gif_path, fps=args.fps)
                saved_video = True
                if wandb_run is not None:
                    try:
                        wandb.log(
                            {"rollout_video": wandb.Video(str(mp4_path), fps=args.fps, format="mp4")}
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"W&B video log failed: {e}")

        horizon_mse[horizon] = float(np.mean(mses))
        horizon_psnr[horizon] = float(np.mean(psnrs))
        append_csv(metrics_path, [horizon, horizon_mse[horizon], horizon_psnr[horizon]])
        if wandb_run is not None:
            wandb.log(
                {"mse": horizon_mse[horizon], "psnr": horizon_psnr[horizon], "horizon": horizon},
                step=horizon,
            )

    horizons_sorted = sorted(horizon_mse.keys())
    mse_values = [horizon_mse[h] for h in horizons_sorted]
    psnr_values = [horizon_psnr[h] for h in horizons_sorted]
    plot_path = save_rollout_metric_plot(
        horizons=horizons_sorted,
        mse_values=mse_values,
        psnr_values=psnr_values,
        out_path=plot_dir / "rollout_metrics_vs_horizon.png",
        title="Open-loop rollout metrics vs horizon",
    )

    print(f"Eval complete. Results in {run_dir}")
    env.close()
    if wandb_run is not None:
        try:
            wandb.log({"rollout_metrics_vs_horizon": wandb.Image(str(plot_path))})
        except Exception as e:  # noqa: BLE001
            print(f"W&B image log failed: {e}")
        wandb_run.finish()


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for evaluation."""
    p = argparse.ArgumentParser(description="Evaluate world model")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--game", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 32])
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="Force CPU execution")
    p.add_argument("--wandb_project", type=str, default="world_model")
    p.add_argument("--wandb_entity", type=str, default="mayavan-projects")
    p.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode (use offline to avoid network calls)",
    )
    return p


def main() -> None:
    """Entry point for CLI."""
    args = build_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
