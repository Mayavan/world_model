from __future__ import annotations

"""Evaluation utilities for open-loop rollouts and visualization."""

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import wandb

from src.envs.atari_wrappers import make_atari_env
from src.models.world_model import WorldModel
from src.utils.io import ensure_dir, timestamp_dir, init_csv, append_csv
from src.utils.video import side_by_side, save_gif, save_video_mp4
from src.utils.seed import set_seed


def rollout_open_loop(
    model: WorldModel,
    env,
    horizon: int,
    device: torch.device,
    capture_video: bool = False,
) -> Tuple[float, List[np.ndarray]]:
    """Roll out the model in open-loop for a fixed horizon."""
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

    frames: List[np.ndarray] = []
    if capture_video:
        last_input = pred_stack[-1]
        frames.append(side_by_side(last_input, last_input))

    last_mse = 0.0
    for _ in range(horizon):
        future_actions = [env.action_space.sample() for _ in range(n_future_frames)]
        action = future_actions[0]
        obs_t = torch.from_numpy(pred_stack).unsqueeze(0).to(device)
        future_t = torch.tensor([future_actions], device=device, dtype=torch.int64)
        past_t = torch.tensor([list(past_actions)], device=device, dtype=torch.int64)

        with torch.no_grad():
            logits = model(obs_t, future_t, past_t)
            pred = torch.sigmoid(logits)
        next_frame = pred[:, 0].squeeze(0).cpu().numpy()
        pred_frame = next_frame

        next_obs, _, terminated, truncated, _ = env.step(action)
        gt_frame = next_obs[-1]

        last_mse = float(np.mean((pred_frame - gt_frame) ** 2))

        if capture_video:
            frames.append(side_by_side(gt_frame, pred_frame))

        pred_stack = np.concatenate([pred_stack[1:], pred_frame[None, ...]], axis=0)
        if n_past_actions > 0:
            past_actions.append(action)

        if terminated or truncated:
            break

    return last_mse, frames


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate horizons, save videos, and plot MSE vs horizon."""
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
    n_past_actions = int(model_cfg.get("n_past_actions", 0))
    n_future_frames = int(model_cfg.get("n_future_frames", 1))
    action_embed_dim = int(model_cfg.get("action_embed_dim", 64))
    width_mult = float(model_cfg.get("width_mult", 1.0))

    env = make_atari_env(args.game, seed=args.seed, frame_stack=n_past_frames)
    num_actions = env.action_space.n

    model = WorldModel(
        num_actions=num_actions,
        n_past_frames=n_past_frames,
        n_past_actions=n_past_actions,
        n_future_frames=n_future_frames,
        action_embed_dim=action_embed_dim,
        width_mult=width_mult,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    horizons = [int(h) for h in args.horizons]
    run_dir = timestamp_dir("runs", name=f"{args.game.lower()}_eval")
    video_dir = ensure_dir(Path(run_dir) / "videos")
    plot_dir = ensure_dir(Path(run_dir) / "plots")
    metrics_path = Path(run_dir) / "metrics.csv"
    init_csv(metrics_path, ["horizon", "mse"])

    horizon_mse: Dict[int, float] = {}

    max_horizon = max(horizons)
    saved_video = False

    for horizon in horizons:
        mses: List[float] = []
        for ep in tqdm(range(args.episodes), desc=f"horizon {horizon}"):
            capture = horizon == max_horizon and ep == 0 and not saved_video
            mse, frames = rollout_open_loop(
                model=model,
                env=env,
                horizon=horizon,
                device=device,
                capture_video=capture,
            )
            mses.append(mse)
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
        append_csv(metrics_path, [horizon, horizon_mse[horizon]])
        if wandb_run is not None:
            wandb.log({"mse": horizon_mse[horizon], "horizon": horizon}, step=horizon)

    horizons_sorted = sorted(horizon_mse.keys())
    values = [horizon_mse[h] for h in horizons_sorted]

    plt.figure()
    plt.plot(horizons_sorted, values, marker="o")
    plt.xlabel("Horizon")
    plt.ylabel("MSE")
    plt.title("Open-loop MSE vs Horizon")
    plt.grid(True, linestyle="--", alpha=0.4)
    plot_path = plot_dir / "mse_vs_horizon.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Eval complete. Results in {run_dir}")
    env.close()
    if wandb_run is not None:
        try:
            wandb.log({"mse_vs_horizon": wandb.Image(str(plot_path))})
        except Exception as e:  # noqa: BLE001
            print(f"W&B image log failed: {e}")
        wandb_run.finish()


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for evaluation."""
    p = argparse.ArgumentParser(description="Evaluate Atari world model")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--game", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 30])
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="Force CPU execution")
    p.add_argument("--wandb_project", type=str, default="atari_world_model")
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
