from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

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
    obs, _ = env.reset()
    pred_stack = obs.copy()

    frames: List[np.ndarray] = []

    last_mse = 0.0
    for _ in range(horizon):
        action = env.action_space.sample()
        obs_t = torch.from_numpy(pred_stack).unsqueeze(0).to(device)
        action_t = torch.tensor([action], device=device, dtype=torch.int64)

        with torch.no_grad():
            pred = model(obs_t, action_t)
        pred_frame = pred.squeeze(0).squeeze(0).cpu().numpy()

        next_obs, _, terminated, truncated, _ = env.step(action)
        gt_frame = next_obs[-1]

        last_mse = float(np.mean((pred_frame - gt_frame) ** 2))

        if capture_video:
            frames.append(side_by_side(gt_frame, pred_frame))

        pred_stack = np.concatenate([pred_stack[1:], pred_frame[None, ...]], axis=0)

        if terminated or truncated:
            break

    return last_mse, frames


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    env = make_atari_env(args.game, seed=args.seed)
    num_actions = env.action_space.n

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = WorldModel(num_actions=num_actions, condition_mode=args.condition)
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

        horizon_mse[horizon] = float(np.mean(mses))
        append_csv(metrics_path, [horizon, horizon_mse[horizon]])

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Atari world model")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--game", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 30])
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--condition", type=str, default="concat", choices=["concat", "film"])
    return p


def main() -> None:
    args = build_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
