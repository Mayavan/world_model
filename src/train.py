from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.offline_dataset import OfflineAtariDataset
from src.envs.atari_wrappers import make_atari_env
from src.models.world_model import WorldModel
from src.utils.io import ensure_dir, init_csv, append_csv, timestamp_dir
from src.utils.metrics import huber, mse
from src.utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Atari world model")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--game", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--loss", type=str, default="huber", choices=["huber", "mse"])
    p.add_argument("--delta", type=float, default=1.0, help="Huber delta")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_dir", type=str, default="")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--condition", type=str, default="concat", choices=["concat", "film"])
    return p


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    dataset = OfflineAtariDataset(args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=not args.cpu,
        drop_last=True,
    )

    env = make_atari_env(args.game, seed=args.seed)
    num_actions = env.action_space.n
    env.close()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_cuda = device.type == "cuda"

    model = WorldModel(num_actions=num_actions, condition_mode=args.condition)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=use_cuda)

    if args.run_dir:
        run_dir = ensure_dir(args.run_dir)
    else:
        run_dir = timestamp_dir("runs", name=args.game.lower())

    ckpt_dir = ensure_dir(Path(run_dir) / "checkpoints")
    metrics_path = Path(run_dir) / "metrics.csv"
    init_csv(metrics_path, ["epoch", "step", "loss"])

    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            obs, action, next_obs, _ = batch
            obs = obs.to(device)
            action = action.to(device)
            next_obs = next_obs.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_cuda):
                pred = model(obs, action)
                if args.loss == "huber":
                    loss = huber(pred, next_obs, delta=args.delta)
                else:
                    loss = mse(pred, next_obs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step % args.log_every == 0:
                pbar.set_postfix({"loss": float(loss.item())})
                append_csv(metrics_path, [epoch, global_step, float(loss.item())])
                print(f"step {global_step} loss {loss.item():.6f}")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
        }
        torch.save(ckpt, ckpt_dir / f"ckpt_epoch_{epoch}.pt")
        torch.save(ckpt, Path(run_dir) / "ckpt.pt")

    print(f"Training complete. Run dir: {run_dir}")


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
