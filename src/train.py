from __future__ import annotations

"""Training loop for the Atari world model."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from src.config import apply_overrides, load_config, save_config, validate_data_config
from src.dataset.offline_dataset import create_train_val_loaders
from src.envs.atari_wrappers import make_atari_env
from src.models.world_model import WorldModel
from src.utils.io import ensure_dir, init_csv, append_csv, timestamp_dir
from src.utils.metrics import huber, mse
from src.utils.seed import set_seed
from src.utils.train_utils import (
    log_prediction_image,
    parse_train_cli,
    run_rollout_video,
    run_validation,
    save_image,
)


def train(cfg: dict) -> None:
    """Run a simple teacher-forced training loop and save checkpoints."""
    experiment = cfg["experiment"]
    scheduler_cfg = cfg["scheduler"]
    data_cfg = validate_data_config(cfg["data"])
    train_cfg = cfg["train"]

    set_seed(int(experiment["seed"]))

    run_dir_value = str(experiment["run_dir"])
    if run_dir_value:
        run_dir = ensure_dir(run_dir_value)
    else:
        name = str(experiment["name"]).lower()
        run_dir = timestamp_dir("runs", name=name)

    wandb_run = None
    wandb_cfg = experiment["wandb"]
    if wandb_cfg["mode"] != "disabled":
        try:
            wandb_run = wandb.init(
                project=wandb_cfg["project"],
                entity=wandb_cfg.get("entity", None),
                name=str(experiment["name"]),
                mode=wandb_cfg["mode"],
                dir=str(run_dir),
            )
            wandb.config.update(cfg, allow_val_change=False)
        except Exception as e:  # noqa: BLE001
            print(f"W&B init failed, continuing without logging: {e}")
            wandb_run = None

    save_config(cfg, Path(run_dir) / "resolved_config.yaml")

    env = make_atari_env(data_cfg.game, seed=int(experiment["seed"]))
    num_actions = env.action_space.n
    env.close()

    force_cpu = bool(train_cfg.get("cpu", False))
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    use_cuda = device.type == "cuda"
    train_loader, val_loader = create_train_val_loaders(
        data_cfg,
        seed=int(experiment["seed"]),
        drop_last_train=True,
        drop_last_val=False,
    )
    val_rollout_enabled = bool(train_cfg.get("val_rollout_enabled", True))
    val_rollout_horizon = int(train_cfg.get("val_rollout_horizon", 30))
    val_rollout_fps = int(train_cfg.get("val_rollout_fps", 30))
    motion_weight = float(train_cfg.get("motion_weight", 10.0))
    val_rollout_ready = (
        val_loader is not None
        and int(train_cfg.get("val_every_steps", 0)) > 0
        and val_rollout_enabled
    )
    val_env = None
    if val_rollout_ready:
        val_env = make_atari_env(data_cfg.game, seed=int(experiment["seed"]))

    model = WorldModel(num_actions=num_actions)
    model.to(device)

    if not (scheduler_cfg["enabled"] and scheduler_cfg["type"] == "onecycle"):
        raise ValueError("Scheduler must be enabled with type 'onecycle'.")
    max_lr = float(scheduler_cfg["max_lr"])
    if max_lr <= 0:
        raise ValueError("OneCycleLR requires scheduler.max_lr > 0.")

    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)

    total_steps = len(train_loader) * int(train_cfg["epochs"])
    max_steps = int(train_cfg["max_steps"])
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    if total_steps <= 0:
        raise ValueError("OneCycleLR requires at least 1 total step.")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=float(scheduler_cfg["pct_start"]),
        div_factor=float(scheduler_cfg["div_factor"]),
        final_div_factor=float(scheduler_cfg["final_div_factor"]),
    )

    ckpt_dir = ensure_dir(Path(run_dir) / "checkpoints")
    image_dir = ensure_dir(Path(run_dir) / "images")
    video_dir = ensure_dir(Path(run_dir) / "videos")
    metrics_path = Path(run_dir) / "metrics.csv"
    init_csv(metrics_path, ["epoch", "step", "loss"])
    val_metrics_path = Path(run_dir) / "val_metrics.csv"
    if val_loader is not None and int(train_cfg["val_every_steps"]) > 0:
        init_csv(val_metrics_path, ["step", "val_loss"])

    global_step = 0
    model.train()
    stop_early = False
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            obs, action, next_obs, _ = batch
            obs = obs.to(device)
            action = action.to(device)
            next_obs = next_obs.to(device)

            optimizer.zero_grad(set_to_none=True)
            next_pred = model(obs, action)
            last_frame = obs[:, -1:, :, :]
            delta_gt = next_obs - last_frame
            abs_delta = delta_gt.abs()
            motion_mask = (abs_delta > 0.02).float()
            weights = 1.0 + motion_weight * motion_mask
            if train_cfg["loss"] == "huber":
                base_loss = F.huber_loss(
                    next_pred,
                    next_obs,
                    delta=float(train_cfg["delta"]),
                    reduction="none",
                )
            else:
                base_loss = (next_pred - next_obs) ** 2
            loss = (weights * base_loss).mean()

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            if global_step % int(train_cfg["log_every"]) == 0:
                pbar.set_postfix({"loss": float(loss.item())})
                append_csv(metrics_path, [epoch, global_step, float(loss.item())])
                if wandb_run is not None:
                    wandb.log({"loss": float(loss.item()), "epoch": epoch}, step=global_step)
                log_prediction_image(
                    tag="train",
                    step=global_step,
                    obs=obs,
                    next_pred=next_pred,
                    image_dir=image_dir,
                    wandb_run=wandb_run,
                )
            if val_loader is not None and int(train_cfg["val_every_steps"]) > 0:
                if global_step % int(train_cfg["val_every_steps"]) == 0:
                    val_loss, val_viz = run_validation(
                        model=model,
                        loader=val_loader,
                        device=device,
                        loss_name=str(train_cfg["loss"]),
                        delta=float(train_cfg["delta"]),
                        motion_weight=motion_weight,
                    )
                    append_csv(val_metrics_path, [global_step, val_loss])
                    if wandb_run is not None:
                        wandb.log({"val_loss": val_loss}, step=global_step)
                    if val_viz is not None:
                        save_image(
                            tag="val",
                            step=global_step,
                            image=val_viz,
                            image_dir=image_dir,
                            wandb_run=wandb_run,
                        )
                    if val_rollout_ready and val_env is not None:
                        run_rollout_video(
                            model=model,
                            env=val_env,
                            device=device,
                            horizon=val_rollout_horizon,
                            fps=val_rollout_fps,
                            step=global_step,
                            video_dir=video_dir,
                            wandb_run=wandb_run,
                        )
            if int(train_cfg["max_steps"]) > 0 and global_step >= int(train_cfg["max_steps"]):
                stop_early = True
                break

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }
        torch.save(ckpt, ckpt_dir / f"ckpt_epoch_{epoch}.pt")
        torch.save(ckpt, Path(run_dir) / "ckpt.pt")
        if wandb_run is not None:
            wandb.log({"epoch": epoch}, step=global_step)
        if stop_early:
            break

    print(f"Training complete. Run dir: {run_dir}")
    if wandb_run is not None:
        wandb_run.finish()
    if val_env is not None:
        val_env.close()


def main() -> None:
    """Entry point for CLI."""
    config_path, overrides = parse_train_cli(sys.argv)
    cfg = load_config(config_path)
    apply_overrides(cfg, overrides)
    train(cfg)


if __name__ == "__main__":
    main()
