from __future__ import annotations

"""Training loop for the latent flow-matching world model."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from src.config import apply_overrides, load_config, save_config, validate_data_config
from src.dataset.offline_dataset import create_train_val_loaders
from src.envs.procgen_wrappers import make_procgen_env
from src.models.registry import build_model_from_config
from src.utils.contracts import validate_supervised_batch
from src.utils.io import append_csv, ensure_dir, init_csv, timestamp_dir
from src.utils.rollout_helpers import first_frame_from_prediction, update_rollout_stack
from src.utils.seed import set_seed
from src.utils.train_utils import (
    log_prediction_image,
    parse_train_cli,
    run_rollout_video,
    run_validation,
    save_image,
)


def _slice_future_frame(next_obs: torch.Tensor, *, frame_channels: int, frame_idx: int) -> torch.Tensor:
    start = frame_idx * frame_channels
    end = start + frame_channels
    return next_obs[:, start:end]


def _build_future_action_window(
    future_actions: torch.Tensor,
    *,
    start_idx: int,
    n_future_frames: int,
) -> torch.Tensor:
    window = future_actions[:, start_idx : start_idx + n_future_frames]
    if window.shape[1] == n_future_frames:
        return window
    if window.shape[1] == 0:
        pad_value = future_actions[:, -1:]
    else:
        pad_value = window[:, -1:]
    pad = pad_value.repeat(1, n_future_frames - window.shape[1])
    return torch.cat([window, pad], dim=1)


def _scheduled_sampling_probability(*, global_step: int, rollout_cfg: dict) -> float:
    start_step = int(rollout_cfg.get("start_step", 0))
    if global_step < start_step:
        return 0.0
    ramp_steps = max(1, int(rollout_cfg.get("ramp_steps", 1)))
    max_prob = float(rollout_cfg.get("max_prob", 0.0))
    progress = min(1.0, float(global_step - start_step) / float(ramp_steps))
    return max(0.0, min(max_prob, max_prob * progress))


def _compute_rollout_aware_loss(
    *,
    model: torch.nn.Module,
    obs: torch.Tensor,
    past_actions: torch.Tensor,
    future_actions: torch.Tensor,
    next_obs: torch.Tensor,
    frame_channels: int,
    global_step: int,
    rollout_cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not bool(rollout_cfg.get("enabled", False)):
        return obs.new_zeros(()), {
            "scheduled_sampling_prob": 0.0,
            "rollout_aux_loss": 0.0,
        }

    aux_weight = float(rollout_cfg.get("loss_weight", 0.0))
    if aux_weight <= 0.0:
        return obs.new_zeros(()), {
            "scheduled_sampling_prob": 0.0,
            "rollout_aux_loss": 0.0,
        }

    sampling_steps = int(rollout_cfg.get("sampling_steps", getattr(model, "sampling_steps", 16)))
    rollout_steps = int(rollout_cfg.get("rollout_steps", future_actions.shape[1]))
    rollout_steps = max(1, min(rollout_steps, future_actions.shape[1]))
    schedule_prob = _scheduled_sampling_probability(global_step=global_step, rollout_cfg=rollout_cfg)

    pred_stack = obs.reshape(obs.shape[0], -1, obs.shape[-2], obs.shape[-1])
    if past_actions.shape[1] > 0:
        action_context = past_actions.clone()
    else:
        action_context = future_actions[:, :0]

    per_step_losses: list[torch.Tensor] = []
    for step_idx in range(rollout_steps):
        action_window = _build_future_action_window(
            future_actions,
            start_idx=step_idx,
            n_future_frames=int(getattr(model, "n_future_frames", future_actions.shape[1])),
        )
        pred = model.sample_future(
            pred_stack,
            action_window,
            action_context,
            sampling_steps=sampling_steps,
        )
        pred_frame = pred[:, :frame_channels]
        target_frame = _slice_future_frame(next_obs, frame_channels=frame_channels, frame_idx=step_idx)
        per_step_losses.append(F.mse_loss(pred_frame, target_frame, reduction="mean"))

        use_model_pred = schedule_prob > 0.0
        if use_model_pred:
            mask = (torch.rand(pred_frame.shape[0], 1, 1, 1, device=pred_frame.device) < schedule_prob).to(
                pred_frame.dtype
            )
            next_context_frame = mask * pred_frame + (1.0 - mask) * target_frame
        else:
            next_context_frame = target_frame
        pred_stack = torch.cat([pred_stack[:, frame_channels:], next_context_frame], dim=1)
        if action_context.shape[1] > 0:
            next_action = future_actions[:, step_idx : step_idx + 1]
            action_context = torch.cat([action_context[:, 1:], next_action], dim=1)

    aux_loss = torch.stack(per_step_losses).mean() * aux_weight
    return aux_loss, {
        "scheduled_sampling_prob": float(schedule_prob),
        "rollout_aux_loss": float(torch.stack(per_step_losses).mean().detach().item()),
    }


def train(cfg: dict) -> None:
    """Run latent flow matching training and save checkpoints."""
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

    env = make_procgen_env(
        data_cfg.game,
        seed=int(experiment["seed"]),
        frame_stack=data_cfg.n_past_frames,
        obs_mode="rgb",
        normalize=True,
    )
    num_actions = env.action_space.n
    env.close()

    force_cpu = bool(train_cfg.get("cpu", False))
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    train_loader, val_loader = create_train_val_loaders(
        data_cfg,
        seed=int(experiment["seed"]),
        drop_last_train=True,
        drop_last_val=False,
    )
    val_rollout_enabled = bool(train_cfg.get("val_rollout_enabled", True))
    val_rollout_horizon = int(train_cfg.get("val_rollout_horizon", 32))
    val_rollout_fps = int(train_cfg.get("val_rollout_fps", 30))
    val_rollout_ready = (
        val_loader is not None
        and int(train_cfg.get("val_every_steps", 0)) > 0
        and val_rollout_enabled
    )
    val_env = None

    model_cfg = cfg.get("model", {})
    model, ckpt_model_cfg = build_model_from_config(
        model_cfg=model_cfg,
        num_actions=num_actions,
        n_past_frames=data_cfg.n_past_frames,
        n_past_actions=data_cfg.n_past_actions,
        n_future_frames=data_cfg.n_future_frames,
    )
    model.to(device)
    frame_channels = int(getattr(model, "frame_channels", 1))
    sampling_steps = int(getattr(model, "sampling_steps", 16))
    if frame_channels != 3:
        raise ValueError("latent_flow_world_model requires frame_channels=3")

    if val_rollout_ready:
        val_env = make_procgen_env(
            data_cfg.game,
            seed=int(experiment["seed"]),
            frame_stack=data_cfg.n_past_frames,
            obs_mode="rgb",
            normalize=True,
        )

    if not (scheduler_cfg["enabled"] and scheduler_cfg["type"] == "onecycle"):
        raise ValueError("Scheduler must be enabled with type 'onecycle'.")
    max_lr = float(scheduler_cfg["max_lr"])
    if max_lr <= 0:
        raise ValueError("OneCycleLR requires scheduler.max_lr > 0.")

    rollout_aware_cfg = train_cfg.get("rollout_aware", {})

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters were found for optimization.")
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    if grad_clip_norm < 0.0:
        raise ValueError("train.grad_clip_norm must be >= 0.")
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    if weight_decay < 0.0:
        raise ValueError("train.weight_decay must be >= 0.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=max_lr, weight_decay=weight_decay)

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
    plot_dir = ensure_dir(Path(run_dir) / "plots")
    metrics_path = Path(run_dir) / "metrics.csv"
    init_csv(
        metrics_path,
        [
            "epoch",
            "step",
            "loss",
            "base_loss",
            "rollout_aux_loss",
            "scheduled_sampling_prob",
            "lr",
        ],
    )
    val_metrics_path = Path(run_dir) / "val_metrics.csv"
    rollout_metrics_path = Path(run_dir) / "val_rollout_metrics.csv"
    if val_loader is not None and int(train_cfg["val_every_steps"]) > 0:
        init_csv(val_metrics_path, ["step", "val_loss", "val_mse", "val_psnr", "val_ssim"])
        if val_rollout_ready:
            init_csv(rollout_metrics_path, ["step", "horizon", "mse", "psnr"])

    global_step = 0
    model.train()
    stop_early = False
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            obs, past_actions, future_actions, next_obs, _ = batch
            obs = obs.to(device)
            past_actions = past_actions.to(device)
            future_actions = future_actions.to(device)
            next_obs = next_obs.to(device)
            validate_supervised_batch(
                obs=obs,
                past_actions=past_actions,
                future_actions=future_actions,
                next_obs=next_obs,
                n_past_frames=data_cfg.n_past_frames,
                n_past_actions=data_cfg.n_past_actions,
                n_future_frames=data_cfg.n_future_frames,
                frame_channels=frame_channels,
            )

            optimizer.zero_grad(set_to_none=True)
            losses = model.compute_flow_matching_loss(obs, future_actions, past_actions, next_obs)
            base_loss = losses["loss"]
            rollout_aux_loss, rollout_stats = _compute_rollout_aware_loss(
                model=model,
                obs=obs,
                past_actions=past_actions,
                future_actions=future_actions,
                next_obs=next_obs,
                frame_channels=frame_channels,
                global_step=global_step,
                rollout_cfg=rollout_aware_cfg,
            )
            loss = base_loss + rollout_aux_loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Encountered non-finite loss at global_step={global_step}: {float(loss)}"
                )
            loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=grad_clip_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % int(train_cfg["log_every"]) == 0:
                loss_value = float(loss.item())
                base_loss_value = float(base_loss.item())
                rollout_aux_loss_value = float(rollout_aux_loss.item())
                scheduled_sampling_prob = float(rollout_stats["scheduled_sampling_prob"])
                lr_value = float(optimizer.param_groups[0]["lr"])
                pbar.set_postfix({"loss": loss_value, "aux": rollout_aux_loss_value, "lr": lr_value})
                append_csv(
                    metrics_path,
                    [
                        epoch,
                        global_step,
                        loss_value,
                        base_loss_value,
                        rollout_aux_loss_value,
                        scheduled_sampling_prob,
                        lr_value,
                    ],
                )
                if wandb_run is not None:
                    wandb.log(
                        {
                            "loss": loss_value,
                            "base_loss": base_loss_value,
                            "rollout_aux_loss": rollout_aux_loss_value,
                            "scheduled_sampling_prob": scheduled_sampling_prob,
                            "lr": lr_value,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
                with torch.no_grad():
                    pred = model.sample_future(
                        obs,
                        future_actions,
                        past_actions,
                        sampling_steps=sampling_steps,
                    )
                log_prediction_image(
                    tag="train",
                    step=global_step,
                    obs=obs,
                    next_pred=pred,
                    image_dir=image_dir,
                    wandb_run=wandb_run,
                    frame_channels=frame_channels,
                )
            if val_loader is not None and int(train_cfg["val_every_steps"]) > 0:
                if global_step % int(train_cfg["val_every_steps"]) == 0:
                    val_metrics, val_viz = run_validation(
                        model=model,
                        loader=val_loader,
                        device=device,
                        sampling_steps=sampling_steps,
                    )
                    append_csv(
                        val_metrics_path,
                        [
                            global_step,
                            val_metrics["loss"],
                            val_metrics["mse"],
                            val_metrics["psnr"],
                            val_metrics["ssim"],
                        ],
                    )
                    if wandb_run is not None:
                        wandb.log(
                            {
                                "val_loss": val_metrics["loss"],
                                "val_mse": val_metrics["mse"],
                                "val_psnr": val_metrics["psnr"],
                                "val_ssim": val_metrics["ssim"],
                            },
                            step=global_step,
                        )
                    if val_viz is not None:
                        save_image(
                            tag="val",
                            step=global_step,
                            image=val_viz,
                            image_dir=image_dir,
                            wandb_run=wandb_run,
                        )
                    if val_rollout_ready and val_env is not None:
                        rollout_result = run_rollout_video(
                            model=model,
                            env=val_env,
                            device=device,
                            horizon=val_rollout_horizon,
                            fps=val_rollout_fps,
                            step=global_step,
                            video_dir=video_dir,
                            plot_dir=plot_dir,
                            wandb_run=wandb_run,
                            sampling_steps=sampling_steps,
                        )
                        if rollout_result is not None:
                            horizons = rollout_result["horizons"]
                            mse_by_horizon = rollout_result["mse_by_horizon"]
                            psnr_by_horizon = rollout_result["psnr_by_horizon"]
                            for horizon, mse, psnr in zip(horizons, mse_by_horizon, psnr_by_horizon):
                                append_csv(rollout_metrics_path, [global_step, horizon, mse, psnr])
                            if wandb_run is not None:
                                rollout_log = {
                                    "val_rollout_final_mse": rollout_result["final_mse"],
                                    "val_rollout_final_psnr": rollout_result["final_psnr"],
                                    "val_rollout_mean_mse": rollout_result["mean_mse"],
                                    "val_rollout_mean_psnr": rollout_result["mean_psnr"],
                                }
                                for horizon, mse, psnr in zip(horizons, mse_by_horizon, psnr_by_horizon):
                                    rollout_log[f"val_rollout_mse_h{horizon:02d}"] = mse
                                    rollout_log[f"val_rollout_psnr_h{horizon:02d}"] = psnr
                                wandb.log(rollout_log, step=global_step)
            if int(train_cfg["max_steps"]) > 0 and global_step >= int(train_cfg["max_steps"]):
                stop_early = True
                break

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "model_cfg": ckpt_model_cfg,
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
