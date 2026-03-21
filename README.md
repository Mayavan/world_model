# Action-Conditioned World Model (Offline)

Offline latent flow-matching world model using Gymnasium and PyTorch. This repo provides a full runnable scaffold with dataset generation, autoencoder pretraining, world-model training, and evaluation (including videos and plots). The current environment setup uses Procgen, but core modeling logic is environment-agnostic.

## Setup

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

`scripts/setup_venv.sh` installs core + dev dependencies and Procgen when the
Python version is compatible. Procgen currently requires Python `<3.11`.

Later, you can activate with:
```bash
./scripts/activate_venv.sh
```

## Commands

Generate dataset (random policy):
```bash
python -m src.dataset.generate_dataset --game coinrun --steps 300000 --out_dir data/coinrun
```

Note: CoinRun/Procgen commands require Procgen support (Python `<3.11`).
Procgen env ids are registered into Gymnasium automatically via the procgen+shimmy bridge.

Train (uses `config.yaml`):
```bash
python -m src.train
```

Train ConvAutoencoder (uses `config_autoencoder.yaml`):
```bash
python -m src.train_autoencoder
```

Override config values with dot notation:
```bash
python -m src.train train.max_steps=5 model.sampling_steps=8
```

Evaluate:
```bash
python -m src.eval --checkpoint runs/.../ckpt.pt --game coinrun
```

Interactive dataset-seeded rollout:
```bash
python -m src.rollout_cli \
  --checkpoint runs/.../ckpt.pt \
  --data_dir data/coinrun_rgb \
  --sample_idx 0
```

Notes:
- Enter a single action id to repeat that action across the model's `n_future_frames` plan.
- Or enter a comma-separated action sequence of length `n_future_frames`.
- The session is seeded from an offline dataset window, then rolls forward using model predictions.

### Weights & Biases
Enable W&B logging for training and evaluation:

```bash
wandb login
python -m src.train
python -m src.eval --checkpoint runs/.../ckpt.pt --game coinrun
```

To disable logging:
```bash
python -m src.train experiment.wandb.mode=disabled
python -m src.train_autoencoder experiment.wandb.mode=disabled
python -m src.eval --checkpoint runs/.../ckpt.pt --game coinrun --wandb_mode disabled
```

## Expected outputs

- Dataset shards: `data/coinrun/shard_*_{frames,actions,done}.npy`
- Manifest: `data/coinrun/manifest.json`
- Training run: `runs/<timestamp>_<game>/metrics.csv`, `runs/<timestamp>_<game>/val_metrics.csv`, `runs/<timestamp>_<game>/val_rollout_metrics.csv`, `runs/<timestamp>_<game>/images/`, `runs/<timestamp>_<game>/videos/`, `runs/<timestamp>_<game>/plots/`, `runs/<timestamp>_<game>/checkpoints/`, `runs/<timestamp>_<game>/resolved_config.yaml`
- Autoencoder training run: `runs/<timestamp>_<game>_autoencoder/metrics.csv`, `runs/<timestamp>_<game>_autoencoder/val_metrics.csv`, `runs/<timestamp>_<game>_autoencoder/images/`, `runs/<timestamp>_<game>_autoencoder/checkpoints/`, `runs/<timestamp>_<game>_autoencoder/ckpt.pt`, `runs/<timestamp>_<game>_autoencoder/best_ckpt.pt`, `runs/<timestamp>_<game>_autoencoder/resolved_config.yaml`
- Eval artifacts: `runs/<timestamp>_<game>_eval/videos/` (MP4 + GIF), `runs/<timestamp>_<game>_eval/plots/rollout_metrics_vs_horizon.png`

## Notes

- World model inputs are RGB channel-packed frame stacks: `(B, n_past_frames*3, 84, 84)` float32 in `[0,1]`.
- Targets are `(B, n_future_frames*3, 84, 84)` float32 in `[0,1]`.
- `model.autoencoder_checkpoint` must point to a trained `ConvAutoencoder` checkpoint; the autoencoder is loaded frozen (no gradients).
- Offline dataset shards are memory-mapped `.npy` files; random access pulls one sample at a time.
- DataLoader settings are explicit in `config.yaml` (`num_workers`, `prefetch_factor`, `persistent_workers`, `pin_memory`); set `prefetch_factor: null` when `num_workers: 0`.
- Data config is validated at startup and will raise if values are inconsistent.
- Train/val split is controlled by `data.val_ratio` and validation runs every `train.val_every_steps`.
- Validation logs `val_loss`, `val_mse`, `val_psnr`, and `val_ssim`.
- At `train.log_every`, an image strip of the 4 input frames plus predicted next frame is saved (and logged to W&B if enabled).
- Each validation run can optionally trigger an open-loop rollout video (left=GT, right=prediction), plus horizon-vs-metric plots for rollout `MSE`/`PSNR`, controlled by `train.val_rollout_*` (default horizon is 32).
- Autoencoder loss uses fixed Huber reconstruction plus variance-target regularization:
  `train.var_reg_lambda * (Var(z) - train.var_target)^2` (default `train.var_target=1.0`).
- CPU-only execution is supported via `train.cpu=true`.

## What to modify (core logic practice)

- `src/models/world_model.py`: action conditioning and model variants.
- `src/eval.py`: rollout logic, horizon aggregation, and metrics.
- `src/metrics/`: image-quality and rollout metric helpers.
