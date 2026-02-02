# Atari Action-Conditioned World Model (Offline)

Offline next-frame prediction world model for Atari using Gymnasium and PyTorch. This repo provides a full runnable scaffold with dataset generation, training, and evaluation (including videos and plots). The core modeling logic can be swapped later.

## Setup

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

Later, you can activate with:
```bash
./scripts/activate_venv.sh
```

### ROMs
Gymnasium with Atari support is already included in dependencies. You still need to download ROMs once using AutoROM:

```bash
AutoROM --accept-license
```


## Commands

Generate dataset (random policy):
```bash
python -m src.dataset.generate_dataset --game Pong --steps 300000 --out_dir data/pong
```

Train (uses `config.yaml`):
```bash
python -m src.train
```

Override config values with dot notation:
```bash
python -m src.train optimizer.lr=1e-4 train.max_steps=5
```

Evaluate:
```bash
python -m src.eval --checkpoint runs/.../ckpt.pt --game Pong
```

### Weights & Biases
Enable W&B logging for training and evaluation:

```bash
wandb login
python -m src.train
python -m src.eval --checkpoint runs/.../ckpt.pt --game Pong
```

To disable logging:
```bash
python -m src.train experiment.wandb.mode=disabled
python -m src.eval --checkpoint runs/.../ckpt.pt --game Pong --wandb_mode disabled
```

## Expected outputs

- Dataset shards: `data/pong/shard_*_{obs,next_obs,action,done}.npy`
- Manifest: `data/pong/manifest.json`
- Training run: `runs/<timestamp>_<game>/metrics.csv`, `runs/<timestamp>_<game>/val_metrics.csv`, `runs/<timestamp>_<game>/images/`, `runs/<timestamp>_<game>/videos/`, `runs/<timestamp>_<game>/checkpoints/`, `runs/<timestamp>_<game>/resolved_config.yaml`
- Eval artifacts: `runs/<timestamp>_<game>_eval/videos/` (MP4 + GIF), `runs/<timestamp>_<game>_eval/plots/mse_vs_horizon.png`

## Notes

- Inputs are `(B, 4, 84, 84)` float32 in `[0,1]`, actions `(B,)` int64.
- Targets are `(B, 1, 84, 84)` float32 in `[0,1]`.
- Offline dataset shards are memory-mapped `.npy` files; random access pulls one sample at a time.
- DataLoader settings are explicit in `config.yaml` (`num_workers`, `prefetch_factor`, `persistent_workers`, `pin_memory`); set `prefetch_factor: null` when `num_workers: 0`.
- Data config is validated at startup and will raise if values are inconsistent.
- Train/val split is controlled by `data.val_ratio` and validation runs every `train.val_every_steps`.
- At `train.log_every`, an image strip of the 4 input frames plus predicted next frame is saved (and logged to W&B if enabled).
- Each validation run can optionally trigger an open-loop rollout video (left=GT, right=prediction) controlled by `train.val_rollout_*`.
- Default loss: Huber. Use `train.loss=mse` for MSE.
- CPU-only execution is supported via `train.cpu=true`.

## What to modify (core logic practice)

- `src/models/world_model.py`: action conditioning and model variants.
- `src/eval.py`: rollout logic, horizon aggregation, and metrics.
- `src/utils/metrics.py`: add additional metrics or perceptual losses.
