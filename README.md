# Atari Action-Conditioned World Model (Offline)

Offline next-frame prediction world model for Atari using Gymnasium and PyTorch. This repo provides a full runnable scaffold with dataset generation, training, and evaluation (including videos and plots). The core modeling logic can be swapped later.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### ROMs
Gymnasium Atari requires ROMs. If you see ROM errors:

```bash
pip install "gymnasium[atari,accept-rom-license]"
python -m gymnasium.utils.install_roms
```

## Commands

Generate dataset (random policy):
```bash
python -m src.data.generate_dataset --game Pong --steps 300000 --out_dir data/pong
```

Train:
```bash
python -m src.train --data_dir data/pong --game Pong
```

Evaluate:
```bash
python -m src.eval --checkpoint runs/.../ckpt.pt --game Pong
```

## Expected outputs

- Dataset shards: `data/pong/shard_*.npz`
- Manifest: `data/pong/manifest.json`
- Training run: `runs/<timestamp>_<game>/metrics.csv`, `runs/<timestamp>_<game>/checkpoints/`
- Eval artifacts: `runs/<timestamp>_<game>_eval/videos/` (MP4 + GIF), `runs/<timestamp>_<game>_eval/plots/mse_vs_horizon.png`

## Notes

- Inputs are `(B, 4, 84, 84)` float32 in `[0,1]`, actions `(B,)` int64.
- Targets are `(B, 1, 84, 84)` float32 in `[0,1]`.
- Default loss: Huber. Use `--loss mse` for MSE.
- CPU-only execution is supported via `--cpu` (slower).

## What to modify (core logic practice)

- `src/models/world_model.py`: action conditioning and alternative model variants (e.g., FiLM).
- `src/eval.py`: rollout logic, horizon aggregation, and metrics.
- `src/utils/metrics.py`: add additional metrics or perceptual losses.
