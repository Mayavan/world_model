# Rollout-Aware Training with Scheduled Sampling

## Goal
Evaluate whether adding a rollout-aware auxiliary objective with scheduled sampling improves long-horizon open-loop rollout metrics:

- MSE by horizon
- PSNR by horizon

## Hypothesis
The baseline model is trained primarily under teacher-forced supervision but evaluated autoregressively. This train/inference mismatch should compound over long horizons. Adding an autoregressive next-frame auxiliary loss with scheduled sampling should improve stability in open-loop rollouts, especially at medium and long horizons.

## Experiment design
- Baseline reference run: `runs/20260222_195539_coinrun`
- New branch: `exp/rollout-aware-scheduled-sampling`
- New training variant:
  - preserve the existing latent flow-matching objective
  - add a rollout-aware auxiliary loss over iterative next-frame prediction
  - replace some teacher-forced context frames with model predictions according to a scheduled-sampling probability ramp

## Status
- [x] Implement rollout-aware auxiliary training loss
- [x] Expose scheduled-sampling config knobs
- [ ] Run experiment training
- [ ] Compare baseline vs experiment metrics
- [ ] Add plots and analysis report

## Key config knobs
See `config_rollout_aware.yaml` in this folder.
