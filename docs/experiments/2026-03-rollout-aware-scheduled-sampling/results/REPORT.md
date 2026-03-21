# Rollout-aware training experiment report

- Baseline: `runs/20260222_195539_coinrun`
- Experiment: `runs/20260320_rollout_aware_ss`

## Final validation metrics

- Baseline final val_mse=0.004818, val_psnr=23.203, val_ssim=0.8805
- Experiment final val_mse=0.006118, val_psnr=22.160, val_ssim=0.8602

## Final rollout metrics by selected horizons

| Horizon | Baseline MSE | Experiment MSE | Baseline PSNR | Experiment PSNR |
|---:|---:|---:|---:|---:|
| 1 | 0.000317 | 0.000986 | 34.983 | 30.060 |
| 5 | 0.010332 | 0.010105 | 19.858 | 19.955 |
| 10 | 0.010365 | 0.028426 | 19.844 | 15.463 |
| 20 | 0.030487 | 0.114955 | 15.159 | 9.395 |
| 32 | 0.157678 | 0.178039 | 8.022 | 7.495 |

## Analysis

This rollout-aware scheduled-sampling variant did **not** beat the baseline overall.

Key takeaways:
- Supervised validation regressed at the final checkpoint:
  - baseline `val_mse=0.004818`, experiment `val_mse=0.006118`
  - baseline `val_psnr=23.203`, experiment `val_psnr=22.160`
- Short-horizon rollout was mixed:
  - at horizon 5 the experiment was roughly on par with the baseline
  - at horizon 1 it was clearly worse
- Medium and long horizons degraded substantially:
  - horizon 10: experiment MSE `0.028426` vs baseline `0.010365`
  - horizon 20: experiment MSE `0.114955` vs baseline `0.030487`
  - horizon 32: experiment MSE `0.178039` vs baseline `0.157678`

Most likely interpretation:
- this auxiliary rollout loss was too weak to improve long-horizon stability,
- while the scheduled-sampling exposure introduced enough off-manifold context to hurt both one-step validation and medium-horizon rollout quality.

Recommended next follow-ups:
1. try a later/shallower scheduled-sampling ramp,
2. reduce the rollout auxiliary loss weight,
3. train directly in latent-space for multi-step consistency rather than decoded-frame MSE,
4. compare against a stronger temporal backbone before investing further in this exact schedule.

## Notes

- Compare `validation_curves.png` for overall supervised validation behavior.
- Compare `rollout_curves_final_step.png` for long-horizon open-loop behavior at the final checkpoint.
- Compare `rollout_curves_best_mean_rollout.png` for the best rollout checkpoint found within each run.
