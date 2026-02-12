# Design: Current World Model and Migration to Latent-Flow Video Generation

## 1. Scope and Goal
This document describes:
- The current implementation in this repository (data, model, training, eval, artifacts).
- A practical, step-by-step migration plan to evolve the system into a latent-flow video generation model.

The intent is incremental modernization with measurable checkpoints, not a full rewrite in one pass.

## 2. Current Implementation (As-Is)

### 2.1 Problem Formulation
- Task: action-conditioned video prediction in Procgen, trained offline.
- Inputs:
  - Past frame stack: `(B, n_past_frames, 84, 84)`, float32 in `[0,1]`.
  - Past actions: `(B, n_past_actions)`, int64.
  - Future actions: `(B, n_future_frames)`, int64.
- Target:
  - Future frames: `(B, n_future_frames, 84, 84)`, float32 in `[0,1]`.
- Default config currently uses `n_past_frames=4`, `n_past_actions=3`, `n_future_frames=4`.

### 2.2 Data Pipeline
- Dataset generation (`src/dataset/generate_dataset.py`):
  - Random policy rollouts in Procgen.
  - Stores sharded memmap `.npy` files for frames/actions/done.
  - Manifest contains shard metadata and `seq_len`.
- Offline dataset (`src/dataset/offline_dataset.py`):
  - Memory-mapped random-access reads per sample.
  - Converts uint8 frames to float32 `[0,1]`.
  - Slices sequences into `(past_frames, past_actions, future_actions, future_frames, done)`.
- Train/val split:
  - Single dataset split using seeded random permutation and `data.val_ratio`.

### 2.3 Environment and Preprocessing
- Environment creation (`src/envs/procgen_wrappers.py`):
  - Supports Gymnasium Procgen and legacy Gym Procgen via shimmy compatibility.
  - Wrapper stack:
    1. API compatibility wrapper
    2. Extract RGB key when observation is dict
    3. Grayscale + resize to `84x84`
    4. Float normalize to `[0,1]`
    5. Frame stack along axis 0

### 2.4 Model Architecture
- File: `src/models/world_model.py`.
- Model: encoder-decoder CNN with FiLM conditioning.
- Conditioning:
  - `nn.Embedding` over discrete actions.
  - Concatenate past+future action embeddings into a flat conditioning vector.
  - Inject conditioning into each encoder/decoder block via FiLM (`(1+gamma)*h + beta`).
- Encoder:
  - 4 conv stages with GroupNorm + ReLU.
  - Adds CoordConv-like XY channels.
- Decoder:
  - 4 upsampling stages + conv + GroupNorm + ReLU.
  - Output are logits for Bernoulli/BCE objective.
- Output:
  - `(B, n_future_frames, 84, 84)` logits.

### 2.5 Training Loop
- File: `src/train.py`.
- Optimizer/schedule:
  - Adam + mandatory OneCycleLR.
- Loss:
  - Binary cross-entropy with logits.
  - Motion-weighted per-pixel weighting:
    - Compute motion mask from `|target - last_input_frame| > motion_tau`.
    - Dilate mask with max-pool.
    - Weight BCE map with `1 + motion_weight * motion_mask`.
- Logging/artifacts:
  - CSV metrics, PNG prediction strips, validation metrics, periodic rollout MP4s.
  - Optional W&B logging.
- Checkpoint:
  - Model + optimizer + training step + model hyperparameters.

### 2.6 Evaluation
- File: `src/eval.py`.
- Open-loop rollout:
  - Warm-starts with real frames/actions.
  - Predicts next frame, then feeds prediction back into the input stack.
  - Environment steps with sampled actions to obtain GT for comparison.
- Metrics:
  - MSE vs horizon.
- Artifacts:
  - Side-by-side GT|prediction videos (MP4/GIF), horizon plot, CSV.

## 3. Current Strengths and Gaps

### Strengths
- Clean end-to-end baseline (data generation -> train -> eval).
- Multi-step prediction support (`n_future_frames > 1`).
- Action conditioning integrated throughout via FiLM.
- Reproducible config-driven workflow and basic test coverage.

### Gaps vs modern latent video models
- Pixel-space prediction only (no learned latent compression).
- Deterministic decoder (limited multimodal/stochastic futures).
- BCE on grayscale frames limits perceptual realism.
- No temporal latent dynamics model beyond short action-conditioned mapping.
- No transformer-based temporal modeling.
- Evaluation metric set is narrow (primarily MSE).

## 4. Target Architecture: Latent-Flow Video Generation

### 4.1 High-level target
Move from:
- `x_{t-k:t} + a_{t:t+h-1} -> x_{t+1:t+h}` in pixel space

to:
- `x -> z` via autoencoder (VAE/VQ/RAE style latent encoder)
- conditional flow model in latent space to model `p(z_{future} | z_{past}, actions)`
- latent decoder `z -> x` to reconstruct/generate frames

### 4.2 Proposed components
- `LatentEncoder` / `LatentDecoder`:
  - Compression to spatial latent map (e.g. 4x-16x downsample).
- `LatentDynamicsConditioner`:
  - Encodes action history/future and optional past latents.
- `FlowBackbone`:
  - U-Net or DiT-style network operating on latent tensors.
  - Receives time/noise level embedding + action condition.
- `FlowObjective`:
  - Flow matching or rectified-flow objective in latent space.
- Optional discriminator/perceptual loss in pixel space for sharper reconstructions.

## 5. Step-by-Step Migration Plan

### Phase 0: Stabilize Baseline and Interfaces
- Add explicit interface contracts for batch structure and tensor shapes.
- Freeze reproducible baseline metrics on a fixed checkpoint/config:
  - Train loss curve.
  - Val loss.
  - MSE@{1,5,10,30}.
- Add a model registry entry pattern so old/new models share train/eval plumbing.

Exit criteria:
- Baseline numbers recorded in a markdown table.
- Train/eval scripts can instantiate model by `model.type`.

### Phase 1: Introduce Latent Autoencoder (No Flow Yet)
- Add `src/models/latent_autoencoder.py` with:
  - `encode(frame_or_sequence) -> latent`
  - `decode(latent) -> reconstructed_frame_or_sequence`
- Train AE on existing dataset frames first (frame-wise, then short clips).
- Add reconstruction metrics:
  - MSE/PSNR/SSIM in addition to BCE.
- Keep current world model unchanged; only add offline AE training/eval scripts.

Exit criteria:
- Good reconstructions on validation videos.
- Stable latent scale/statistics tracked in logs.

### Phase 2: Predict Latents Deterministically (Bridge Step)
- Replace pixel decoder target with latent target:
  - Encode target frames with frozen AE encoder.
  - Train action-conditioned predictor `z_past + actions -> z_future` (L2 or Huber in latent space).
- Decode predicted latents for visualization and old eval compatibility.
- Keep same rollout API but route through latent predictor + decoder.

Exit criteria:
- Latent predictor beats pixel baseline on rollout MSE (after decode) or matches with lower compute.
- Multi-step latent rollouts remain stable.

### Phase 3: Add Stochastic Latent Flow Objective
- Introduce flow model module, e.g. `src/models/latent_flow.py`.
- Train with flow matching / rectified flow:
  - Inputs: noisy latent target, conditioning context from past latents + actions.
  - Network predicts velocity/transport field.
- Add sampling procedure for future latent trajectories.
- Support multiple samples per condition and sample selection metrics.

Exit criteria:
- Qualitative diversity appears for uncertain futures.
- Best-of-N and average sample metrics reported.

### Phase 4: Temporal Backbone Upgrade
- Replace simple conditioner with a stronger temporal model:
  - Option A: causal temporal transformer over latent tokens.
  - Option B: DiT-style block with action conditioning cross-attention.
- Add positional encodings for time and frame index.
- Add teacher-forcing and scheduled sampling variants in latent rollout training.

Exit criteria:
- Better long-horizon consistency (MSE and visual coherence at horizon >=30).
- No regression in short-horizon quality.

### Phase 5: Data and Objective Modernization
- Data upgrades:
  - Multi-game training option.
  - Stronger exploration policy than random for richer transitions.
  - Optional RGB mode (instead of grayscale-only) for richer visuals.
- Objective upgrades:
  - Perceptual losses (LPIPS) on decoded frames.
  - Optional adversarial fine-tuning for sharpness.
  - Action-consistency auxiliary loss (predict action from latent transitions).

Exit criteria:
- Improved visual quality and temporal consistency on held-out seeds.
- Clear tradeoff documentation between compute and quality.

### Phase 6: Inference and Productization
- Add configurable samplers (steps, guidance scale, temperature/noise).
- Export fast eval script for batch generation and benchmark dashboards.
- Add checkpoint versioning and compatibility loader for baseline checkpoints.

Exit criteria:
- One-command inference path for both deterministic and stochastic models.
- Reproducible benchmark report generated from CI or scripted run.

## 6. Repository-Level Change Map
- New files likely needed:
  - `src/models/latent_autoencoder.py`
  - `src/models/latent_predictor.py`
  - `src/models/latent_flow.py`
  - `src/train_autoencoder.py`
  - `src/train_latent_flow.py`
  - `src/eval_latent.py`
- Existing files to refactor gradually:
  - `src/train.py` (model factory and loss abstraction)
  - `src/eval.py` (support latent decoding and multi-sample eval)
  - `src/config.py` + `config.yaml` (new model/loss/sampler sections)
  - `src/utils/metrics.py` (PSNR/SSIM/LPIPS, sample-based metrics)

## 7. Risks and Mitigations
- Risk: Latent collapse or poor reconstructions.
  - Mitigation: start with reconstruction-only AE and monitor latent stats.
- Risk: Flow training instability.
  - Mitigation: begin with deterministic latent predictor bridge (Phase 2) before stochastic flow.
- Risk: Metric mismatch with perceptual quality.
  - Mitigation: add both distortion and perceptual metrics, plus fixed qualitative grids.
- Risk: Compute growth.
  - Mitigation: keep small-latent baseline configs and stage complexity progressively.

## 8. Suggested Immediate Next Sprint (Concrete)
1. Add model registry (`model.type: pixel_world_model | latent_ae | latent_predictor | latent_flow_stub`).
2. Implement and train `latent_autoencoder.py` on current dataset frames.
3. Add `eval_autoencoder.py` that writes recon videos + PSNR/SSIM.
4. Implement deterministic `latent_predictor.py` bridge and hook into existing rollout eval.
5. Lock comparison table in README for baseline vs latent bridge.

This sequence minimizes risk while steadily moving toward a modern latent-flow video generator.
