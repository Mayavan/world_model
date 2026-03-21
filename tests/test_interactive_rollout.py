from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.interactive_rollout import InteractiveRolloutSession
from src.models.world_model import WorldModel
from src.utils.rollout_helpers import build_future_action_tensor, update_rollout_stack


def _write_tiny_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(2, 8, 84, 84, 3), dtype=np.uint8)
    actions = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 1, 2, 2, 3, 3, 4, 4],
        ],
        dtype=np.int64,
    )
    done = np.zeros((2, 8), dtype=np.bool_)
    np.save(data_dir / "shard_000000_frames.npy", frames)
    np.save(data_dir / "shard_000000_actions.npy", actions)
    np.save(data_dir / "shard_000000_done.npy", done)
    manifest = {
        "game": "coinrun",
        "steps": 16,
        "seq_len": 8,
        "shards": [
            {
                "id": 0,
                "count": 2,
                "frames": "shard_000000_frames.npy",
                "actions": "shard_000000_actions.npy",
                "done": "shard_000000_done.npy",
            }
        ],
        "total": 2,
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_checkpoint(path: Path) -> None:
    model = WorldModel(
        num_actions=15,
        autoencoder_model_cfg={
            "type": "conv_autoencoder",
            "in_channels": 3,
            "image_height": 84,
            "image_width": 84,
            "base_channels": 16,
            "latent_channels": 8,
            "downsample_factor": 8,
            "decoder_type": "upsample_conv",
        },
        n_past_frames=4,
        n_past_actions=3,
        n_future_frames=4,
        sampling_steps=2,
        width_mult=0.5,
    )
    ckpt = {
        "model": model.state_dict(),
        "model_cfg": {
            "type": "latent_flow_world_model",
            "game": "coinrun",
            "n_past_frames": 4,
            "n_past_actions": 3,
            "n_future_frames": 4,
            "frame_channels": 3,
            "action_embed_dim": 64,
            "time_embed_dim": 128,
            "sampling_steps": 2,
            "width_mult": 0.5,
            "autoencoder_model_cfg": model.autoencoder_model_cfg,
        },
    }
    torch.save(ckpt, path)


def test_build_future_action_tensor_repeats_single_action():
    tensor = build_future_action_tensor(3, n_future_frames=4, device=torch.device("cpu"))
    assert tensor.shape == (1, 4)
    assert tensor.tolist() == [[3, 3, 3, 3]]


def test_update_rollout_stack_appends_new_frame():
    stack = np.zeros((4, 8, 8, 3), dtype=np.float32)
    next_frame = np.ones((8, 8, 3), dtype=np.float32)
    updated = update_rollout_stack(stack, next_frame)
    assert updated.shape == stack.shape
    assert np.allclose(updated[-1], 1.0)
    assert np.allclose(updated[0], 0.0)


def test_interactive_rollout_session_step(tmp_path: Path):
    data_dir = tmp_path / "coinrun_rgb"
    ckpt_path = tmp_path / "ckpt.pt"
    _write_tiny_dataset(data_dir)
    _write_checkpoint(ckpt_path)

    session = InteractiveRolloutSession(
        checkpoint=ckpt_path,
        data_dir=data_dir,
        device="cpu",
        sampling_steps=2,
    )
    seed = session.load_seed(0)
    assert seed.sample_idx == 0
    assert seed.pred_stack.shape == (4, 84, 84, 3)
    assert seed.past_actions == [0, 1, 2]
    assert seed.recorded_future_actions == [3, 4, 5, 6]

    result = session.step(4)
    assert result["step_index"] == 1
    assert result["action_plan"] == [4, 4, 4, 4]
    assert int(result["applied_action"]) == 4
    pred_frame = np.asarray(result["predicted_frame"])
    assert pred_frame.shape == (84, 84, 3)
    assert np.isfinite(pred_frame).all()
    assert result["gt_frame"] is not None
    assert result["gt_action"] == 3
    assert result["matches_recorded_action"] is False

    context_strip = session.render_context_strip()
    assert context_strip.shape == (84, 84 * 4, 3)

    rollout_strip = session.render_predicted_rollout_strip()
    assert rollout_strip is not None
    assert rollout_strip.shape == (84, 84, 3)

    session.reset()
    assert session.step_index == 0
    assert session.action_history == []
