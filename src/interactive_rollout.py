from __future__ import annotations

"""Interactive dataset-seeded rollout session for action-conditioned world-model inference."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.dataset.offline_dataset import OfflineDataset
from src.models.registry import build_model_from_checkpoint_cfg
from src.utils.contracts import validate_rollout_prediction, validate_rollout_stack
from src.utils.rollout_helpers import (
    build_future_action_tensor,
    first_frame_from_prediction,
    packed_channels_to_frames,
    update_rollout_stack,
)


@dataclass
class SeedSample:
    """Dataset-derived seed context and optional recorded continuation metadata."""

    sample_idx: int
    pred_stack: np.ndarray
    past_actions: list[int]
    recorded_future_actions: list[int]
    recorded_future_frames: np.ndarray


class InteractiveRolloutSession:
    """Stateful interactive rollout seeded from an offline dataset sample."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        data_dir: str | Path,
        device: str | torch.device | None = None,
        sampling_steps: int | None = None,
    ):
        self.checkpoint = Path(checkpoint)
        self.data_dir = Path(data_dir)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        if device is None:
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved_device = torch.device(device)
        self.device = resolved_device

        ckpt = torch.load(self.checkpoint, map_location=self.device)
        model_cfg = ckpt.get("model_cfg")
        if not isinstance(model_cfg, dict):
            raise ValueError("Checkpoint missing dict model_cfg metadata.")
        self.model_cfg = dict(model_cfg)
        self.n_past_frames = int(self.model_cfg.get("n_past_frames", 4))
        self.n_past_actions = int(self.model_cfg.get("n_past_actions", 0))
        self.n_future_frames = int(self.model_cfg.get("n_future_frames", 1))
        self.frame_channels = int(self.model_cfg.get("frame_channels", 3))
        self.sampling_steps = (
            int(sampling_steps)
            if sampling_steps is not None
            else int(self.model_cfg.get("sampling_steps", 16))
        )

        num_actions = self._infer_num_actions()
        self.model = build_model_from_checkpoint_cfg(model_cfg=self.model_cfg, num_actions=num_actions)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()
        self.num_actions = int(getattr(self.model, "num_actions", num_actions))

        self.dataset = OfflineDataset(
            self.data_dir,
            n_past_frames=self.n_past_frames,
            n_past_actions=self.n_past_actions,
            n_future_frames=self.n_future_frames,
        )

        self.seed: SeedSample | None = None
        self.pred_stack: np.ndarray | None = None
        self.past_actions: deque[int] = deque(maxlen=self.n_past_actions)
        self.step_index = 0
        self.action_history: list[int] = []
        self.predicted_frames: list[np.ndarray] = []

    def _infer_num_actions(self) -> int:
        game = str(self.model_cfg.get("game", "coinrun"))
        if game == "coinrun":
            return 15
        raise ValueError(
            "Unable to infer num_actions from checkpoint metadata. "
            "Add game/num_actions metadata or extend inference logic."
        )

    def load_seed(self, sample_idx: int) -> SeedSample:
        """Load a dataset sample and initialize rollout state from it."""
        obs, past_actions, future_actions, next_obs, _ = self.dataset[sample_idx]
        pred_stack = self._obs_tensor_to_stack(obs)
        validate_rollout_stack(
            pred_stack=pred_stack,
            n_past_frames=self.n_past_frames,
            frame_channels=self.frame_channels,
        )
        seed = SeedSample(
            sample_idx=int(sample_idx),
            pred_stack=pred_stack.copy(),
            past_actions=[int(a) for a in past_actions.tolist()],
            recorded_future_actions=[int(a) for a in future_actions.tolist()],
            recorded_future_frames=self._obs_tensor_to_stack(next_obs),
        )
        self.seed = seed
        self.pred_stack = seed.pred_stack.copy()
        self.past_actions = deque(seed.past_actions, maxlen=self.n_past_actions)
        self.step_index = 0
        self.action_history = []
        self.predicted_frames = []
        return seed

    def reset(self) -> None:
        """Reset the session to the last loaded seed state."""
        if self.seed is None:
            raise RuntimeError("No seed loaded. Call load_seed(sample_idx) first.")
        self.pred_stack = self.seed.pred_stack.copy()
        self.past_actions = deque(self.seed.past_actions, maxlen=self.n_past_actions)
        self.step_index = 0
        self.action_history = []
        self.predicted_frames = []

    def step(self, action: int | Sequence[int]) -> dict[str, object]:
        """Roll out one model step from the current context using the supplied action plan."""
        if self.pred_stack is None:
            raise RuntimeError("No seed loaded. Call load_seed(sample_idx) first.")
        future_t = build_future_action_tensor(
            action,
            n_future_frames=self.n_future_frames,
            device=self.device,
        )
        self._validate_actions(future_t)
        obs_t = self._stack_to_packed_tensor(self.pred_stack)
        if self.n_past_actions > 0:
            past_action_values = list(self.past_actions)
        else:
            past_action_values = []
        past_t = torch.tensor([past_action_values], device=self.device, dtype=torch.int64)

        with torch.no_grad():
            pred = self.model.sample_future(
                obs_t,
                future_t,
                past_t,
                sampling_steps=self.sampling_steps,
            )
            validate_rollout_prediction(
                pred=pred,
                expected_channels=self.n_future_frames * self.frame_channels,
                height=self.pred_stack.shape[1],
                width=self.pred_stack.shape[2],
            )
        pred_frame = first_frame_from_prediction(pred, frame_channels=self.frame_channels)
        self.pred_stack = update_rollout_stack(self.pred_stack, pred_frame)
        first_action = int(future_t[0, 0].item())
        if self.n_past_actions > 0:
            self.past_actions.append(first_action)
        self.step_index += 1
        self.action_history.append(first_action)
        self.predicted_frames.append(pred_frame.copy())

        gt_frame = None
        gt_action = None
        if self.seed is not None and self.step_index <= self.seed.recorded_future_frames.shape[0]:
            gt_frame = self.seed.recorded_future_frames[self.step_index - 1]
        if self.seed is not None and self.step_index <= len(self.seed.recorded_future_actions):
            gt_action = self.seed.recorded_future_actions[self.step_index - 1]

        return {
            "step_index": self.step_index,
            "action_plan": future_t[0].detach().cpu().tolist(),
            "applied_action": first_action,
            "predicted_frame": pred_frame,
            "gt_frame": gt_frame,
            "gt_action": gt_action,
            "matches_recorded_action": gt_action == first_action if gt_action is not None else None,
        }

    def step_many(self, actions: Sequence[int] | Sequence[Sequence[int]]) -> list[dict[str, object]]:
        """Apply multiple interactive steps in sequence."""
        results: list[dict[str, object]] = []
        for action in actions:
            results.append(self.step(action))
        return results

    def render_context_strip(self) -> np.ndarray:
        """Render the current context stack as a horizontal visualization strip."""
        if self.pred_stack is None:
            raise RuntimeError("No seed loaded. Call load_seed(sample_idx) first.")
        if self.frame_channels == 3:
            frames = [np.clip(frame, 0.0, 1.0) for frame in self.pred_stack]
        elif self.frame_channels == 1:
            frames = [np.repeat(frame[:, :, None], 3, axis=2) for frame in self.pred_stack]
        else:
            raise ValueError(f"Unsupported frame_channels={self.frame_channels}")
        return np.concatenate(frames, axis=1)

    def render_predicted_rollout_strip(self) -> np.ndarray | None:
        """Render predicted frames produced so far in the current session."""
        if not self.predicted_frames:
            return None
        if self.frame_channels == 3:
            frames = [np.clip(frame, 0.0, 1.0) for frame in self.predicted_frames]
        elif self.frame_channels == 1:
            frames = [np.repeat(frame[:, :, None], 3, axis=2) for frame in self.predicted_frames]
        else:
            raise ValueError(f"Unsupported frame_channels={self.frame_channels}")
        return np.concatenate(frames, axis=1)

    def decode_prediction_tensor(self, pred: torch.Tensor) -> list[np.ndarray]:
        """Unpack a packed prediction tensor into a list of frames."""
        pred_np = pred[0].detach().cpu().float().clamp(0.0, 1.0).numpy()
        return packed_channels_to_frames(pred_np, frame_channels=self.frame_channels)

    def _stack_to_packed_tensor(self, stack: np.ndarray) -> torch.Tensor:
        if self.frame_channels == 3:
            packed = np.transpose(stack, (0, 3, 1, 2)).reshape(
                stack.shape[0] * 3,
                stack.shape[1],
                stack.shape[2],
            )
        elif self.frame_channels == 1:
            packed = stack
        else:
            raise ValueError(f"Unsupported frame_channels={self.frame_channels}")
        return torch.from_numpy(packed).unsqueeze(0).to(device=self.device, dtype=torch.float32)

    def _obs_tensor_to_stack(self, x: torch.Tensor) -> np.ndarray:
        obs_np = x.detach().cpu().float().numpy()
        if obs_np.ndim != 3:
            raise ValueError(f"Expected CHW tensor, got shape={obs_np.shape}")
        channels, height, width = obs_np.shape
        expected = self.frame_channels
        if channels % expected != 0:
            raise ValueError(
                f"Expected channels divisible by frame_channels={expected}, got {channels}"
            )
        frames = channels // expected
        if self.frame_channels == 3:
            chw = obs_np.reshape(frames, 3, height, width)
            return np.transpose(chw, (0, 2, 3, 1))
        if self.frame_channels == 1:
            return obs_np.reshape(frames, height, width)
        raise ValueError(f"Unsupported frame_channels={self.frame_channels}")

    def _validate_actions(self, future_t: torch.Tensor) -> None:
        if future_t.ndim != 2 or future_t.shape[1] != self.n_future_frames:
            raise ValueError(
                f"Expected future action tensor shape (1, {self.n_future_frames}), got {future_t.shape}"
            )
        min_action = int(future_t.min().item())
        max_action = int(future_t.max().item())
        if min_action < 0 or max_action >= self.num_actions:
            raise ValueError(
                f"Action out of bounds for num_actions={self.num_actions}: "
                f"min={min_action} max={max_action}"
            )
