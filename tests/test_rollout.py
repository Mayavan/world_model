import numpy as np
import torch
from gymnasium import spaces

from src.eval import rollout_open_loop
from src.models.world_model import WorldModel


class FakeEnv:
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self._rng = np.random.default_rng(0)
        self._step = 0

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        obs = self._rng.random((4, 84, 84), dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = self._rng.random((4, 84, 84), dtype=np.float32)
        reward = 0.0
        terminated = self._step >= 5
        truncated = False
        return obs, reward, terminated, truncated, {}


def test_rollout_small_horizon():
    env = FakeEnv()
    model = WorldModel(num_actions=4)
    device = torch.device("cpu")
    mse, frames = rollout_open_loop(model, env, horizon=3, device=device, capture_video=True)
    assert isinstance(mse, float)
    assert len(frames) <= 3
