from __future__ import annotations

"""Procgen environment wrappers for preprocessing and frame stacking."""

import importlib.util
import logging
from collections import deque
from typing import Deque, Optional

import gymnasium as gym
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class StepAPICompatibility(gym.Wrapper):
    """Normalize reset/step outputs to Gymnasium's API."""

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}

    def step(self, action):
        out = self.env.step(action)
        if not isinstance(out, tuple):
            raise TypeError("Environment step() must return a tuple.")
        if len(out) == 5:
            return out
        if len(out) == 4:
            obs, reward, done, info = out
            return obs, reward, bool(done), False, info
        raise ValueError(f"Unexpected step() return length: {len(out)}")


class ExtractRGB(gym.ObservationWrapper):
    """Extract `rgb` frame from dict observations when needed."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        space = env.observation_space
        if not hasattr(space, "spaces") or "rgb" not in space.spaces:
            raise ValueError("ExtractRGB requires a Dict observation space with key 'rgb'.")
        self.observation_space = space.spaces["rgb"]

    def observation(self, observation):
        if isinstance(observation, dict) and "rgb" in observation:
            return observation["rgb"]
        raise ValueError("Expected dict observation containing key 'rgb'.")


class FloatNormalize(gym.ObservationWrapper):
    """Convert observations to float32 in [0, 1]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        low = np.zeros(env.observation_space.shape, dtype=np.float32)
        high = np.ones(env.observation_space.shape, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        obs = np.asarray(observation)
        if obs.dtype != np.float32:
            obs = obs.astype(np.float32)
        if obs.max() > 1.0:
            obs = obs / 255.0
        return obs


class GrayscaleResize(gym.ObservationWrapper):
    """Convert RGB observations to grayscale and resize to (84, 84)."""

    def __init__(self, env: gym.Env, shape: tuple[int, int] = (84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=np.uint8,
        )

    def observation(self, observation):
        obs = np.asarray(observation)
        if obs.ndim == 3 and obs.shape[-1] == 3:
            gray = np.asarray(Image.fromarray(obs).convert("L").resize(self.shape, Image.BILINEAR))
        elif obs.ndim == 2:
            gray = np.asarray(Image.fromarray(obs).resize(self.shape, Image.BILINEAR))
        else:
            raise ValueError(f"Unexpected observation shape for GrayscaleResize: {obs.shape}")
        return gray.astype(np.uint8)


class FrameStack(gym.Wrapper):
    """Stack last k frames along axis 0 to produce (k, H, W)."""

    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames: Deque[np.ndarray] = deque(maxlen=k)
        low = np.repeat(env.observation_space.low[None, ...], k, axis=0)
        high = np.repeat(env.observation_space.high[None, ...], k, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=0)


class LegacyResetNoSeed(gym.Wrapper):
    """Legacy Procgen reset() does not accept seed/options kwargs."""

    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        return self.env.reset(**kwargs)


class _LegacyGymRenderModeProxy:
    """Attach render_mode for shim wrappers expecting Gymnasium fields."""

    def __init__(self, env, render_mode: Optional[str]):
        self._env = env
        self.render_mode = render_mode

    def __getattr__(self, name):
        return getattr(self._env, name)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _find_env_id_in_gymnasium(game: str) -> str | None:
    env_ids = (
        f"procgen-{game.lower()}-v0",
        f"procgen:procgen-{game.lower()}-v0",
    )
    registry = gym.registry
    for env_id in env_ids:
        if env_id in registry:
            return env_id
    return None


def _find_env_id_in_legacy_gym(game: str) -> str | None:
    if not _module_available("gym"):
        return None
    import gym as old_gym  # type: ignore

    env_ids = (
        f"procgen-{game.lower()}-v0",
        f"procgen:procgen-{game.lower()}-v0",
    )
    registry = old_gym.envs.registry
    for env_id in env_ids:
        if env_id in registry:
            return env_id
    return None


def make_procgen_env(
    game: str,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    frame_stack: int = 4,
    distribution_mode: str = "easy",
    num_levels: int = 0,
) -> gym.Env:
    """Create a preprocessed Procgen environment."""
    import procgen  # noqa: F401
    logger.info("Imported procgen module and registered environments")

    env_id = _find_env_id_in_gymnasium(game)
    uses_legacy = False
    if env_id is not None:
        logger.info("Creating Procgen env via Gymnasium id '%s'", env_id)
        env = gym.make(
            env_id,
            render_mode=render_mode,
            distribution_mode=distribution_mode,
            num_levels=num_levels,
            start_level=0 if seed is None else int(seed),
        )
    else:
        legacy_env_id = _find_env_id_in_legacy_gym(game)
        if legacy_env_id is None:
            raise RuntimeError(f"Unable to find Procgen env id for game='{game}'.")
        if not _module_available("shimmy"):
            raise RuntimeError(
                "Legacy Procgen env detected but shimmy is missing. "
                "Install optional deps: `pip install -e '.[procgen]'`."
            )
        import gym as old_gym  # type: ignore
        from shimmy import GymV21CompatibilityV0

        logger.warning(
            "Using legacy Gym Procgen id '%s' via shimmy compatibility wrapper",
            legacy_env_id,
        )
        old_env = old_gym.make(
            legacy_env_id,
            distribution_mode=distribution_mode,
            num_levels=num_levels,
            start_level=0 if seed is None else int(seed),
        )
        env = GymV21CompatibilityV0(env=_LegacyGymRenderModeProxy(old_env, render_mode))
        uses_legacy = True

    logger.info("Applying Procgen preprocessing wrappers")
    if uses_legacy:
        env = LegacyResetNoSeed(env)
    env = StepAPICompatibility(env)
    if hasattr(env.observation_space, "spaces") and "rgb" in env.observation_space.spaces:
        logger.info("Extracting 'rgb' key from dict observation space")
        env = ExtractRGB(env)
    env = GrayscaleResize(env, shape=(84, 84))
    env = FloatNormalize(env)
    env = FrameStack(env, k=frame_stack)

    if seed is not None:
        logger.info("Seeding environment with seed=%d", int(seed))
        if uses_legacy:
            env.reset()
        else:
            env.reset(seed=int(seed))
        env.action_space.seed(int(seed))
    else:
        logger.info("No seed provided")

    obs_space = env.observation_space
    if not isinstance(obs_space, gym.spaces.Box) or obs_space.shape != (frame_stack, 84, 84):
        raise RuntimeError(
            "Unexpected observation space after Procgen wrappers. "
            f"Got {obs_space}."
        )

    logger.info("Procgen env ready: obs_shape=%s action_space=%s", obs_space.shape, env.action_space)
    return env
