import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy, and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_env(env, seed: int) -> None:
    """Seed a Gymnasium environment and its action space."""
    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
    if hasattr(env, "action_space") and env.action_space is not None:
        try:
            env.action_space.seed(seed)
        except Exception:
            pass


def maybe_get_seed(seed: Optional[int]) -> int:
    return int(seed) if seed is not None else 0
