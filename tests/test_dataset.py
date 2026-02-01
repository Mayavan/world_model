import json
from pathlib import Path

import numpy as np
import torch

from src.data.offline_dataset import OfflineAtariDataset


def test_dataset_shapes_and_dtypes(tmp_path: Path):
    obs = np.random.rand(10, 4, 84, 84).astype(np.float32)
    action = np.random.randint(0, 4, size=(10,), dtype=np.int64)
    next_obs = np.random.rand(10, 1, 84, 84).astype(np.float32)
    done = np.random.rand(10) > 0.5

    shard = tmp_path / "shard_000000.npz"
    np.savez_compressed(shard, obs=obs, action=action, next_obs=next_obs, done=done)

    manifest = {
        "game": "Pong",
        "steps": 10,
        "shards": [{"path": shard.name, "count": 10}],
        "total": 10,
    }
    with open(tmp_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    ds = OfflineAtariDataset(tmp_path)
    o, a, n, d = ds[0]

    assert o.shape == (4, 84, 84)
    assert n.shape == (1, 84, 84)
    assert o.dtype == torch.float32
    assert n.dtype == torch.float32
    assert a.dtype == torch.int64
    assert d.dtype == torch.bool
