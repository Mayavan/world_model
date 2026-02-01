import torch

from src.models.world_model import WorldModel


def test_model_forward_shape():
    model = WorldModel(num_actions=6)
    obs = torch.randn(2, 4, 84, 84)
    action = torch.randint(0, 6, (2,))
    out = model(obs, action)
    assert out.shape == (2, 1, 84, 84)
