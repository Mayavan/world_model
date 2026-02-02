from __future__ import annotations

"""Baseline action-conditioned world model (encoder + conditioner + decoder)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * channels),
        )
        # optional: init last layer near zero so FiLM starts as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # h: (B,C,H,W), cond: (B,cond_dim)
        gamma_beta = self.net(cond)              # (B,2C)
        gamma, beta = gamma_beta.chunk(2, dim=1) # (B,C), (B,C)
        gamma = gamma[:, :, None, None]
        beta  = beta[:, :, None, None]
        return (1.0 + gamma) * h + beta


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=4, stride=2)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.gn2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.gn3 = nn.GroupNorm(16, 128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.gn4 = nn.GroupNorm(32, 256)

    def _coord_grid(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        y = torch.linspace(-1.0, 1.0, steps=height, device=x.device, dtype=x.dtype)
        x_lin = torch.linspace(-1.0, 1.0, steps=width, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(y, x_lin, indexing="ij")
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return coords.expand(x.size(0), -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coords = self._coord_grid(x)
        x = torch.cat([x, coords], dim=1)
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        x = F.relu(self.gn4(self.conv4(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, cond_dim: int):
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=0)  # 6 -> 14
        self.gn1 = nn.GroupNorm(16, 128)
        self.film1 = FiLM(cond_dim, 128)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # 14 -> 28
        self.gn2 = nn.GroupNorm(16, 64)
        self.film2 = FiLM(cond_dim, 64)

        # 28 -> 84 using stride=3 is aggressive; works but can alias.
        # For Pong it’s usually OK, but if you see artifacts use two stages (28->56->84).
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=3, padding=1, output_padding=1)   # 28 -> 84
        self.gn3 = nn.GroupNorm(8, 32)
        self.film3 = FiLM(cond_dim, 32)

        self.conv_out = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.deconv1(x)
        x = self.film1(x, cond)
        x = F.relu(self.gn1(x))

        x = self.deconv2(x)
        x = self.film2(x, cond)
        x = F.relu(self.gn2(x))

        x = self.deconv3(x)
        x = self.film3(x, cond)
        x = F.relu(self.gn3(x))

        raw = self.conv_out(x)  # (B,1,84,84)
        return torch.sigmoid(raw)


class WorldModel(nn.Module):
    def __init__(
        self,
        num_actions: int,
        action_embed_dim: int = 64,
    ):
        super().__init__()
        self.num_actions = num_actions

        self.encoder = Encoder()
        self.action_embed = nn.Embedding(num_actions, action_embed_dim)

        self.decoder = Decoder(cond_dim=action_embed_dim)

    def forward(self, obs_stack: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs_stack)          # (B, 256, 6, 6)
        a = self.action_embed(action)        # (B, action_embed_dim)
        return self.decoder(z, a)            # (B, 1, 84, 84) in [0,1]
