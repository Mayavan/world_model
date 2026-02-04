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


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, cond_dim: int, width_mult: float = 1.0):
        super().__init__()
        c1 = int(32 * width_mult)
        c2 = int(64 * width_mult)
        c3 = int(128 * width_mult)
        c4 = int(256 * width_mult)
        self.conv1 = nn.Conv2d(in_channels + 2, c1, kernel_size=4, stride=2)
        self.gn1 = _gn(c1)
        self.film1 = FiLM(cond_dim, c1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=4, stride=2)
        self.gn2 = _gn(c2)
        self.film2 = FiLM(cond_dim, c2)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=4, stride=2)
        self.gn3 = _gn(c3)
        self.film3 = FiLM(cond_dim, c3)
        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, stride=1)
        self.gn4 = _gn(c4)
        self.film4 = FiLM(cond_dim, c4)

    def _coord_grid(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        y = torch.linspace(-1.0, 1.0, steps=height, device=x.device, dtype=x.dtype)
        x_lin = torch.linspace(-1.0, 1.0, steps=width, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(y, x_lin, indexing="ij")
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return coords.expand(x.size(0), -1, -1, -1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        coords = self._coord_grid(x)
        x = torch.cat([x, coords], dim=1)
        x = self.conv1(x)
        x = self.film1(x, cond)
        x = F.relu(self.gn1(x))
        x = self.conv2(x)
        x = self.film2(x, cond)
        x = F.relu(self.gn2(x))
        x = self.conv3(x)
        x = self.film3(x, cond)
        x = F.relu(self.gn3(x))
        x = self.conv4(x)
        x = self.film4(x, cond)
        x = F.relu(self.gn4(x))
        return x

class Decoder(nn.Module):
    def __init__(self, cond_dim: int, out_channels: int, width_mult: float = 1.0):
        super().__init__()
        c1 = int(32 * width_mult)
        c2 = int(64 * width_mult)
        c3 = int(128 * width_mult)
        c4 = int(256 * width_mult)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")  # 6 -> 12
        self.conv1 = nn.Conv2d(c4, c3, kernel_size=3, padding=1)
        self.gn1 = _gn(c3)
        self.film1 = FiLM(cond_dim, c3)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")  # 12 -> 24
        self.conv2 = nn.Conv2d(c3, c2, kernel_size=3, padding=1)
        self.gn2 = _gn(c2)
        self.film2 = FiLM(cond_dim, c2)

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")  # 24 -> 48
        self.conv3 = nn.Conv2d(c2, c1, kernel_size=3, padding=1)
        self.gn3 = _gn(c1)
        self.film3 = FiLM(cond_dim, c1)

        self.up4 = nn.Upsample(size=(84, 84), mode="nearest")  # 48 -> 84
        self.conv4 = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.gn4 = _gn(c1)
        self.film4 = FiLM(cond_dim, c1)

        self.conv_out = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.conv1(x)
        x = self.film1(x, cond)
        x = F.relu(self.gn1(x))

        x = self.up2(x)
        x = self.conv2(x)
        x = self.film2(x, cond)
        x = F.relu(self.gn2(x))

        x = self.up3(x)
        x = self.conv3(x)
        x = self.film3(x, cond)
        x = F.relu(self.gn3(x))

        x = self.up4(x)
        x = self.conv4(x)
        x = self.film4(x, cond)
        x = F.relu(self.gn4(x))

        return self.conv_out(x)  # (B,out_channels,84,84) logits


class WorldModel(nn.Module):
    def __init__(
        self,
        num_actions: int,
        *,
        n_past_frames: int = 4,
        n_past_actions: int = 0,
        n_future_frames: int = 1,
        action_embed_dim: int = 64,
        width_mult: float = 1.0,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.n_past_frames = int(n_past_frames)
        self.n_past_actions = int(n_past_actions)
        self.n_future_frames = int(n_future_frames)
        if self.n_past_frames <= 0:
            raise ValueError("n_past_frames must be > 0")
        if self.n_future_frames <= 0:
            raise ValueError("n_future_frames must be > 0")
        if self.n_past_actions < 0:
            raise ValueError("n_past_actions must be >= 0")

        cond_dim = action_embed_dim * (self.n_past_actions + self.n_future_frames)
        self.encoder = Encoder(
            in_channels=self.n_past_frames,
            cond_dim=cond_dim,
            width_mult=width_mult,
        )
        self.action_embed = nn.Embedding(num_actions, action_embed_dim)
        self.decoder = Decoder(
            cond_dim=cond_dim,
            out_channels=self.n_future_frames,
            width_mult=width_mult,
        )

    def forward(
        self,
        obs_stack: torch.Tensor,
        future_actions: torch.Tensor,
        past_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if past_actions is None:
            if self.n_past_actions != 0:
                raise ValueError("past_actions must be provided when n_past_actions > 0")
            past_actions = future_actions[:, :0]
        future_embed = self.action_embed(future_actions)  # (B, n_future, D)
        past_embed = self.action_embed(past_actions)      # (B, n_past, D)
        cond = torch.cat([past_embed, future_embed], dim=1)
        cond = cond.reshape(cond.size(0), -1)
        z = self.encoder(obs_stack, cond)  # (B, 512, 6, 6)
        return self.decoder(z, cond)  # (B, n_future, 84, 84) logits
