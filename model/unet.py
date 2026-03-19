"""
U-Net for DDPM noise prediction -- fully from scratch.

Architecture: encoder (downsampling) -> bottleneck -> decoder (upsampling)
with skip connections, time conditioning, and self-attention at specified resolutions.

Interface: model(x, t) -> predicted_noise
  x: (B, C, H, W) noisy image in [-1, 1]
  t: (B,) integer timesteps in [0, T)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.blocks import (
    Downsample,
    ResBlock,
    SelfAttention,
    Upsample,
    timestep_embedding,
)
from model.config import ModelConfig


class UNet(nn.Module):
    """Diffusion U-Net with time conditioning and optional self-attention."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        ch = config.base_channels
        time_dim = ch * 4

        # -- Time embedding MLP --
        self.time_mlp = nn.Sequential(
            nn.Linear(ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # -- Input conv --
        self.input_conv = nn.Conv2d(config.in_channels, ch, 3, padding=1)

        # -- Encoder (down) --
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        channels = [ch]
        current_res = max(config.sample_width, config.sample_height)
        in_ch = ch

        for level, mult in enumerate(config.channel_mult):
            out_ch = ch * mult
            for _ in range(config.num_res_blocks):
                block = [ResBlock(in_ch, out_ch, time_dim, config.dropout)]
                if current_res in config.attention_resolutions:
                    block.append(SelfAttention(out_ch, config.num_heads))
                self.down_blocks.append(nn.ModuleList(block))
                channels.append(out_ch)
                in_ch = out_ch
            if level < len(config.channel_mult) - 1:
                self.down_samples.append(Downsample(in_ch))
                channels.append(in_ch)
                current_res //= 2
            else:
                self.down_samples.append(nn.Identity())

        # -- Bottleneck --
        self.mid_block1 = ResBlock(in_ch, in_ch, time_dim, config.dropout)
        self.mid_attn = SelfAttention(in_ch, config.num_heads)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_dim, config.dropout)

        # -- Decoder (up) --
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level, mult in enumerate(reversed(config.channel_mult)):
            out_ch = ch * mult
            for i in range(config.num_res_blocks + 1):
                skip_ch = channels.pop()
                block = [ResBlock(in_ch + skip_ch, out_ch, time_dim, config.dropout)]
                if current_res in config.attention_resolutions:
                    block.append(SelfAttention(out_ch, config.num_heads))
                self.up_blocks.append(nn.ModuleList(block))
                in_ch = out_ch
            if level < len(config.channel_mult) - 1:
                self.up_samples.append(Upsample(in_ch))
                current_res *= 2
            else:
                self.up_samples.append(nn.Identity())

        # -- Output --
        self.out_norm = nn.GroupNorm(32, in_ch)
        self.out_conv = nn.Conv2d(in_ch, config.out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(t, self.config.base_channels)
        t_emb = self.time_mlp(t_emb)

        h = self.input_conv(x)
        skips = [h]

        # -- Encoder --
        block_idx = 0
        for level in range(len(self.config.channel_mult)):
            for _ in range(self.config.num_res_blocks):
                layers = self.down_blocks[block_idx]
                h = layers[0](h, t_emb)
                for layer in layers[1:]:
                    h = layer(h)
                skips.append(h)
                block_idx += 1
            h = self.down_samples[level](h)
            if not isinstance(self.down_samples[level], nn.Identity):
                skips.append(h)

        # -- Bottleneck --
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # -- Decoder --
        block_idx = 0
        for level in range(len(self.config.channel_mult)):
            for _ in range(self.config.num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                layers = self.up_blocks[block_idx]
                h = layers[0](h, t_emb)
                for layer in layers[1:]:
                    h = layer(h)
                block_idx += 1
            h = self.up_samples[level](h)

        h = torch.nn.functional.silu(self.out_norm(h))
        return self.out_conv(h)
