# Copyright (c) 2026 LBH-123-AI
# Adapted from Comfyui_Minimax_h3_latent_Upscaler under the MIT license.
# See licenses/LBH-MIT.txt and THIRD_PARTY_NOTICES.md.

"""The released 24-channel 3D upscaler, using ComfyUI operations and full context."""

import torch
from torch import nn
import torch.nn.functional as F

from comfy import model_management, ops


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, operations):
        super().__init__()
        self.in_layers = nn.Sequential(
            operations.GroupNorm(32, channels), nn.SiLU(),
            operations.Conv3d(channels, channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), operations.Linear(64, 2 * channels))
        self.out_norm = operations.GroupNorm(32, channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Identity(), operations.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb).to(h.dtype).chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        return x + self.out_layers(h)


class TemporalConv(nn.Module):
    def __init__(self, channels, operations):
        super().__init__()
        self.norm = operations.GroupNorm(32, channels)
        self.dwconv = operations.Conv3d(channels, channels, (5, 1, 1), padding=(2, 0, 0), groups=channels)
        self.pwconv = operations.Conv3d(channels, channels, 1)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentUpscaler(nn.Module):
    def __init__(self, latents_mean, latents_std, channels=512, blocks=12, operations=ops.manual_cast):
        super().__init__()
        self.conv_in = operations.Conv3d(24, channels, 3, padding=1)
        self.embed = nn.Sequential(operations.Linear(1, 64), nn.SiLU(), operations.Linear(64, 64))
        self.in_blocks = self.make_blocks(channels, blocks, operations)
        self.out_blocks = self.make_blocks(channels, blocks, operations)
        self.norm_out = operations.GroupNorm(32, channels)
        self.conv_out = operations.Conv3d(channels, 24, 3, padding=1)
        self.register_buffer("latents_mean", latents_mean.reshape(1, 24, 1, 1, 1).clone(), persistent=False)
        self.register_buffer("latents_std", latents_std.reshape(1, 24, 1, 1, 1).clone(), persistent=False)

    @staticmethod
    def make_blocks(channels, count, operations):
        blocks = []
        for index in range(count):
            blocks.append(ResBlockEmb3D(channels, operations))
            if index % 2 == 0:
                blocks.append(TemporalConv(channels, operations))
        return nn.ModuleList(blocks)

    def forward(self, z, target_height, target_width):
        if z.shape[-2:] == (target_height, target_width):
            return z
        scale = (target_height / z.shape[-2] + target_width / z.shape[-1]) / 2
        mean = ops.cast_to_input(self.latents_mean, z)
        std = ops.cast_to_input(self.latents_std, z)
        # Preserve the reference node's D(U(N(z))) convention, even for normalized H3 input.
        x = (z - mean) / std
        emb = self.embed(z.new_full((1, 1), scale - 1))
        x = self.conv_in(x)
        for block in self.in_blocks:
            model_management.throw_exception_if_processing_interrupted()
            x = block(x, emb.expand(x.shape[0], -1)) if isinstance(block, ResBlockEmb3D) else block(x)
        x = F.interpolate(x, size=(z.shape[2], target_height, target_width), mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            model_management.throw_exception_if_processing_interrupted()
            x = block(x, emb.expand(x.shape[0], -1)) if isinstance(block, ResBlockEmb3D) else block(x)
        return self.conv_out(F.silu(self.norm_out(x))) * std + mean
