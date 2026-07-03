"""Compact U-Net architecture for the released FireWx-FM checkpoints."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(norm_type: str, num_channels: int, norm_groups: int) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "group":
        groups = max(1, min(int(norm_groups), num_channels))
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_type: str, norm_groups: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            make_norm(norm_type, out_ch, norm_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            make_norm(norm_type, out_ch, norm_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmallFlex(nn.Module):
    def __init__(
        self,
        in_ch: int,
        base: int = 32,
        dropout: float = 0.1,
        norm_type: str = "group",
        norm_groups: int = 8,
        prior_prob: float | None = None,
        use_aux_spatial_head: bool = False,
        aux_prior_prob: float | None = None,
    ):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base, norm_type, norm_groups)
        self.enc2 = ConvBlock(base, base * 2, norm_type, norm_groups)
        self.enc3 = ConvBlock(base * 2, base * 4, norm_type, norm_groups)
        self.enc4 = ConvBlock(base * 4, base * 8, norm_type, norm_groups)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 8, base * 16, norm_type, norm_groups)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8, norm_type, norm_groups)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4, norm_type, norm_groups)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2, norm_type, norm_groups)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base, norm_type, norm_groups)
        self.drop = nn.Dropout2d(p=dropout)
        self.head = nn.Conv2d(base, 1, kernel_size=1)
        self.use_aux_spatial_head = bool(use_aux_spatial_head)
        self.aux_head = nn.Conv2d(base, 1, kernel_size=1) if self.use_aux_spatial_head else None
        if prior_prob is not None:
            prior_prob = float(min(max(prior_prob, 1e-6), 1.0 - 1e-6))
            nn.init.constant_(self.head.bias, math.log(prior_prob / (1.0 - prior_prob)))
        if self.aux_head is not None and aux_prior_prob is not None:
            aux_prior_prob = float(min(max(aux_prior_prob, 1e-6), 1.0 - 1e-6))
            nn.init.constant_(self.aux_head.bias, math.log(aux_prior_prob / (1.0 - aux_prior_prob)))

    @staticmethod
    def _match_hw(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        diff_y = ref.size(2) - x.size(2)
        diff_x = ref.size(3) - x.size(3)
        if diff_y > 0 or diff_x > 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        if diff_y < 0:
            y0 = (-diff_y) // 2
            x = x[:, :, y0 : y0 + ref.size(2), :]
        if diff_x < 0:
            x0 = (-diff_x) // 2
            x = x[:, :, :, x0 : x0 + ref.size(3)]
        return x

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self._match_hw(self.up4(b), e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._match_hw(self.up3(d4), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match_hw(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match_hw(self.up1(d2), e1), e1], dim=1))
        features = self.drop(d1)
        logits = self.head(features)
        if return_aux and self.aux_head is not None:
            return logits, self.aux_head(features)
        return logits
