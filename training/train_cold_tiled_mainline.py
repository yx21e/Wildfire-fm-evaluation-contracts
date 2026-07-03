from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from eval_metrics import metric_bundle
from train_utils import set_seed


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def integral_image(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(np.int32), ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)


def rect_sum(ii: np.ndarray, top: int, left: int, height: int, width: int) -> int:
    bottom = top + height
    right = left + width
    return int(ii[bottom, right] - ii[top, right] - ii[bottom, left] + ii[top, left])


def centered_tile_origin(y: int, x: int, tile_h: int, tile_w: int, h: int, w: int) -> Tuple[int, int]:
    top = min(max(y - tile_h // 2, 0), h - tile_h)
    left = min(max(x - tile_w // 2, 0), w - tile_w)
    return int(top), int(left)


def containing_tile_origin(y: int, x: int, tile_h: int, tile_w: int, h: int, w: int, rng: random.Random) -> Tuple[int, int]:
    max_top = max(h - tile_h, 0)
    max_left = max(w - tile_w, 0)
    top_min = max(0, y - tile_h + 1)
    top_max = min(y, max_top)
    left_min = max(0, x - tile_w + 1)
    left_max = min(x, max_left)
    if top_min > top_max or left_min > left_max:
        return centered_tile_origin(y, x, tile_h, tile_w, h, w)
    return int(rng.randint(top_min, top_max)), int(rng.randint(left_min, left_max))


def positive_tile_origins(
    mask: np.ndarray,
    tile_h: int,
    tile_w: int,
    max_tiles: int,
    rng: random.Random,
    placement: str = "center",
) -> List[Tuple[int, int]]:
    ys, xs = np.where(mask > 0.5)
    if ys.size == 0:
        return []
    origin_fn = containing_tile_origin if placement == "random_containing" else centered_tile_origin
    origins = []
    seen = set()
    indices = list(range(int(ys.size)))
    rng.shuffle(indices)
    for i in indices:
        y = int(ys[i])
        x = int(xs[i])
        if placement == "random_containing":
            origin = origin_fn(y, x, tile_h, tile_w, mask.shape[0], mask.shape[1], rng)
        else:
            origin = origin_fn(y, x, tile_h, tile_w, mask.shape[0], mask.shape[1])
        if origin in seen:
            continue
        seen.add(origin)
        origins.append(origin)
        if max_tiles > 0 and len(origins) >= max_tiles:
            break
    rng.shuffle(origins)
    if max_tiles > 0:
        origins = origins[:max_tiles]
    return origins


def negative_tile_origins(
    mask: np.ndarray,
    tile_h: int,
    tile_w: int,
    desired: int,
    rng: random.Random,
    forbidden: Iterable[Tuple[int, int]] = (),
) -> List[Tuple[int, int]]:
    h, w = mask.shape
    ii = integral_image(mask)
    forbidden_set = set(forbidden)
    out: List[Tuple[int, int]] = []
    seen = set(forbidden_set)
    max_top = max(h - tile_h, 0)
    max_left = max(w - tile_w, 0)
    max_attempts = max(2000, desired * 50)
    attempts = 0
    while len(out) < desired and attempts < max_attempts:
        attempts += 1
        top = rng.randint(0, max_top)
        left = rng.randint(0, max_left)
        origin = (top, left)
        if origin in seen:
            continue
        if rect_sum(ii, top, left, tile_h, tile_w) != 0:
            continue
        seen.add(origin)
        out.append(origin)
    return out


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


def normalization_config(config: Dict[str, object]) -> Dict[str, object]:
    raw = config.get("input_normalization", {})
    if raw is True:
        return {"enabled": True}
    if isinstance(raw, dict):
        return raw
    return {"enabled": False}


def compute_input_normalization_stats(
    rows: List[Dict[str, str]],
    continuous_channel_indices: List[int],
    eps: float,
) -> Dict[str, object]:
    count: Dict[int, int] = {int(idx): 0 for idx in continuous_channel_indices}
    sum_x: Dict[int, float] = {int(idx): 0.0 for idx in continuous_channel_indices}
    sum_x2: Dict[int, float] = {int(idx): 0.0 for idx in continuous_channel_indices}
    channel_names: List[str] | None = None
    static_names: List[str] | None = None
    for row in rows:
        sample = np.load(row["sample_path"], allow_pickle=True)
        weather = ColdFeatureStore._sanitize(sample["weather"].astype(np.float32))
        firewx = ColdFeatureStore._sanitize(sample["firewx"].astype(np.float32))
        extra = []
        if "firewx_valid" in sample:
            extra.append(sample["firewx_valid"].astype(np.float32))
        static_npz = np.load(row["static_path"], allow_pickle=True)
        static = ColdFeatureStore._sanitize(static_npz["static"].astype(np.float32))
        static_parts = []
        if "static_valid" in static_npz:
            static_parts.append(static_npz["static_valid"].astype(np.float32))
        static_parts.append(static)
        if channel_names is None:
            weather_names = [str(v) for v in sample.get("weather_names", np.array([], dtype=object)).tolist()]
            firewx_names = [str(v) for v in sample.get("firewx_names", np.array([], dtype=object)).tolist()]
            extra_names = ["firewx_valid"] if "firewx_valid" in sample else []
            static_names = [str(v) for v in static_npz.get("static_names", np.array([], dtype=object)).tolist()]
            static_valid_names = ["static_valid"] if "static_valid" in static_npz else []
            channel_names = weather_names + firewx_names + extra_names + static_valid_names + static_names
        x = np.concatenate([weather, firewx, *extra, *static_parts], axis=0).astype(np.float32)
        for idx in continuous_channel_indices:
            arr = x[int(idx)].astype(np.float64, copy=False).ravel()
            count[int(idx)] += int(arr.size)
            sum_x[int(idx)] += float(arr.sum())
            sum_x2[int(idx)] += float(np.square(arr).sum())
    channels = []
    for idx in continuous_channel_indices:
        n = max(int(count[int(idx)]), 1)
        mean = float(sum_x[int(idx)] / n)
        variance = max(float(sum_x2[int(idx)] / n - mean * mean), 0.0)
        std = float(max(math.sqrt(variance), eps))
        channels.append(
            {
                "index": int(idx),
                "name": channel_names[int(idx)] if channel_names and int(idx) < len(channel_names) else str(idx),
                "mean": mean,
                "std": std,
                "count": int(count[int(idx)]),
            }
        )
    return {
        "enabled": True,
        "method": "per-channel z-score",
        "stats_source": "train split full maps",
        "continuous_channel_indices": [int(idx) for idx in continuous_channel_indices],
        "eps": float(eps),
        "channels": channels,
        "channel_names": channel_names or [],
        "static_names": static_names or [],
    }


def load_or_compute_input_normalization_stats(
    config: Dict[str, object],
    train_rows: List[Dict[str, str]],
    metric_dir: Path,
) -> Dict[str, object] | None:
    norm_cfg = normalization_config(config)
    if not bool(norm_cfg.get("enabled", False)):
        return None
    stats_path_value = str(norm_cfg.get("stats_path", "")).strip()
    if stats_path_value:
        stats = load_json(Path(stats_path_value))
    else:
        stats = compute_input_normalization_stats(
            rows=train_rows,
            continuous_channel_indices=[int(v) for v in norm_cfg.get("continuous_channel_indices", list(range(10)))],
            eps=float(norm_cfg.get("eps", 1e-6)),
        )
    write_json(metric_dir / "input_normalization_stats.json", stats)
    return stats


class ColdFeatureStore:
    def __init__(self, rows: List[Dict[str, str]], normalization_stats: Dict[str, object] | None = None):
        self.cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.normalization_stats = normalization_stats
        static_path = Path(rows[0]["static_path"])
        static_npz = np.load(static_path, allow_pickle=True)
        static = self._sanitize(static_npz["static"].astype(np.float32))
        static_valid = static_npz["static_valid"].astype(np.float32) if "static_valid" in static_npz else None
        static_parts = []
        if static_valid is not None:
            static_parts.append(static_valid.astype(np.float32))
        static_parts.append(static)
        static_x = np.concatenate(static_parts, axis=0).astype(np.float32)
        for row in rows:
            sample = np.load(row["sample_path"], allow_pickle=True)
            weather = self._sanitize(sample["weather"].astype(np.float32))
            firewx = self._sanitize(sample["firewx"].astype(np.float32))
            extra = []
            if "firewx_valid" in sample:
                extra.append(sample["firewx_valid"].astype(np.float32))
            x = np.concatenate([weather, firewx, *extra, static_x], axis=0).astype(np.float32)
            x = self._apply_normalization(x)
            y = np.nan_to_num(sample["y_occ"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            self.cache[str(row["sample_id"])] = {"x": x, "y": y}

    @staticmethod
    def _sanitize(x: np.ndarray) -> np.ndarray:
        x = np.where(x <= -9000.0, np.nan, x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.astype(np.float32, copy=False)

    def _apply_normalization(self, x: np.ndarray) -> np.ndarray:
        if not self.normalization_stats:
            return x
        for item in self.normalization_stats.get("channels", []):
            idx = int(item["index"])
            if idx < 0 or idx >= x.shape[0]:
                continue
            mean = float(item["mean"])
            std = max(float(item["std"]), float(self.normalization_stats.get("eps", 1e-6)))
            x[idx] = (x[idx] - mean) / std
        return x.astype(np.float32, copy=False)

    def get(self, sample_id: str) -> Dict[str, np.ndarray]:
        return self.cache[sample_id]


def pooled_positive_rate(rows: List[Dict[str, str]], store: ColdFeatureStore, radius: int) -> float:
    if radius <= 0:
        return float(full_map_stats(rows, store)["positive_rate"])
    pos = 0.0
    total = 0.0
    for row in rows:
        sample = store.get(str(row["sample_id"]))
        y = torch.from_numpy(sample["y"].astype(np.float32)).unsqueeze(0)
        pooled = F.max_pool2d(y, kernel_size=radius * 2 + 1, stride=1, padding=radius)
        pos += float((pooled > 0.5).sum().item())
        total += float(pooled.numel())
    return float(pos / total) if total > 0 else 0.0


def transform_spatial_aux_target(target: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return target
    return F.max_pool2d(target, kernel_size=radius * 2 + 1, stride=1, padding=radius)


def transform_train_target(target: torch.Tensor, config: Dict[str, object]) -> torch.Tensor:
    mode = str(config.get("train_target_mode", "hard"))
    radius = int(config.get("train_target_radius", 0))
    if mode == "hard" or radius <= 0:
        return target
    kernel = radius * 2 + 1
    if mode == "dilate_max":
        return F.max_pool2d(target, kernel_size=kernel, stride=1, padding=radius)
    if mode == "soft_pool":
        return F.avg_pool2d(target, kernel_size=kernel, stride=1, padding=radius)
    raise ValueError(f"Unsupported train_target_mode: {mode}")


def maybe_unpack_logits(output):
    if isinstance(output, tuple):
        return output
    return output, None


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        alpha_t = torch.where(
            targets > 0.5,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1.0 - self.alpha),
        )
        loss = alpha_t * ((1.0 - pt) ** self.gamma) * bce
        return loss.mean()


class OHEMBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: float, neg_pos_ratio: float = 8.0, min_negatives: int = 64):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.neg_pos_ratio = float(neg_pos_ratio)
        self.min_negatives = int(min_negatives)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype),
        )
        batch_losses: List[torch.Tensor] = []
        batch_size = int(logits.shape[0])
        for i in range(batch_size):
            loss_i = losses[i].reshape(-1)
            tgt_i = targets[i].reshape(-1) > 0.5
            pos_loss = loss_i[tgt_i]
            neg_loss = loss_i[~tgt_i]
            num_pos = int(pos_loss.numel())
            keep_neg = max(self.min_negatives, int(self.neg_pos_ratio * max(num_pos, 1)))
            keep_neg = min(keep_neg, int(neg_loss.numel()))
            if keep_neg > 0:
                neg_loss = torch.topk(neg_loss, k=keep_neg, largest=True).values
            else:
                neg_loss = neg_loss[:0]
            denom = max(num_pos + int(neg_loss.numel()), 1)
            batch_losses.append((pos_loss.sum() + neg_loss.sum()) / denom)
        return torch.stack(batch_losses).mean()


class SoftFbetaBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: float, beta: float = 2.0, bce_weight: float = 0.4, smooth: float = 1.0):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.beta = float(beta)
        self.bce_weight = float(bce_weight)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.clamp(0.0, 1.0)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="mean",
            pos_weight=torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype),
        )
        probs = torch.sigmoid(logits.float())
        targets_f = targets.float()
        dims = tuple(range(1, probs.ndim))
        tp = (probs * targets_f).sum(dim=dims)
        fp = (probs * (1.0 - targets_f)).sum(dim=dims)
        fn = ((1.0 - probs) * targets_f).sum(dim=dims)
        beta_sq = self.beta * self.beta
        fbeta = ((1.0 + beta_sq) * tp + self.smooth) / ((1.0 + beta_sq) * tp + beta_sq * fn + fp + self.smooth)
        soft_loss = 1.0 - fbeta.mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * soft_loss


class TverskyBCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float,
        alpha: float = 0.3,
        beta: float = 0.7,
        bce_weight: float = 0.4,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.bce_weight = float(bce_weight)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.clamp(0.0, 1.0)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="mean",
            pos_weight=torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype),
        )
        probs = torch.sigmoid(logits.float())
        targets_f = targets.float()
        dims = tuple(range(1, probs.ndim))
        tp = (probs * targets_f).sum(dim=dims)
        fp = (probs * (1.0 - targets_f)).sum(dim=dims)
        fn = ((1.0 - probs) * targets_f).sum(dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * (1.0 - tversky.mean())


class FSSBCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float,
        radii: Iterable[int] = (1, 2),
        bce_weight: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.radii = [int(v) for v in radii if int(v) > 0]
        if not self.radii:
            self.radii = [1]
        self.bce_weight = float(bce_weight)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.clamp(0.0, 1.0)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="mean",
            pos_weight=torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype),
        )
        probs = torch.sigmoid(logits.float())
        targets_f = targets.float()
        fss_losses: List[torch.Tensor] = []
        for radius in self.radii:
            kernel = radius * 2 + 1
            pred_frac = F.avg_pool2d(probs, kernel_size=kernel, stride=1, padding=radius)
            target_frac = F.avg_pool2d(targets_f, kernel_size=kernel, stride=1, padding=radius)
            mse = torch.mean((pred_frac - target_frac) ** 2)
            reference = torch.mean(pred_frac**2 + target_frac**2)
            fss = 1.0 - mse / (reference + self.eps)
            fss_losses.append(1.0 - fss)
        fss_loss = torch.stack(fss_losses).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * fss_loss


def build_loss(config: Dict[str, object], pos_weight: float, device: torch.device) -> nn.Module:
    loss_type = str(config.get("loss_type", "bce"))
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    if loss_type == "focal_bce":
        return FocalBCEWithLogitsLoss(
            gamma=float(config.get("focal_gamma", 2.0)),
            alpha=float(config.get("focal_alpha", 0.75)),
        )
    if loss_type == "ohem_bce":
        return OHEMBCEWithLogitsLoss(
            pos_weight=pos_weight,
            neg_pos_ratio=float(config.get("ohem_neg_pos_ratio", 8.0)),
            min_negatives=int(config.get("ohem_min_negatives", 64)),
        )
    if loss_type == "soft_fbeta_bce":
        return SoftFbetaBCEWithLogitsLoss(
            pos_weight=pos_weight,
            beta=float(config.get("soft_fbeta_beta", 2.0)),
            bce_weight=float(config.get("soft_metric_bce_weight", 0.4)),
            smooth=float(config.get("soft_metric_smooth", 1.0)),
        )
    if loss_type == "tversky_bce":
        return TverskyBCEWithLogitsLoss(
            pos_weight=pos_weight,
            alpha=float(config.get("tversky_alpha", 0.3)),
            beta=float(config.get("tversky_beta", 0.7)),
            bce_weight=float(config.get("soft_metric_bce_weight", 0.4)),
            smooth=float(config.get("soft_metric_smooth", 1.0)),
        )
    if loss_type == "fss_bce":
        return FSSBCEWithLogitsLoss(
            pos_weight=pos_weight,
            radii=[int(v) for v in config.get("fss_loss_radii", [1, 2])],
            bce_weight=float(config.get("soft_metric_bce_weight", 0.5)),
        )
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def full_map_stats(rows: List[Dict[str, str]], store: ColdFeatureStore) -> Dict[str, float]:
    pos = 0.0
    total = 0.0
    for row in rows:
        sample = store.get(str(row["sample_id"]))
        y = sample["y"]
        pos += float((y > 0.5).sum())
        total += float(y.size)
    neg = max(total - pos, 0.0)
    raw_pos_weight = float(max(neg / max(pos, 1.0), 1.0)) if total > 0 else 1.0
    return {
        "source": "full_map",
        "positive_cells": pos,
        "total_cells": total,
        "positive_rate": float(pos / total) if total > 0 else 0.0,
        "raw_pos_weight": raw_pos_weight,
    }


def tile_stats(tile_rows: List[Dict[str, object]], store: ColdFeatureStore) -> Dict[str, float]:
    pos = 0.0
    total = 0.0
    for row in tile_rows:
        sample = store.get(str(row["sample_id"]))
        top = int(row["top"])
        left = int(row["left"])
        tile_size = int(row["tile_size"])
        target = sample["y"][0, top : top + tile_size, left : left + tile_size]
        pos += float((target > 0.5).sum())
        total += float(target.size)
    neg = max(total - pos, 0.0)
    raw_pos_weight = float(max(neg / max(pos, 1.0), 1.0)) if total > 0 else 1.0
    return {
        "source": "tiles",
        "positive_cells": pos,
        "total_cells": total,
        "positive_rate": float(pos / total) if total > 0 else 0.0,
        "raw_pos_weight": raw_pos_weight,
    }


def select_stats(source: str, train_rows: List[Dict[str, str]], tile_rows: List[Dict[str, object]], store: ColdFeatureStore) -> Dict[str, float]:
    if source == "full_map":
        return full_map_stats(train_rows, store)
    if source == "tiles":
        return tile_stats(tile_rows, store)
    raise ValueError(f"Unsupported stats source: {source}")


def build_train_tiles(
    train_rows: List[Dict[str, str]],
    store: ColdFeatureStore,
    tile_size: int,
    max_positive_tiles_per_sample: int,
    min_negative_tiles_per_sample: int,
    neg_pos_ratio: float,
    positive_tile_placement: str,
    rng: random.Random,
) -> List[Dict[str, object]]:
    tile_rows: List[Dict[str, object]] = []
    for row in train_rows:
        sample_id = str(row["sample_id"])
        mask = store.get(sample_id)["y"][0]
        pos_origins = positive_tile_origins(
            mask=mask,
            tile_h=tile_size,
            tile_w=tile_size,
            max_tiles=max_positive_tiles_per_sample,
            rng=rng,
            placement=positive_tile_placement,
        )
        neg_count = max(min_negative_tiles_per_sample, int(math.ceil(len(pos_origins) * neg_pos_ratio)))
        neg_origins = negative_tile_origins(
            mask=mask,
            tile_h=tile_size,
            tile_w=tile_size,
            desired=neg_count,
            rng=rng,
            forbidden=pos_origins,
        )
        for top, left in pos_origins:
            tile_rows.append(
                {
                    "sample_id": sample_id,
                    "tile_type": "positive",
                    "top": int(top),
                    "left": int(left),
                    "tile_size": int(tile_size),
                }
            )
        for top, left in neg_origins:
            tile_rows.append(
                {
                    "sample_id": sample_id,
                    "tile_type": "negative",
                    "top": int(top),
                    "left": int(left),
                    "tile_size": int(tile_size),
                }
            )
    return tile_rows


class TrainTileDataset(Dataset):
    def __init__(self, tile_rows: List[Dict[str, object]], store: ColdFeatureStore, augment_flip: bool, seed: int):
        self.rows = tile_rows
        self.store = store
        self.augment_flip = bool(augment_flip)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        sample = self.store.get(str(row["sample_id"]))
        top = int(row["top"])
        left = int(row["left"])
        tile_size = int(row["tile_size"])
        x = sample["x"][:, top : top + tile_size, left : left + tile_size]
        y = sample["y"][:, top : top + tile_size, left : left + tile_size]
        if self.augment_flip:
            if self.rng.random() < 0.5:
                x = x[:, :, ::-1].copy()
                y = y[:, :, ::-1].copy()
            if self.rng.random() < 0.5:
                x = x[:, ::-1, :].copy()
                y = y[:, ::-1, :].copy()
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y)}


class FullMapDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]], store: ColdFeatureStore):
        self.rows = rows
        self.store = store

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        sample = self.store.get(str(row["sample_id"]))
        return {"x": torch.from_numpy(sample["x"]), "y": torch.from_numpy(sample["y"]), "sample_id": str(row["sample_id"])}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: List[float],
    topk_area_fractions: List[float],
    fss_radii: List[int],
    n_bins: int,
    reference_positive_rate: float,
    criterion: nn.Module,
    amp: bool,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_prob_maps: List[np.ndarray] = []
    all_target_maps: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = torch.nan_to_num(batch["x"], nan=0.0, posinf=0.0, neginf=0.0).to(device, non_blocking=True)
            y = torch.nan_to_num(batch["y"], nan=0.0, posinf=0.0, neginf=0.0).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                logits, _ = maybe_unpack_logits(model(x))
                loss = criterion(logits, y)
            prob = torch.sigmoid(logits.float()).detach().cpu().numpy()[:, 0, :, :]
            target = y.float().detach().cpu().numpy()[:, 0, :, :]
            all_prob_maps.append(prob)
            all_target_maps.append(target)
            total_loss += float(loss.item()) * x.size(0)
            total_items += int(x.size(0))
    prob_maps = np.concatenate(all_prob_maps, axis=0)
    target_maps = np.concatenate(all_target_maps, axis=0)
    metrics = metric_bundle(
        prob_maps=prob_maps,
        target_maps=target_maps,
        thresholds=thresholds,
        topk_fractions=topk_area_fractions,
        fss_radii=fss_radii,
        n_bins=n_bins,
        reference_positive_rate=reference_positive_rate,
    )
    metrics.update({"loss": total_loss / max(total_items, 1), "num_samples": int(prob_maps.shape[0])})
    return metrics


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, float], config: Dict[str, object]) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    set_seed(int(config.get("seed", 7)))
    rng = random.Random(int(config.get("seed", 7)))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    index_root = Path(config["index_root"])
    run_root = Path(config["run_root"])
    ckpt_dir = run_root / "checkpoints" / args.run_name
    metric_dir = run_root / "metrics" / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_rows(index_root / "splits" / "train.csv")
    val_rows = read_rows(index_root / "splits" / "val.csv")
    test_rows = read_rows(index_root / "splits" / "test.csv")
    normalization_stats = load_or_compute_input_normalization_stats(config, train_rows, metric_dir)
    store = ColdFeatureStore(train_rows + val_rows + test_rows, normalization_stats=normalization_stats)

    tile_rows = build_train_tiles(
        train_rows=train_rows,
        store=store,
        tile_size=int(config.get("tile_size", 16)),
        max_positive_tiles_per_sample=int(config.get("max_positive_tiles_per_sample", 64)),
        min_negative_tiles_per_sample=int(config.get("min_negative_tiles_per_sample", 4)),
        neg_pos_ratio=float(config.get("negative_to_positive_ratio", 2.0)),
        positive_tile_placement=str(config.get("positive_tile_placement", "center")),
        rng=rng,
    )
    train_ds = TrainTileDataset(tile_rows=tile_rows, store=store, augment_flip=bool(config.get("augment_flip", True)), seed=int(config.get("seed", 7)))
    val_ds = FullMapDataset(val_rows, store)
    test_ds = FullMapDataset(test_rows, store)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_ds, batch_size=int(config.get("batch_size", 64)), shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=int(config.get("eval_batch_size", 8)), shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=int(config.get("eval_batch_size", 8)), shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    positive_rate_source = str(config.get("positive_rate_source", "full_map"))
    pos_weight_source = str(config.get("pos_weight_source", "full_map"))
    metric_reference_positive_rate = full_map_stats(train_rows, store)["positive_rate"]
    positive_rate_stats = select_stats(positive_rate_source, train_rows, tile_rows, store)
    pos_weight_stats = select_stats(pos_weight_source, train_rows, tile_rows, store)
    train_positive_rate = float(positive_rate_stats["positive_rate"])
    raw_pos_weight = float(pos_weight_stats["raw_pos_weight"])
    pos_weight_cap = float(config.get("pos_weight_cap", 300.0))
    pos_weight = float(min(max(raw_pos_weight, 1.0), pos_weight_cap))
    use_aux_spatial_head = bool(config.get("use_aux_spatial_head", False))
    aux_spatial_radius = int(config.get("aux_spatial_radius", 1))
    aux_positive_rate = pooled_positive_rate(train_rows, store, aux_spatial_radius) if use_aux_spatial_head else None

    in_ch = int(store.get(str(train_rows[0]["sample_id"]))["x"].shape[0])
    model = UNetSmallFlex(
        in_ch=in_ch,
        base=int(config.get("base_channels", 32)),
        dropout=float(config.get("dropout", 0.1)),
        norm_type=str(config.get("norm_type", "group")),
        norm_groups=int(config.get("norm_groups", 8)),
        prior_prob=train_positive_rate if bool(config.get("init_head_bias_from_positive_rate", True)) else None,
        use_aux_spatial_head=use_aux_spatial_head,
        aux_prior_prob=aux_positive_rate if bool(config.get("init_head_bias_from_positive_rate", True)) else None,
    ).to(device)
    init_checkpoint = config.get("init_checkpoint")
    if init_checkpoint:
        checkpoint = torch.load(str(init_checkpoint), map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        if not isinstance(state, dict):
            raise RuntimeError(f"Unexpected init_checkpoint format: {init_checkpoint}")
        model.load_state_dict(state, strict=False)
    criterion = build_loss(config=config, pos_weight=pos_weight, device=device)
    aux_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 3e-4)), weight_decay=float(config.get("weight_decay", 1e-4)))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.get("amp", True)) and device.type == "cuda")

    thresholds = [float(v) for v in config.get("metric_thresholds", [0.1, 0.2, 0.3, 0.5])]
    topk_area_fractions = [float(v) for v in config.get("topk_area_fractions", [0.01, 0.05, 0.1])]
    fss_radii = [int(v) for v in config.get("fss_radii", [1, 2, 4, 8])]
    n_bins = int(config.get("reliability_bins", 10))
    default_threshold_key = f"{float(config.get('threshold', 0.3)):.4f}"

    summary_seed = {
        "index_root": str(index_root),
        "tile_size": int(config.get("tile_size", 16)),
        "num_train_tiles": len(tile_rows),
        "positive_train_tiles": int(sum(1 for row in tile_rows if row["tile_type"] == "positive")),
        "negative_train_tiles": int(sum(1 for row in tile_rows if row["tile_type"] == "negative")),
        "input_channels": in_ch,
        "positive_tile_placement": str(config.get("positive_tile_placement", "center")),
        "train_positive_rate": train_positive_rate,
        "metric_reference_positive_rate": metric_reference_positive_rate,
        "pos_weight_source": pos_weight_source,
        "positive_rate_source": positive_rate_source,
        "pos_weight": pos_weight,
        "loss_type": str(config.get("loss_type", "bce")),
        "train_target_mode": str(config.get("train_target_mode", "hard")),
        "train_target_radius": int(config.get("train_target_radius", 0)),
        "input_normalization": normalization_stats if normalization_stats else {"enabled": False},
    }
    (metric_dir / "tile_summary.json").write_text(json.dumps(summary_seed, indent=2), encoding="utf-8")

    best_pr = -1.0
    best_state = None
    history: List[Dict[str, float]] = []
    aux_spatial_loss_weight = float(config.get("aux_spatial_loss_weight", 0.0))
    for epoch in range(1, int(config.get("epochs", 30)) + 1):
        model.train()
        epoch_start = time.time()
        train_loss = 0.0
        train_main_loss = 0.0
        train_aux_spatial_loss = 0.0
        train_items = 0
        for batch in train_loader:
            x = torch.nan_to_num(batch["x"], nan=0.0, posinf=0.0, neginf=0.0).to(device, non_blocking=True)
            y = torch.nan_to_num(batch["y"], nan=0.0, posinf=0.0, neginf=0.0).to(device, non_blocking=True)
            train_target = transform_train_target(y.float(), config)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bool(config.get("amp", True)) and device.type == "cuda"):
                logits, aux_logits = maybe_unpack_logits(model(x, return_aux=use_aux_spatial_head))
                main_loss = criterion(logits.float(), train_target.float())
            aux_loss = torch.zeros((), device=device, dtype=torch.float32)
            if aux_logits is not None and aux_spatial_loss_weight > 0.0:
                aux_target = transform_spatial_aux_target(y.float(), aux_spatial_radius)
                aux_loss = aux_criterion(aux_logits.float(), aux_target)
            loss = main_loss + aux_spatial_loss_weight * aux_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite tiled cold-start loss detected.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * x.size(0)
            train_main_loss += float(main_loss.item()) * x.size(0)
            train_aux_spatial_loss += float(aux_loss.item()) * x.size(0)
            train_items += int(x.size(0))

        metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            thresholds=thresholds,
            topk_area_fractions=topk_area_fractions,
            fss_radii=fss_radii,
            n_bins=n_bins,
            reference_positive_rate=metric_reference_positive_rate,
            criterion=criterion,
            amp=bool(config.get("amp", True)),
        )
        threshold_metrics = metrics["threshold_metrics"][default_threshold_key]
        record = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_items, 1),
            "train_main_loss": train_main_loss / max(train_items, 1),
            "train_aux_spatial_loss": train_aux_spatial_loss / max(train_items, 1),
            "val_loss": metrics["loss"],
            "val_pr_auc": metrics["pr_auc"],
            "val_auroc": metrics["auroc"],
            "val_brier": metrics["brier"],
            "val_brier_skill_score": metrics["brier_skill_score"],
            "val_ece": metrics["ece"],
            "val_positive_rate": metrics["positive_rate"],
            "val_precision": threshold_metrics["precision"],
            "val_positive_recall": threshold_metrics["recall"],
            "val_far": threshold_metrics["far"],
            "val_csi": threshold_metrics["csi"],
            "val_f1": threshold_metrics["f1"],
            "val_f2": threshold_metrics["f2"],
            "val_frequency_bias": threshold_metrics["frequency_bias"],
            "val_threshold_metrics": metrics["threshold_metrics"],
            "val_topk_area_metrics": metrics["topk_area_metrics"],
            "val_fss": metrics["fss"],
            "minutes": (time.time() - epoch_start) / 60.0,
        }
        history.append(record)
        (metric_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, epoch, record, config)
        if record["val_pr_auc"] > best_pr:
            best_pr = record["val_pr_auc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "epoch": epoch, "metrics": record, "config": config}, ckpt_dir / "best_firms_prauc.pt")
        print(json.dumps(record), flush=True)

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    val_best = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        thresholds=thresholds,
        topk_area_fractions=topk_area_fractions,
        fss_radii=fss_radii,
        n_bins=n_bins,
        reference_positive_rate=metric_reference_positive_rate,
        criterion=criterion,
        amp=bool(config.get("amp", True)),
    )
    test_best = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        thresholds=thresholds,
        topk_area_fractions=topk_area_fractions,
        fss_radii=fss_radii,
        n_bins=n_bins,
        reference_positive_rate=metric_reference_positive_rate,
        criterion=criterion,
        amp=bool(config.get("amp", True)),
    )
    summary = {
        "run_name": args.run_name,
        "device": str(device),
        "index_root": str(index_root),
        "input_channels": in_ch,
        "tile_size": int(config.get("tile_size", 16)),
        "num_train_tiles": len(tile_rows),
        "positive_train_tiles": int(sum(1 for row in tile_rows if row["tile_type"] == "positive")),
        "negative_train_tiles": int(sum(1 for row in tile_rows if row["tile_type"] == "negative")),
        "positive_tile_placement": str(config.get("positive_tile_placement", "center")),
        "train_positive_rate": train_positive_rate,
        "metric_reference_positive_rate": metric_reference_positive_rate,
        "pos_weight": pos_weight,
        "use_aux_spatial_head": use_aux_spatial_head,
        "aux_spatial_radius": aux_spatial_radius,
        "aux_spatial_loss_weight": aux_spatial_loss_weight,
        "aux_positive_rate": aux_positive_rate,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else "",
        "input_normalization": normalization_stats if normalization_stats else {"enabled": False},
        "best_val_pr_auc": float(val_best["pr_auc"]),
        "best_val_auroc": float(val_best["auroc"]),
        "best_test_pr_auc": float(test_best["pr_auc"]),
        "best_test_auroc": float(test_best["auroc"]),
        "best_test_brier": float(test_best["brier"]),
        "best_test_ece": float(test_best["ece"]),
        "best_test_topk_area_metrics": test_best["topk_area_metrics"],
        "best_test_threshold_metrics": test_best["threshold_metrics"],
        "best_test_fss": test_best["fss"],
        "epochs": int(config.get("epochs", 30)),
    }
    (metric_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
