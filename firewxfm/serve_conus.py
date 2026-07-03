#!/usr/bin/env python3
"""Run final FireWx-FM CONUS inference from a 16-channel NumPy stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from firewxfm.modeling_unet import UNetSmallFlex
from firewxfm.tiled_inference import predict_probability_tiled


DEFAULT_SHIFTS = [(dy, dx) for dy in (0, 4, 8, 12) for dx in (0, 4, 8, 12)]


def load_checkpoint(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint format: {path}")


def parse_shifts(text: str | None) -> list[tuple[int, int]]:
    if not text:
        return DEFAULT_SHIFTS
    shifts: list[tuple[int, int]] = []
    for item in text.split(";"):
        if not item.strip():
            continue
        y, x = item.split(",")
        shifts.append((int(y), int(x)))
    return shifts


def load_stats(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _as_channel_array(value, channels: int) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        out = np.zeros(channels, dtype=np.float32)
        seen = False
        for key, val in value.items():
            try:
                idx = int(key)
            except ValueError:
                continue
            if 0 <= idx < channels:
                out[idx] = float(val)
                seen = True
        return out if seen else None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == channels:
        return arr.reshape(channels)
    return None


def normalize_stack(x: np.ndarray, stats: dict | None) -> np.ndarray:
    if not stats or not stats.get("enabled", True):
        return x.astype(np.float32, copy=True)
    channels = x.shape[0]
    mean = _as_channel_array(stats.get("mean") or stats.get("means"), channels)
    std = _as_channel_array(stats.get("std") or stats.get("stds"), channels)
    normalize_channels_raw = stats.get("normalize_channels") or stats.get("continuous_channel_indices")
    out = x.astype(np.float32, copy=True)
    if mean is None or std is None:
        # Some training summaries store channel records instead of dense arrays.
        records = stats.get("channel_stats") or stats.get("stats")
        if records is None and isinstance(stats.get("channels"), list):
            records = stats.get("channels")
        if isinstance(records, list):
            mean = np.zeros(channels, dtype=np.float32)
            std = np.ones(channels, dtype=np.float32)
            inferred_channels: list[int] = []
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                idx = int(rec.get("index", rec.get("channel", -1)))
                if 0 <= idx < channels:
                    mean[idx] = float(rec.get("mean", 0.0))
                    std[idx] = float(rec.get("std", 1.0))
                    inferred_channels.append(idx)
            if normalize_channels_raw is None:
                normalize_channels_raw = inferred_channels
        else:
            raise ValueError("Normalization stats must contain mean/std arrays or channel_stats records.")
    if normalize_channels_raw is None:
        normalize_channels = list(range(channels))
    else:
        normalize_channels = [int(c) for c in normalize_channels_raw]
    std = np.where(np.abs(std) < 1e-6, 1.0, std)
    for c in normalize_channels:
        if 0 <= c < channels:
            out[c] = (out[c] - mean[c]) / std[c]
    return out


def shifted_predict(
    model: torch.nn.Module,
    x: np.ndarray,
    shifts: list[tuple[int, int]],
    window: int,
    stride: int,
    halo: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    preds = []
    for dy, dx in shifts:
        shifted = np.roll(x, shift=(dy, dx), axis=(1, 2))
        pred = predict_probability_tiled(
            model,
            torch.from_numpy(shifted),
            tile_size=window,
            stride=stride,
            halo=halo,
            device=device,
            batch_size=batch_size,
        ).numpy()
        preds.append(np.roll(pred, shift=(-dy, -dx), axis=(0, 1)))
    return np.mean(preds, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npy", type=Path, required=True, help="Input stack with shape [16,H,W].")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path)
    parser.add_argument("--output-npy", type=Path, required=True)
    parser.add_argument("--zero-channels", default="14,15", help="Comma-separated channels to zero at serving time.")
    parser.add_argument("--shifts", default=None, help="Semicolon list such as '0,0;0,4;4,0'.")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--halo", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    x = np.load(args.input_npy).astype(np.float32)
    if x.ndim != 3 or x.shape[0] != 16:
        raise ValueError(f"Expected input shape [16,H,W], got {x.shape}")
    x = normalize_stack(x, load_stats(args.normalization_stats))
    for channel in [int(c) for c in args.zero_channels.split(",") if c.strip()]:
        x[channel] = 0.0

    device = torch.device(args.device)
    model = UNetSmallFlex(
        in_ch=16,
        base=args.base_channels,
        dropout=args.dropout,
        norm_type="group",
        norm_groups=8,
        use_aux_spatial_head=True,
    ).to(device)
    state = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(state, strict=False)
    pred = shifted_predict(
        model=model,
        x=x,
        shifts=parse_shifts(args.shifts),
        window=args.window,
        stride=args.stride,
        halo=args.halo,
        batch_size=args.batch_size,
        device=device,
    )
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, pred)
    print(f"wrote {args.output_npy} shape={pred.shape} mean={float(pred.mean()):.6f} max={float(pred.max()):.6f}")


if __name__ == "__main__":
    main()
