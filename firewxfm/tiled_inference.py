"""Overlap-tiled inference helpers for FireWx-FM probability maps."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn.functional as F


def _starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    out = list(range(0, max(length - tile_size + 1, 1), stride))
    last = length - tile_size
    if out[-1] != last:
        out.append(last)
    return out


def _crop_slices(top: int, left: int, tile_size: int, height: int, width: int, halo: int) -> Tuple[slice, slice, slice, slice]:
    y0 = 0 if top == 0 else halo
    x0 = 0 if left == 0 else halo
    y1 = tile_size if top + tile_size >= height else tile_size - halo
    x1 = tile_size if left + tile_size >= width else tile_size - halo
    return slice(y0, y1), slice(x0, x1), slice(top + y0, top + y1), slice(left + x0, left + x1)


def predict_probability_tiled(
    model: torch.nn.Module,
    x: torch.Tensor,
    tile_size: int = 32,
    stride: int = 16,
    halo: int = 8,
    device: torch.device | str | None = None,
    batch_size: int = 16,
) -> torch.Tensor:
    """Predict a full probability map from an input tensor using overlap tiles.

    Parameters
    ----------
    model:
        FireWx-FM model returning logits or ``(logits, aux_logits)``.
    x:
        Input tensor in ``[C, H, W]`` or ``[1, C, H, W]`` order.
    tile_size:
        Spatial tile size used for model calls.
    stride:
        Distance between tile origins. Use a value smaller than ``tile_size``
        for overlap.
    halo:
        Number of pixels cropped away from interior tile borders before
        stitching. Border tiles keep the image edge.
    device:
        Device for inference. Defaults to the model parameter device.
    batch_size:
        Number of tiles evaluated per model call.

    Returns
    -------
    torch.Tensor
        Probability map in ``[H, W]`` order on CPU.
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("x must have shape [C, H, W] or [1, C, H, W].")
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive.")
    if halo < 0 or halo * 2 >= tile_size:
        raise ValueError("halo must be non-negative and smaller than tile_size / 2.")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    model.eval()

    _, channels, height, width = x.shape
    pad_h = max(tile_size - height, 0)
    pad_w = max(tile_size - width, 0)
    if pad_h or pad_w:
        x_work = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    else:
        x_work = x
    _, _, work_h, work_w = x_work.shape

    output = torch.zeros((work_h, work_w), dtype=torch.float32)
    weight = torch.zeros((work_h, work_w), dtype=torch.float32)
    coords = [(top, left) for top in _starts(work_h, tile_size, stride) for left in _starts(work_w, tile_size, stride)]

    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            batch_coords = coords[start : start + batch_size]
            tiles = torch.cat(
                [x_work[:, :, top : top + tile_size, left : left + tile_size] for top, left in batch_coords],
                dim=0,
            ).to(device)
            pred = model(tiles)
            logits = pred[0] if isinstance(pred, tuple) else pred
            probs = torch.sigmoid(logits.float()).detach().cpu()[:, 0]
            for prob, (top, left) in zip(probs, batch_coords):
                sy, sx, dy, dx = _crop_slices(top, left, tile_size, work_h, work_w, halo)
                output[dy, dx] += prob[sy, sx]
                weight[dy, dx] += 1.0
    output = output / weight.clamp_min(1.0)
    return output[:height, :width].contiguous()


__all__ = ["predict_probability_tiled"]
