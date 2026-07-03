from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from scipy import ndimage as _scipy_ndimage
except Exception:  # pragma: no cover - optional on login nodes, present in Slurm env.
    _scipy_ndimage = None


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def threshold_metrics(prob: np.ndarray, target: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = prob >= threshold
    pos = target > 0.5
    tp = int(np.logical_and(pred, pos).sum())
    fp = int(np.logical_and(pred, ~pos).sum())
    fn = int(np.logical_and(~pred, pos).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    far = _safe_div(fp, tp + fp)
    csi = _safe_div(tp, tp + fp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    beta_sq = 4.0
    f2 = _safe_div((1.0 + beta_sq) * precision * recall, beta_sq * precision + recall)
    freq_bias = _safe_div(tp + fp, tp + fn)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "far": far,
        "csi": csi,
        "f1": f1,
        "f2": f2,
        "frequency_bias": freq_bias,
        "predicted_positive_rate": float(pred.mean()),
    }


def log_score(prob: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    prob_clip = np.clip(np.asarray(prob, dtype=np.float32), eps, 1.0 - eps)
    target_arr = np.asarray(target, dtype=np.float32)
    loss = -(target_arr * np.log(prob_clip) + (1.0 - target_arr) * np.log(1.0 - prob_clip))
    return float(np.mean(loss))


def _spatial_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask > 0.5
    pooled = F.max_pool2d(
        mask.float().unsqueeze(1),
        kernel_size=radius * 2 + 1,
        stride=1,
        padding=radius,
    ).squeeze(1)
    return pooled > 0.5


def _spatial_erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask > 0.5
    inv = 1.0 - (mask > 0.5).float()
    pooled = F.max_pool2d(
        inv.unsqueeze(1),
        kernel_size=radius * 2 + 1,
        stride=1,
        padding=radius,
    ).squeeze(1)
    return pooled < 0.5


def _boundary_mask(mask: torch.Tensor, width: int = 1) -> torch.Tensor:
    mask_bool = mask > 0.5
    if width <= 0:
        return mask_bool
    eroded = _spatial_erode(mask_bool.float(), width)
    return mask_bool & ~eroded


def _sample_times_to_hours(sample_times: np.ndarray) -> np.ndarray:
    times = np.asarray(sample_times)
    if np.issubdtype(times.dtype, np.datetime64):
        return times.astype("datetime64[h]").astype(np.int64)
    return times.astype(np.int64)


def _build_tolerance_support(
    pred_bool: torch.Tensor,
    target_bool: torch.Tensor,
    sample_times: np.ndarray,
    temporal_tolerance_steps: int,
    spatial_tolerance_radius: int,
    time_step_hours: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    times_h = _sample_times_to_hours(sample_times)
    target_support = torch.zeros_like(target_bool, dtype=torch.bool)
    pred_support = torch.zeros_like(pred_bool, dtype=torch.bool)
    tolerance_hours = int(temporal_tolerance_steps) * int(time_step_hours)

    for idx, current_time in enumerate(times_h):
        window = np.abs(times_h - current_time) <= tolerance_hours
        target_union = target_bool[window].float().amax(dim=0, keepdim=True)
        pred_union = pred_bool[window].float().amax(dim=0, keepdim=True)
        target_support[idx] = _spatial_dilate(target_union, spatial_tolerance_radius)[0]
        pred_support[idx] = _spatial_dilate(pred_union, spatial_tolerance_radius)[0]
    return target_support, pred_support, tolerance_hours


def tolerant_threshold_metrics(
    prob_maps: np.ndarray,
    target_maps: np.ndarray,
    sample_times: np.ndarray,
    threshold: float,
    temporal_tolerance_steps: int,
    spatial_tolerance_radius: int,
    region_mask: np.ndarray | None = None,
    time_step_hours: int = 24,
) -> Dict[str, float]:
    pred = torch.from_numpy((prob_maps >= threshold).astype(np.float32))
    target = torch.from_numpy((target_maps > 0.5).astype(np.float32))
    pred_bool = pred > 0.5
    target_bool = target > 0.5
    target_support, pred_support, tolerance_hours = _build_tolerance_support(
        pred_bool=pred_bool,
        target_bool=target_bool,
        sample_times=sample_times,
        temporal_tolerance_steps=temporal_tolerance_steps,
        spatial_tolerance_radius=spatial_tolerance_radius,
        time_step_hours=time_step_hours,
    )

    matched_pred = pred_bool & target_support
    matched_target = target_bool & pred_support

    if region_mask is not None:
        region = torch.from_numpy(_region_mask_to_bool(region_mask, prob_maps.shape[1:])).to(dtype=torch.bool)
        pred_bool = pred_bool & region.unsqueeze(0)
        target_bool = target_bool & region.unsqueeze(0)
        matched_pred = matched_pred & region.unsqueeze(0)
        matched_target = matched_target & region.unsqueeze(0)
        total_cells = int(region.sum().item()) * int(pred_bool.shape[0])
    else:
        total_cells = int(pred_bool.numel())

    pred_total = int(pred_bool.sum().item())
    target_total = int(target_bool.sum().item())
    matched_pred_total = int(matched_pred.sum().item())
    matched_target_total = int(matched_target.sum().item())
    precision = _safe_div(matched_pred_total, pred_total)
    recall = _safe_div(matched_target_total, target_total)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "threshold": float(threshold),
        "temporal_tolerance_steps": int(temporal_tolerance_steps),
        "temporal_tolerance_hours": int(tolerance_hours),
        "time_step_hours": int(time_step_hours),
        "spatial_tolerance_radius": int(spatial_tolerance_radius),
        "predicted_positive_cells": pred_total,
        "target_positive_cells": target_total,
        "matched_predicted_cells": matched_pred_total,
        "matched_target_cells": matched_target_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": _safe_div(pred_total, total_cells),
    }


def neighborhood_contingency(
    prob_maps: np.ndarray,
    target_maps: np.ndarray,
    sample_times: np.ndarray,
    threshold: float,
    temporal_tolerance_steps: int,
    spatial_tolerance_radius: int,
    region_mask: np.ndarray | None = None,
    time_step_hours: int = 24,
) -> Dict[str, float]:
    pred_bool = torch.from_numpy((prob_maps >= threshold).astype(np.float32)) > 0.5
    target_bool = torch.from_numpy((target_maps > 0.5).astype(np.float32)) > 0.5
    target_support, pred_support, tolerance_hours = _build_tolerance_support(
        pred_bool=pred_bool,
        target_bool=target_bool,
        sample_times=sample_times,
        temporal_tolerance_steps=temporal_tolerance_steps,
        spatial_tolerance_radius=spatial_tolerance_radius,
        time_step_hours=time_step_hours,
    )

    if region_mask is not None:
        region = torch.from_numpy(_region_mask_to_bool(region_mask, prob_maps.shape[1:])).to(dtype=torch.bool)
        pred_bool = pred_bool & region.unsqueeze(0)
        target_bool = target_bool & region.unsqueeze(0)
        target_support = target_support & region.unsqueeze(0)
        pred_support = pred_support & region.unsqueeze(0)

    hits = 0
    false_alarms = 0
    misses = 0
    true_negatives = 0
    for idx in range(int(pred_bool.shape[0])):
        pred_event = bool(pred_bool[idx].any().item())
        target_event = bool(target_bool[idx].any().item())
        pred_match = bool((pred_bool[idx] & target_support[idx]).any().item()) if pred_event else False
        target_match = bool((target_bool[idx] & pred_support[idx]).any().item()) if target_event else False
        if target_event and target_match:
            hits += 1
        elif target_event:
            misses += 1
        elif not target_event and not pred_event:
            true_negatives += 1
        if pred_event and not pred_match:
            false_alarms += 1

    precision = _safe_div(hits, hits + false_alarms)
    recall = _safe_div(hits, hits + misses)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    far = _safe_div(false_alarms, hits + false_alarms)
    csi = _safe_div(hits, hits + false_alarms + misses)
    return {
        "threshold": float(threshold),
        "temporal_tolerance_steps": int(temporal_tolerance_steps),
        "temporal_tolerance_hours": int(tolerance_hours),
        "time_step_hours": int(time_step_hours),
        "spatial_tolerance_radius": int(spatial_tolerance_radius),
        "hits": int(hits),
        "false_alarms": int(false_alarms),
        "misses": int(misses),
        "true_negatives": int(true_negatives),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": far,
        "csi": csi,
        "predicted_event_rate": _safe_div(hits + false_alarms, pred_bool.shape[0]),
        "target_event_rate": _safe_div(hits + misses, pred_bool.shape[0]),
    }


def neighborhood_contingency_metrics(
    prob_maps: np.ndarray,
    target_maps: np.ndarray,
    sample_times: np.ndarray,
    thresholds: Iterable[float],
    temporal_tolerances_steps: Iterable[int],
    spatial_tolerances_radii: Iterable[int],
    time_step_hours: int = 24,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    temporal_values = [int(v) for v in temporal_tolerances_steps]
    spatial_values = [int(v) for v in spatial_tolerances_radii]
    for temporal_steps in temporal_values:
        for spatial_radius in spatial_values:
            if temporal_steps == 0 and spatial_radius == 0:
                continue
            combo_key = f"t{temporal_steps}_s{spatial_radius}"
            out[combo_key] = {
                f"{float(t):.4f}": neighborhood_contingency(
                    prob_maps=prob_maps,
                    target_maps=target_maps,
                    sample_times=sample_times,
                    threshold=float(t),
                    temporal_tolerance_steps=temporal_steps,
                    spatial_tolerance_radius=spatial_radius,
                    time_step_hours=time_step_hours,
                )
                for t in thresholds
            }
    return out


def reliability_bins(prob: np.ndarray, target: np.ndarray, n_bins: int) -> List[Dict[str, float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: List[Dict[str, float]] = []
    for idx in range(n_bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == n_bins - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": idx,
                    "lo": float(lo),
                    "hi": float(hi),
                    "count": 0,
                    "mean_confidence": 0.0,
                    "empirical_accuracy": 0.0,
                }
            )
            continue
        mean_conf = float(prob[mask].mean())
        acc = float(target[mask].mean())
        rows.append(
            {
                "bin": idx,
                "lo": float(lo),
                "hi": float(hi),
                "count": count,
                "mean_confidence": mean_conf,
                "empirical_accuracy": acc,
            }
        )
    return rows


def expected_calibration_error(prob: np.ndarray, target: np.ndarray, n_bins: int) -> float:
    bins = reliability_bins(prob, target, n_bins)
    total = max(int(prob.size), 1)
    return float(
        sum(abs(row["empirical_accuracy"] - row["mean_confidence"]) * row["count"] for row in bins) / total
    )


def topk_area_metrics(prob: np.ndarray, target: np.ndarray, fractions: Iterable[float]) -> Dict[str, Dict[str, float]]:
    order = np.argsort(prob)[::-1]
    total = prob.size
    positive_rate = float(target.mean())
    pos_total = max(float(target.sum()), 1.0)
    out: Dict[str, Dict[str, float]] = {}
    for frac in fractions:
        frac = float(frac)
        keep = max(int(math.ceil(total * frac)), 1)
        idx = order[:keep]
        top_target = target[idx]
        precision = float(top_target.mean())
        recall = float(top_target.sum() / pos_total)
        lift = float(precision / positive_rate) if positive_rate > 0 else 0.0
        out[f"{frac:.4f}"] = {
            "fraction": frac,
            "cells": keep,
            "precision": precision,
            "recall": recall,
            "lift": lift,
        }
    return out


def _crop_for_factor(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    h = int(arr.shape[-2])
    w = int(arr.shape[-1])
    new_h = (h // factor) * factor
    new_w = (w // factor) * factor
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"Cannot coarsen shape {(h, w)} by factor={factor}")
    return arr[..., :new_h, :new_w]


def _avg_pool_maps(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr.astype(np.float32, copy=False)
    cropped = _crop_for_factor(arr, factor).astype(np.float32, copy=False)
    tensor = torch.from_numpy(cropped).unsqueeze(1)
    pooled = F.avg_pool2d(tensor, kernel_size=factor, stride=factor)
    return pooled.squeeze(1).cpu().numpy().astype(np.float32, copy=False)


def _max_pool_binary_maps(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return (arr > 0.5).astype(np.float32, copy=False)
    cropped = _crop_for_factor(arr, factor).astype(np.float32, copy=False)
    tensor = torch.from_numpy((cropped > 0.5).astype(np.float32)).unsqueeze(1)
    pooled = F.max_pool2d(tensor, kernel_size=factor, stride=factor)
    return (pooled.squeeze(1).cpu().numpy() > 0.5).astype(np.float32)


def coarsened_metrics(
    prob_maps: np.ndarray,
    target_maps: np.ndarray,
    thresholds: Iterable[float],
    factors: Iterable[int],
    reference_positive_rate: float | None = None,
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    target_binary = (target_maps > 0.5).astype(np.float32, copy=False)
    for factor in factors:
        factor = int(factor)
        if factor <= 1:
            continue
        prob_coarse = _avg_pool_maps(prob_maps, factor)
        target_fraction = _avg_pool_maps(target_binary, factor)
        target_any = _max_pool_binary_maps(target_binary, factor)

        prob_flat = prob_coarse.reshape(-1)
        any_flat = target_any.reshape(-1)
        fraction_flat = target_fraction.reshape(-1)

        any_metrics: Dict[str, object] = {
            "positive_rate": float(any_flat.mean()) if any_flat.size > 0 else 0.0,
            "positive_cells": int(any_flat.sum()) if any_flat.size > 0 else 0,
            "total_cells": int(any_flat.size),
            "pr_auc": float(average_precision_score(any_flat, prob_flat)) if float(any_flat.sum()) > 0 else 0.0,
            "brier": float(np.mean((prob_flat - any_flat) ** 2)) if any_flat.size > 0 else 0.0,
            "log_score": log_score(prob_flat, any_flat) if any_flat.size > 0 else 0.0,
            "threshold_metrics": {
                f"{float(t):.4f}": threshold_metrics(prob_flat, any_flat, float(t)) for t in thresholds
            }
            if any_flat.size > 0
            else {},
        }
        if any_flat.size > 0 and float(np.unique(any_flat).size) > 1:
            any_metrics["auroc"] = float(roc_auc_score(any_flat, prob_flat))
        else:
            any_metrics["auroc"] = 0.0
        ref_rate = reference_positive_rate if reference_positive_rate is not None else float(any_flat.mean())
        brier_ref = float(np.mean((ref_rate - any_flat) ** 2)) if any_flat.size > 0 else 0.0
        any_metrics["reference_positive_rate"] = float(ref_rate)
        any_metrics["brier_skill_score"] = float(1.0 - any_metrics["brier"] / brier_ref) if brier_ref > 0 else 0.0

        out[f"x{factor}"] = {
            "grid_shape": {"lat": int(prob_coarse.shape[-2]), "lon": int(prob_coarse.shape[-1])},
            "any": any_metrics,
            "fraction": {
                "mean_target_fraction": float(fraction_flat.mean()) if fraction_flat.size > 0 else 0.0,
                "mean_predicted_probability": float(prob_flat.mean()) if prob_flat.size > 0 else 0.0,
                "mae": float(np.mean(np.abs(prob_flat - fraction_flat))) if fraction_flat.size > 0 else 0.0,
                "rmse": float(np.sqrt(np.mean((prob_flat - fraction_flat) ** 2))) if fraction_flat.size > 0 else 0.0,
                "brier": float(np.mean((prob_flat - fraction_flat) ** 2)) if fraction_flat.size > 0 else 0.0,
            },
        }
    return out


def _region_mask_to_bool(mask: np.ndarray, sample_shape: tuple[int, ...]) -> np.ndarray:
    region = np.asarray(mask).astype(bool)
    if region.shape == sample_shape:
        return region
    raise ValueError(f"Region mask shape {region.shape} does not match sample shape {sample_shape}")


def region_metric_bundle(
    prob_maps: np.ndarray,
    target_maps: np.ndarray,
    mask: np.ndarray,
    thresholds: Iterable[float],
    topk_fractions: Iterable[float],
    n_bins: int,
    reference_positive_rate: float | None = None,
    sample_times: np.ndarray | None = None,
    temporal_tolerances_steps: Iterable[int] | None = None,
    spatial_tolerances_radii: Iterable[int] | None = None,
    time_step_hours: int = 24,
) -> Dict[str, object]:
    region = _region_mask_to_bool(mask, prob_maps.shape[1:])
    prob = prob_maps[:, region].reshape(-1)
    target = target_maps[:, region].reshape(-1)
    positive_rate = float(target.mean()) if target.size > 0 else 0.0
    metrics: Dict[str, object] = {
        "mask_cells": int(region.sum()),
        "mask_fraction": float(region.mean()),
        "positive_rate": positive_rate,
        "positive_cells": int(target.sum()) if target.size > 0 else 0,
        "total_cells": int(target.size),
        "pr_auc": float(average_precision_score(target, prob)) if float(target.sum()) > 0 else 0.0,
        "brier": float(np.mean((prob - target) ** 2)) if target.size > 0 else 0.0,
        "log_score": log_score(prob, target) if target.size > 0 else 0.0,
        "ece": expected_calibration_error(prob, target, n_bins) if target.size > 0 else 0.0,
        "reliability_bins": reliability_bins(prob, target, n_bins) if target.size > 0 else [],
        "threshold_metrics": {f"{float(t):.4f}": threshold_metrics(prob, target, float(t)) for t in thresholds}
        if target.size > 0
        else {},
        "topk_area_metrics": topk_area_metrics(prob, target, topk_fractions) if target.size > 0 else {},
    }
    if target.size > 0 and float(np.unique(target).size) > 1:
        metrics["auroc"] = float(roc_auc_score(target, prob))
    else:
        metrics["auroc"] = 0.0
    ref_rate = reference_positive_rate if reference_positive_rate is not None else positive_rate
    brier_ref = float(np.mean((ref_rate - target) ** 2)) if target.size > 0 else 0.0
    metrics["reference_positive_rate"] = float(ref_rate)
    metrics["brier_skill_score"] = float(1.0 - metrics["brier"] / brier_ref) if brier_ref > 0 else 0.0
    if sample_times is not None:
        temporal_values = [int(v) for v in (temporal_tolerances_steps or [])]
        spatial_values = [int(v) for v in (spatial_tolerances_radii or [])]
        tolerant_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        for temporal_steps in temporal_values:
            for spatial_radius in spatial_values:
                if temporal_steps == 0 and spatial_radius == 0:
                    continue
                combo_key = f"t{temporal_steps}_s{spatial_radius}"
                tolerant_metrics[combo_key] = {
                    f"{float(t):.4f}": tolerant_threshold_metrics(
                        prob_maps=prob_maps,
                        target_maps=target_maps,
                        sample_times=sample_times,
                        threshold=float(t),
                        temporal_tolerance_steps=temporal_steps,
                        spatial_tolerance_radius=spatial_radius,
                        region_mask=region,
                        time_step_hours=time_step_hours,
                    )
                    for t in thresholds
                }
        if tolerant_metrics:
            metrics["tolerant_threshold_metrics"] = tolerant_metrics
    return metrics


def fss_metrics(
    prob_maps: np.ndarray,
    target_maps: np.ndarray,
    thresholds: Iterable[float],
    radii: Iterable[int],
) -> Dict[str, Dict[str, float]]:
    prob_t = torch.from_numpy(prob_maps.astype(np.float32))
    tgt_t = torch.from_numpy((target_maps > 0.5).astype(np.float32))
    out: Dict[str, Dict[str, float]] = {}
    for threshold in thresholds:
        pred = (prob_t >= float(threshold)).float()
        row: Dict[str, float] = {}
        for radius in radii:
            radius = int(radius)
            kernel = radius * 2 + 1
            frac_pred = F.avg_pool2d(pred.unsqueeze(1), kernel_size=kernel, stride=1, padding=radius).squeeze(1)
            frac_tgt = F.avg_pool2d(tgt_t.unsqueeze(1), kernel_size=kernel, stride=1, padding=radius).squeeze(1)
            mse = torch.mean((frac_pred - frac_tgt) ** 2).item()
            ref = torch.mean(frac_pred**2 + frac_tgt**2).item()
            score = 1.0 - (mse / ref) if ref > 0 else 1.0
            row[str(radius)] = float(score)
        out[f"{float(threshold):.4f}"] = row
    return out


def boundary_metrics(
    prob_maps: np.ndarray,
    target_maps: np.ndarray,
    thresholds: Iterable[float],
    radii: Iterable[int],
    boundary_width: int = 1,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    prob_t = torch.from_numpy(prob_maps.astype(np.float32))
    tgt_boundary = _boundary_mask(torch.from_numpy((target_maps > 0.5).astype(np.float32)), width=boundary_width)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for threshold in thresholds:
        pred_boundary = _boundary_mask((prob_t >= float(threshold)).float(), width=boundary_width)
        row: Dict[str, Dict[str, float]] = {}
        pred_boundary_cells = int(pred_boundary.sum().item())
        tgt_boundary_cells = int(tgt_boundary.sum().item())
        for radius in radii:
            radius = int(radius)
            pred_band = _spatial_dilate(pred_boundary.float(), radius)
            tgt_band = _spatial_dilate(tgt_boundary.float(), radius)
            band_intersection = int((pred_band & tgt_band).sum().item())
            band_union = int((pred_band | tgt_band).sum().item())
            pred_match = int((pred_boundary & tgt_band).sum().item())
            tgt_match = int((tgt_boundary & pred_band).sum().item())
            denom = pred_boundary_cells + tgt_boundary_cells
            row[str(radius)] = {
                "boundary_iou": _safe_div(band_intersection, band_union) if band_union > 0 else 1.0,
                "surface_dice": _safe_div(pred_match + tgt_match, denom) if denom > 0 else 1.0,
                "pred_boundary_cells": pred_boundary_cells,
                "target_boundary_cells": tgt_boundary_cells,
            }
        out[f"{float(threshold):.4f}"] = row
    return out


def buffered_overlap_metrics(
    prob_maps: np.ndarray,
    target_maps: np.ndarray,
    thresholds: Iterable[float],
    radii: Iterable[int],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    prob_t = torch.from_numpy(prob_maps.astype(np.float32))
    target_bool = torch.from_numpy((target_maps > 0.5).astype(np.float32)) > 0.5
    target_cells = int(target_bool.sum().item())
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for threshold in thresholds:
        pred_bool = prob_t >= float(threshold)
        pred_cells = int(pred_bool.sum().item())
        row: Dict[str, Dict[str, float]] = {}
        for radius in radii:
            radius = int(radius)
            pred_dilated = _spatial_dilate(pred_bool.float(), radius)
            target_dilated = _spatial_dilate(target_bool.float(), radius)
            pred_match = int((pred_bool & target_dilated).sum().item())
            target_match = int((target_bool & pred_dilated).sum().item())
            buffered_precision = _safe_div(pred_match, pred_cells)
            buffered_recall = _safe_div(target_match, target_cells)
            buffered_f1 = _safe_div(
                2.0 * buffered_precision * buffered_recall,
                buffered_precision + buffered_recall,
            )
            pred_d_target_intersection = int((pred_dilated & target_bool).sum().item())
            pred_d_target_union = int((pred_dilated | target_bool).sum().item())
            pred_target_d_intersection = int((pred_bool & target_dilated).sum().item())
            pred_target_d_union = int((pred_bool | target_dilated).sum().item())
            sym_intersection = int((pred_dilated & target_dilated).sum().item())
            sym_union = int((pred_dilated | target_dilated).sum().item())
            row[str(radius)] = {
                "predicted_positive_cells": pred_cells,
                "target_positive_cells": target_cells,
                "buffered_precision": buffered_precision,
                "buffered_recall": buffered_recall,
                "buffered_f1": buffered_f1,
                "pred_dilated_iou": _safe_div(pred_d_target_intersection, pred_d_target_union),
                "target_dilated_iou": _safe_div(pred_target_d_intersection, pred_target_d_union),
                "symmetric_dilated_iou": _safe_div(sym_intersection, sym_union),
            }
        out[f"{float(threshold):.4f}"] = row
    return out


def distance_transform_metrics(
    prob_maps: np.ndarray,
    target_maps: np.ndarray,
    thresholds: Iterable[float],
    cutoffs: Iterable[int],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if _scipy_ndimage is None:
        return {}
    target_bool = target_maps > 0.5
    cutoff_values = [int(v) for v in cutoffs if int(v) > 0]
    if not cutoff_values:
        return {}
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for threshold in thresholds:
        pred_bool = prob_maps >= float(threshold)
        per_cutoff: Dict[int, Dict[str, object]] = {
            cutoff: {
                "pred_to_target": [],
                "target_to_pred": [],
                "baddeley_delta": [],
                "empty_empty": 0,
                "pred_empty_target_nonempty": 0,
                "pred_nonempty_target_empty": 0,
            }
            for cutoff in cutoff_values
        }
        for idx in range(int(pred_bool.shape[0])):
            pred_i = pred_bool[idx]
            target_i = target_bool[idx]
            pred_any = bool(pred_i.any())
            target_any = bool(target_i.any())
            if pred_any:
                dt_pred = _scipy_ndimage.distance_transform_edt(~pred_i)
            else:
                dt_pred = None
            if target_any:
                dt_target = _scipy_ndimage.distance_transform_edt(~target_i)
            else:
                dt_target = None
            for cutoff in cutoff_values:
                bucket = per_cutoff[cutoff]
                pred_to_target: List[float] = bucket["pred_to_target"]  # type: ignore[assignment]
                target_to_pred: List[float] = bucket["target_to_pred"]  # type: ignore[assignment]
                baddeley_delta: List[float] = bucket["baddeley_delta"]  # type: ignore[assignment]
                if pred_any and target_any and dt_pred is not None and dt_target is not None:
                    pred_to_target.extend(np.minimum(dt_target[pred_i], cutoff).astype(np.float32).tolist())
                    target_to_pred.extend(np.minimum(dt_pred[target_i], cutoff).astype(np.float32).tolist())
                    pred_dt_clip = np.minimum(dt_pred, cutoff)
                    target_dt_clip = np.minimum(dt_target, cutoff)
                    baddeley_delta.append(float(np.sqrt(np.mean((pred_dt_clip - target_dt_clip) ** 2))))
                elif pred_any and not target_any:
                    pred_to_target.extend([float(cutoff)] * int(pred_i.sum()))
                    baddeley_delta.append(float(cutoff))
                    bucket["pred_nonempty_target_empty"] = int(bucket["pred_nonempty_target_empty"]) + 1
                elif target_any and not pred_any:
                    target_to_pred.extend([float(cutoff)] * int(target_i.sum()))
                    baddeley_delta.append(float(cutoff))
                    bucket["pred_empty_target_nonempty"] = int(bucket["pred_empty_target_nonempty"]) + 1
                else:
                    baddeley_delta.append(0.0)
                    bucket["empty_empty"] = int(bucket["empty_empty"]) + 1
        row: Dict[str, Dict[str, float]] = {}
        for cutoff in cutoff_values:
            bucket = per_cutoff[cutoff]
            pred_to_target_arr = np.asarray(bucket["pred_to_target"], dtype=np.float32)
            target_to_pred_arr = np.asarray(bucket["target_to_pred"], dtype=np.float32)
            if pred_to_target_arr.size and target_to_pred_arr.size:
                symmetric = np.concatenate([pred_to_target_arr, target_to_pred_arr])
            elif pred_to_target_arr.size:
                symmetric = pred_to_target_arr
            else:
                symmetric = target_to_pred_arr
            baddeley_arr = np.asarray(bucket["baddeley_delta"], dtype=np.float32)
            row[str(cutoff)] = {
                "distance_cutoff": float(cutoff),
                "mean_pred_to_target_distance": float(pred_to_target_arr.mean()) if pred_to_target_arr.size else 0.0,
                "mean_target_to_pred_distance": float(target_to_pred_arr.mean()) if target_to_pred_arr.size else 0.0,
                "mean_symmetric_surface_distance": float(symmetric.mean()) if symmetric.size else 0.0,
                "hausdorff95_distance": float(np.percentile(symmetric, 95)) if symmetric.size else 0.0,
                "baddeley_delta_p2": float(baddeley_arr.mean()) if baddeley_arr.size else 0.0,
                "empty_empty_samples": float(bucket["empty_empty"]),
                "pred_empty_target_nonempty_samples": float(bucket["pred_empty_target_nonempty"]),
                "pred_nonempty_target_empty_samples": float(bucket["pred_nonempty_target_empty"]),
            }
        out[f"{float(threshold):.4f}"] = row
    return out


def metric_bundle(
    prob_maps: np.ndarray,
    target_maps: np.ndarray,
    thresholds: Iterable[float],
    topk_fractions: Iterable[float],
    fss_radii: Iterable[int],
    n_bins: int,
    boundary_radii: Iterable[int] | None = None,
    coarsen_factors: Iterable[int] | None = None,
    distance_cutoffs: Iterable[int] | None = None,
    reference_positive_rate: float | None = None,
    sample_times: np.ndarray | None = None,
    temporal_tolerances_steps: Iterable[int] | None = None,
    spatial_tolerances_radii: Iterable[int] | None = None,
    region_masks: Dict[str, np.ndarray] | None = None,
    time_step_hours: int = 24,
) -> Dict[str, object]:
    prob = prob_maps.reshape(-1)
    target = target_maps.reshape(-1)
    positive_rate = float(target.mean())
    metrics: Dict[str, object] = {
        "positive_rate": positive_rate,
        "positive_cells": int(target.sum()),
        "total_cells": int(target.size),
        "pr_auc": float(average_precision_score(target, prob)) if float(target.sum()) > 0 else 0.0,
        "brier": float(np.mean((prob - target) ** 2)),
        "log_score": log_score(prob, target),
        "ece": expected_calibration_error(prob, target, n_bins),
        "reliability_bins": reliability_bins(prob, target, n_bins),
        "threshold_metrics": {f"{float(t):.4f}": threshold_metrics(prob, target, float(t)) for t in thresholds},
        "topk_area_metrics": topk_area_metrics(prob, target, topk_fractions),
        "fss": fss_metrics(prob_maps, target_maps, thresholds, fss_radii),
        "boundary_metrics": boundary_metrics(
            prob_maps,
            target_maps,
            thresholds,
            boundary_radii if boundary_radii is not None else [1, 2, 4],
        ),
        "buffered_overlap_metrics": buffered_overlap_metrics(
            prob_maps,
            target_maps,
            thresholds,
            boundary_radii if boundary_radii is not None else [1, 2, 4],
        ),
        "coarsened_metrics": coarsened_metrics(
            prob_maps,
            target_maps,
            thresholds,
            coarsen_factors if coarsen_factors is not None else [2, 4, 8],
            reference_positive_rate=reference_positive_rate,
        ),
    }
    if distance_cutoffs is not None:
        metrics["distance_transform_metrics"] = distance_transform_metrics(
            prob_maps,
            target_maps,
            thresholds,
            distance_cutoffs,
        )
    if float(np.unique(target).size) > 1:
        metrics["auroc"] = float(roc_auc_score(target, prob))
    else:
        metrics["auroc"] = 0.0
    ref_rate = reference_positive_rate if reference_positive_rate is not None else positive_rate
    brier_ref = float(np.mean((ref_rate - target) ** 2))
    metrics["reference_positive_rate"] = float(ref_rate)
    metrics["brier_skill_score"] = float(1.0 - metrics["brier"] / brier_ref) if brier_ref > 0 else 0.0
    if sample_times is not None:
        temporal_values = [int(v) for v in (temporal_tolerances_steps or [])]
        spatial_values = [int(v) for v in (spatial_tolerances_radii or [])]
        tolerant_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        for temporal_steps in temporal_values:
            for spatial_radius in spatial_values:
                if temporal_steps == 0 and spatial_radius == 0:
                    continue
                combo_key = f"t{temporal_steps}_s{spatial_radius}"
                tolerant_metrics[combo_key] = {
                    f"{float(t):.4f}": tolerant_threshold_metrics(
                        prob_maps=prob_maps,
                        target_maps=target_maps,
                        sample_times=sample_times,
                        threshold=float(t),
                        temporal_tolerance_steps=temporal_steps,
                        spatial_tolerance_radius=spatial_radius,
                        time_step_hours=time_step_hours,
                    )
                    for t in thresholds
                }
        if tolerant_metrics:
            metrics["tolerant_threshold_metrics"] = tolerant_metrics
        neighborhood_metrics = neighborhood_contingency_metrics(
            prob_maps=prob_maps,
            target_maps=target_maps,
            sample_times=sample_times,
            thresholds=thresholds,
            temporal_tolerances_steps=temporal_values,
            spatial_tolerances_radii=spatial_values,
            time_step_hours=time_step_hours,
        )
        if neighborhood_metrics:
            metrics["neighborhood_contingency_metrics"] = neighborhood_metrics
    if region_masks:
        metrics["region_metrics"] = {
            name: region_metric_bundle(
                prob_maps=prob_maps,
                target_maps=target_maps,
                mask=mask,
                thresholds=thresholds,
                topk_fractions=topk_fractions,
                n_bins=n_bins,
                reference_positive_rate=reference_positive_rate,
                sample_times=sample_times,
                temporal_tolerances_steps=temporal_tolerances_steps,
                spatial_tolerances_radii=spatial_tolerances_radii,
                time_step_hours=time_step_hours,
            )
            for name, mask in region_masks.items()
        }
    return metrics
