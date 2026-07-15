"""Region-based segmentation metrics for binary masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskScores:
    iou: float
    dice: float
    precision: float
    recall: float
    gt_present: bool
    pred_present: bool


def _counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    return tp, fp, fn


def score_masks(pred: np.ndarray, gt: np.ndarray) -> MaskScores:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    tp, fp, fn = _counts(pred, gt)
    gt_present = bool(gt.any())
    pred_present = bool(pred.any())

    union = tp + fp + fn
    iou = tp / union if union else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0

    return MaskScores(iou=iou, dice=dice, precision=precision, recall=recall,
                      gt_present=gt_present, pred_present=pred_present)