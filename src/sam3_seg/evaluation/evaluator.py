"""Evaluate a SAM 3 TrackingResult against CholecSeg8k ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sam3_seg.data.mask_utils import TOOL_CLASSES
from sam3_seg.evaluation.metrics import MaskScores, score_masks
from sam3_seg.models.sam3_video import TrackingResult


@dataclass(frozen=True)
class ClipEvaluation:
    clip_id: str
    semantic: dict[str, float]
    per_class: dict[str, dict[str, float]]
    n_frames: int


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _aggregate(scores: list[MaskScores], gt_present_only: bool) -> dict[str, float]:
    rows = [s for s in scores if s.gt_present] if gt_present_only else scores
    if not rows:
        return {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "n": 0}
    return {
        "iou": _mean([s.iou for s in rows]),
        "dice": _mean([s.dice for s in rows]),
        "precision": _mean([s.precision for s in rows]),
        "recall": _mean([s.recall for s in rows]),
        "n": len(rows),
    }


def _assign_objects(frame_result, gt_frame: dict[str, np.ndarray]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for oid in frame_result.object_ids:
        best_cls, best_iou = None, 0.0
        for cls in TOOL_CLASSES:
            s = score_masks(frame_result.masks[oid], gt_frame[cls])
            if s.iou > best_iou:
                best_cls, best_iou = cls, s.iou
        if best_cls is not None:
            mapping[oid] = best_cls
    return mapping


def evaluate_clip(
    clip_id: str,
    result: TrackingResult,
    gt: dict[str, list[np.ndarray]],
    gt_present_only: bool = True,
) -> ClipEvaluation:
    n_frames = len(result.frames)
    shape = (result.height, result.width)

    semantic_scores: list[MaskScores] = []
    per_class_scores: dict[str, list[MaskScores]] = {c: [] for c in TOOL_CLASSES}

    for fidx in range(n_frames):
        fr = result.frames[fidx]
        gt_frame = {c: gt[c][fidx] for c in TOOL_CLASSES}

        pred_any = np.zeros(shape, dtype=bool)
        for oid in fr.object_ids:
            pred_any |= fr.masks[oid]
        gt_any = np.zeros(shape, dtype=bool)
        for c in TOOL_CLASSES:
            gt_any |= gt_frame[c]
        semantic_scores.append(score_masks(pred_any, gt_any))

        obj_to_cls = _assign_objects(fr, gt_frame)
        for cls in TOOL_CLASSES:
            pred_cls = np.zeros(shape, dtype=bool)
            for oid, assigned in obj_to_cls.items():
                if assigned == cls:
                    pred_cls |= fr.masks[oid]
            per_class_scores[cls].append(score_masks(pred_cls, gt_frame[cls]))

    return ClipEvaluation(
        clip_id=clip_id,
        semantic=_aggregate(semantic_scores, gt_present_only),
        per_class={c: _aggregate(per_class_scores[c], gt_present_only)
                   for c in TOOL_CLASSES},
        n_frames=n_frames,
    )