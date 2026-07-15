"""Load a clip's frames and ground-truth tool masks as ordered arrays."""

from __future__ import annotations

import numpy as np

from sam3_seg.data.dataset import CholecSeg8k, Clip
from sam3_seg.data.mask_utils import TOOL_CLASSES, binary_mask_for_class


def load_clip_frames(ds: CholecSeg8k, clip: Clip) -> list[np.ndarray]:
    return [ds.load_image(fp) for fp in clip.frames]


def load_clip_gt(ds: CholecSeg8k, clip: Clip) -> dict[str, list[np.ndarray]]:
    gt: dict[str, list[np.ndarray]] = {name: [] for name in TOOL_CLASSES}
    for fp in clip.frames:
        ws = ds.load_watershed(fp)
        for name in TOOL_CLASSES:
            gt[name].append(binary_mask_for_class(ws, name))
    return gt