"""Unit tests for region segmentation metrics."""

import numpy as np
import pytest

from sam3_seg.evaluation.metrics import score_masks


def test_perfect_overlap():
    m = np.zeros((10, 10), dtype=bool)
    m[2:6, 2:6] = True
    s = score_masks(m, m)
    assert s.iou == 1.0 and s.dice == 1.0
    assert s.precision == 1.0 and s.recall == 1.0


def test_no_overlap():
    pred = np.zeros((10, 10), dtype=bool); pred[0:3, 0:3] = True
    gt = np.zeros((10, 10), dtype=bool); gt[7:10, 7:10] = True
    s = score_masks(pred, gt)
    assert s.iou == 0.0 and s.dice == 0.0


def test_half_overlap_known_values():
    # pred = 4x4 block, gt = 4x4 block shifted to share exactly half.
    pred = np.zeros((4, 8), dtype=bool); pred[:, 0:4] = True
    gt = np.zeros((4, 8), dtype=bool); gt[:, 2:6] = True
    # intersection = 4x2 = 8, each mask = 16, union = 24
    s = score_masks(pred, gt)
    assert s.iou == pytest.approx(8 / 24)
    assert s.dice == pytest.approx(2 * 8 / (16 + 16))
    assert s.precision == pytest.approx(8 / 16)
    assert s.recall == pytest.approx(8 / 16)


def test_both_empty_is_perfect():
    empty = np.zeros((10, 10), dtype=bool)
    s = score_masks(empty, empty)
    assert s.iou == 1.0 and s.dice == 1.0
    assert not s.gt_present and not s.pred_present


def test_gt_present_pred_empty_is_zero():
    pred = np.zeros((10, 10), dtype=bool)
    gt = np.zeros((10, 10), dtype=bool); gt[2:6, 2:6] = True
    s = score_masks(pred, gt)
    assert s.iou == 0.0 and s.recall == 0.0
    assert s.gt_present and not s.pred_present


def test_dice_iou_relationship():
    pred = np.zeros((10, 10), dtype=bool); pred[2:8, 2:7] = True
    gt = np.zeros((10, 10), dtype=bool); gt[3:8, 2:8] = True
    s = score_masks(pred, gt)
    assert s.dice == pytest.approx(2 * s.iou / (1 + s.iou))