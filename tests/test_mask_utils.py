"""Unit tests for the watershed mask decoder.

These use synthetic masks so they run instantly with no dataset present —
they verify the *logic*, and pin the class mapping so it can never silently
drift.
"""

import numpy as np
import pytest

from sam3_seg.data import mask_utils as mu


def test_mapping_is_bijective():
    # Every value maps to a unique name and back.
    assert len(mu.CHOLECSEG8K_CLASSES) == len(mu.NAME_TO_VALUE)
    for value, name in mu.CHOLECSEG8K_CLASSES.items():
        assert mu.NAME_TO_VALUE[name] == value


def test_tool_values_are_locked():
    # The two facts the whole project depends on. If these ever change, fail loudly.
    assert mu.NAME_TO_VALUE["grasper"] == 31
    assert mu.NAME_TO_VALUE["l_hook_electrocautery"] == 32


def _toy_mask():
    # 4x4 mask: top-left grasper(31), top-right l-hook(32),
    # bottom-left liver(21), bottom-right artifact(255).
    m = np.zeros((4, 4), dtype=np.uint8)
    m[:2, :2] = 31
    m[:2, 2:] = 32
    m[2:, :2] = 21
    m[2:, 2:] = 255
    return m


def test_binary_mask_for_class_selects_exact_pixels():
    m = _toy_mask()
    g = mu.binary_mask_for_class(m, "grasper")
    assert g.dtype == bool
    assert g.sum() == 4                 # exactly the 2x2 block
    assert g[0, 0] and not g[0, 2]      # grasper region True, l-hook region False


def test_tool_masks_returns_both_tools_disjoint():
    m = _toy_mask()
    masks = mu.tool_masks(m)
    assert set(masks) == {"grasper", "l_hook_electrocautery"}
    # Two tools never overlap.
    assert not np.logical_and(masks["grasper"], masks["l_hook_electrocautery"]).any()


def test_present_classes_excludes_artifacts():
    m = _toy_mask()
    present = set(mu.present_classes(m))
    assert present == {"grasper", "l_hook_electrocautery", "liver"}
    assert "255" not in present         # artifact value 255 must not appear


def test_unknown_class_raises():
    with pytest.raises(KeyError):
        mu.binary_mask_for_class(_toy_mask(), "spaceship")


def test_pixel_fraction():
    m = _toy_mask()
    assert mu.class_pixel_fraction(m, "grasper") == pytest.approx(4 / 16)