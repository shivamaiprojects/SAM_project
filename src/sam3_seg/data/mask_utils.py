"""CholecSeg8k watershed-mask decoding.

Ground truth is stored in the *watershed* PNGs, where each class is a single
integer pixel value repeated across all three channels. The mapping below was
verified empirically (scripts/explore_dataset.py) against the actual pixel
values in the dataset, then cross-referenced to the official class names on the
Kaggle dataset page. NOTE: the Kaggle page writes these as "hex" (#313131) but
the bytes on disk are the plain decimal values used here (31, 32, ...).
"""

from __future__ import annotations

import numpy as np

# ---- The authoritative mapping: pixel value -> class name ------------------
CHOLECSEG8K_CLASSES: dict[int, str] = {
    50: "black_background",
    11: "abdominal_wall",
    21: "liver",
    12: "fat",
    13: "gi_tract",
    31: "grasper",
    22: "gallbladder",
    23: "connective_tissue",
    24: "blood",
    25: "cystic_duct",
    32: "l_hook_electrocautery",
    33: "hepatic_vein",
    5: "liver_ligament",
}

NAME_TO_VALUE: dict[str, int] = {v: k for k, v in CHOLECSEG8K_CLASSES.items()}

# Values seen in the data that are NOT real classes (compression/border noise).
ARTIFACT_VALUES: frozenset[int] = frozenset({0, 255})

# The subset this project targets. Order defines a stable class index (0, 1).
TOOL_CLASSES: tuple[str, ...] = ("grasper", "l_hook_electrocautery")


def read_watershed_mask(path) -> np.ndarray:
    """Load a watershed PNG as a single-channel uint8 array of class values."""
    import cv2

    ws = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if ws is None:
        raise FileNotFoundError(f"Could not read watershed mask: {path}")
    if ws.ndim == 3:
        # All channels identical by construction; take the first.
        ws = ws[..., 0]
    return ws.astype(np.uint8)


def binary_mask_for_class(watershed: np.ndarray, class_name: str) -> np.ndarray:
    """Return a boolean mask where True == pixels belonging to `class_name`."""
    if class_name not in NAME_TO_VALUE:
        raise KeyError(
            f"Unknown class '{class_name}'. Valid: {sorted(NAME_TO_VALUE)}"
        )
    return watershed == NAME_TO_VALUE[class_name]


def tool_masks(watershed: np.ndarray) -> dict[str, np.ndarray]:
    """Return {tool_name: boolean_mask} for each targeted tool class."""
    return {name: binary_mask_for_class(watershed, name) for name in TOOL_CLASSES}


def present_classes(watershed: np.ndarray) -> list[str]:
    """Names of real classes present in this mask (artifacts excluded)."""
    vals = set(np.unique(watershed).tolist()) - ARTIFACT_VALUES
    return [CHOLECSEG8K_CLASSES[v] for v in vals if v in CHOLECSEG8K_CLASSES]


def class_pixel_fraction(watershed: np.ndarray, class_name: str) -> float:
    """Fraction of the frame occupied by a class — useful for filtering clips."""
    return float(binary_mask_for_class(watershed, class_name).mean())