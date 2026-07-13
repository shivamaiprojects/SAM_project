"""CholecSeg8k dataset indexing and access.

A *clip* is 80 consecutive frames living in one folder. Each frame has a raw
endoscope image and a watershed mask (our ground truth).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sam3_seg.data.mask_utils import class_pixel_fraction, read_watershed_mask


@dataclass(frozen=True)
class FramePaths:
    stem: str
    image: Path
    watershed: Path
    color_mask: Path


@dataclass(frozen=True)
class Clip:
    video: str
    name: str            # e.g. "video01_00080"
    frames: tuple[FramePaths, ...]

    @property
    def clip_id(self) -> str:
        return f"{self.video}/{self.name}"

    def __len__(self) -> int:
        return len(self.frames)


class CholecSeg8k:
    """Indexes the whole dataset; lazily loads pixels on request."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.root}. Run scripts/download_data.py"
            )
        self.clips: list[Clip] = self._index()
        if not self.clips:
            raise RuntimeError(f"No clips found under {self.root}")

    def _index(self) -> list[Clip]:
        clips: list[Clip] = []
        for video_dir in sorted(d for d in self.root.iterdir() if d.is_dir()):
            for clip_dir in sorted(d for d in video_dir.iterdir() if d.is_dir()):
                frames = []
                for img in sorted(clip_dir.glob("*_endo.png")):
                    stem = img.name.replace("_endo.png", "")
                    ws = clip_dir / f"{stem}_endo_watershed_mask.png"
                    cm = clip_dir / f"{stem}_endo_color_mask.png"
                    if ws.exists():
                        frames.append(FramePaths(stem, img, ws, cm))
                if frames:
                    clips.append(
                        Clip(video_dir.name, clip_dir.name, tuple(frames))
                    )
        return clips

    # ---- access -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.clips)

    def get_clip(self, clip_id: str) -> Clip:
        for c in self.clips:
            if c.clip_id == clip_id or c.name == clip_id:
                return c
        raise KeyError(f"Clip '{clip_id}' not found")

    @staticmethod
    def load_image(fp: FramePaths) -> np.ndarray:
        """RGB uint8 image (H, W, 3)."""
        return cv2.cvtColor(cv2.imread(str(fp.image)), cv2.COLOR_BGR2RGB)

    @staticmethod
    def load_watershed(fp: FramePaths) -> np.ndarray:
        """Single-channel class-value mask (H, W)."""
        return read_watershed_mask(fp.watershed)

    # ---- selection helpers -----------------------------------------------
    def clips_containing(self, class_name: str, min_fraction: float = 0.01,
                         min_frames: int = 5) -> list[Clip]:
        """Clips where `class_name` covers >= min_fraction in >= min_frames frames.

        Used to pick good demo/eval clips — e.g. clips where a grasper is
        actually visible for a meaningful stretch, not one stray frame.
        """
        hits = []
        for clip in self.clips:
            count = sum(
                class_pixel_fraction(self.load_watershed(fp), class_name)
                >= min_fraction
                for fp in clip.frames
            )
            if count >= min_frames:
                hits.append(clip)
        return hits

    def summary(self) -> dict:
        n_frames = sum(len(c) for c in self.clips)
        videos = sorted({c.video for c in self.clips})
        return {"videos": len(videos), "clips": len(self.clips),
                "frames": n_frames}