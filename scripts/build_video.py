"""Reconstruct CholecSeg8k clips (80 frames) into playable MP4 videos.

Usage:
  python scripts/build_video.py --clip video01/video01_00080
  python scripts/build_video.py --clip video01_00080 --with-mask
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from sam3_seg.data.dataset import CholecSeg8k
from sam3_seg.data.mask_utils import TOOL_CLASSES, binary_mask_for_class
from sam3_seg.utils.config import load_config

TOOL_COLORS = {                       # BGR for OpenCV writing
    "grasper": (60, 220, 60),         # green
    "l_hook_electrocautery": (60, 60, 240),  # red
}


def overlay_gt(rgb: np.ndarray, ws: np.ndarray) -> np.ndarray:
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for name in TOOL_CLASSES:
        m = binary_mask_for_class(ws, name)
        out[m] = (0.4 * out[m] + 0.6 * np.array(TOOL_COLORS[name])).astype(np.uint8)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="clip id, e.g. video01/video01_00080")
    ap.add_argument("--with-mask", action="store_true",
                    help="overlay ground-truth tool masks")
    args = ap.parse_args()

    cfg = load_config()
    ds = CholecSeg8k(cfg["paths"]["raw_data"])
    clip = ds.get_clip(args.clip)
    fps = cfg["data"]["fps"]

    first = ds.load_image(clip.frames[0])
    h, w = first.shape[:2]
    out_dir = cfg["paths"]["outputs"] / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_gt" if args.with_mask else ""
    out_path = out_dir / f"{clip.video}_{clip.name}{suffix}.mp4"

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for fp in clip.frames:
        rgb = ds.load_image(fp)
        if args.with_mask:
            frame_bgr = overlay_gt(rgb, ds.load_watershed(fp))
        else:
            frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()
    print(f"Wrote {len(clip)} frames -> {out_path}")


if __name__ == "__main__":
    main()