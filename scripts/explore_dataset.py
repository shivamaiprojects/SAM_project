"""

Outputs (all under outputs/figures/exploration/):
  - contact_sheet.png        : frame | color mask | watershed, for N random clips
  - value_isolation_*.png    : each watershed value highlighted on a real frame
  - value_presence.png       : which values appear in which sampled clips
Console: structure report + unique watershed pixel value table.
"""

import random
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from sam3_seg.utils.config import load_config

N_CONTACT_CLIPS = 6      # clips shown in the contact sheet
N_PRESENCE_CLIPS = 30    # clips sampled for the presence matrix
HIGHLIGHT = np.array([255, 60, 60], dtype=np.uint8)  # red overlay


# ---------------------------------------------------------------- helpers --
def read_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def read_watershed(path: Path) -> np.ndarray:
    ws = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return ws[..., 0] if ws.ndim == 3 else ws


def clip_frame_stems(clip: Path) -> list[str]:
    return [p.name.replace("_endo.png", "") for p in sorted(clip.glob("*_endo.png"))]


def all_clips(root: Path) -> list[Path]:
    return sorted(
        c for v in root.iterdir() if v.is_dir() for c in v.iterdir() if c.is_dir()
    )


def overlay_value(frame: np.ndarray, ws: np.ndarray, value: int) -> np.ndarray:
    """Return frame with pixels equal to `value` tinted red."""
    out = frame.copy()
    m = ws == value
    out[m] = (0.35 * out[m] + 0.65 * HIGHLIGHT).astype(np.uint8)
    return out


# ------------------------------------------------------------------- main --
def main() -> None:
    cfg = load_config()
    root: Path = cfg["paths"]["raw_data"]
    assert root.exists(), f"Dataset not found at {root} — run download_data.py first"
    fig_dir = cfg["paths"]["outputs"] / "figures" / "exploration"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["project"]["seed"])

    # ---- 1. Structure report ----------------------------------------------
    videos = sorted(d for d in root.iterdir() if d.is_dir())
    clips = all_clips(root)
    frames_per_clip = Counter(len(clip_frame_stems(c)) for c in clips)
    print(f"Videos: {len(videos)} | Clips: {len(clips)}")
    print(f"Frames-per-clip distribution: {dict(frames_per_clip)}")

    # ---- 2. Global unique watershed values (1 mask per clip) ---------------
    value_counter: Counter[int] = Counter()
    clip_values: dict[str, set[int]] = {}
    for c in clips:
        ws_files = sorted(c.glob("*_watershed_mask.png"))
        if not ws_files:
            continue
        ws = read_watershed(ws_files[0])
        vals = np.unique(ws)
        clip_values[f"{c.parent.name}/{c.name}"] = set(vals.tolist())
        for v_, n_ in zip(*np.unique(ws, return_counts=True)):
            value_counter[int(v_)] += int(n_)

    print("\nUnique watershed pixel values (value : total pixels in sample):")
    for val, cnt in sorted(value_counter.items()):
        pct = 100 * cnt / sum(value_counter.values())
        print(f"  {val:>4} : {cnt:>12,}  ({pct:5.2f}%)")

    # ---- 3. Contact sheet across random clips ------------------------------
    sample_clips = rng.sample(clips, k=min(N_CONTACT_CLIPS, len(clips)))
    fig, axes = plt.subplots(len(sample_clips), 3, figsize=(15, 4 * len(sample_clips)))
    for row, clip in zip(axes, sample_clips):
        stem = rng.choice(clip_frame_stems(clip))
        row[0].imshow(read_rgb(clip / f"{stem}_endo.png"))
        row[0].set_title(f"{clip.parent.name}/{clip.name}\n{stem} — frame", fontsize=9)
        row[1].imshow(read_rgb(clip / f"{stem}_endo_color_mask.png"))
        row[1].set_title("color mask", fontsize=9)
        row[2].imshow(read_watershed(clip / f"{stem}_endo_watershed_mask.png"),
                      cmap="tab20", interpolation="nearest")
        row[2].set_title("watershed mask", fontsize=9)
        for ax in row:
            ax.axis("off")
    fig.savefig(fig_dir / "contact_sheet.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {fig_dir / 'contact_sheet.png'}")

    # ---- 4. Value isolation panels -----------------------------------------
    # Pick the clip containing the MOST distinct values so we see many classes.
    richest_name = max(clip_values, key=lambda k: len(clip_values[k]))
    v_name, c_name = richest_name.split("/")
    clip = root / v_name / c_name
    stem = clip_frame_stems(clip)[len(clip_frame_stems(clip)) // 2]  # middle frame
    frame = read_rgb(clip / f"{stem}_endo.png")
    ws = read_watershed(clip / f"{stem}_endo_watershed_mask.png")
    values_here = [int(v) for v in np.unique(ws)]

    n = len(values_here) + 1
    cols = 4
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
    axes = np.atleast_2d(axes)
    axes.flat[0].imshow(frame)
    axes.flat[0].set_title(f"{richest_name}/{stem}\noriginal", fontsize=9)
    for ax, val in zip(list(axes.flat)[1:], values_here):
        ax.imshow(overlay_value(frame, ws, val))
        share = 100 * float((ws == val).mean())
        ax.set_title(f"value = {val}  ({share:.1f}% of frame)", fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(fig_dir / f"value_isolation_{v_name}_{c_name}.png",
                dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_dir / f'value_isolation_{v_name}_{c_name}.png'}")

    # ---- 5. Value-presence matrix ------------------------------------------
    sampled = rng.sample(sorted(clip_values), k=min(N_PRESENCE_CLIPS, len(clip_values)))
    all_vals = sorted(value_counter)
    grid = np.array([[v in clip_values[c] for v in all_vals] for c in sampled])
    fig, ax = plt.subplots(figsize=(0.5 * len(all_vals) + 3, 0.28 * len(sampled) + 2))
    ax.imshow(grid, cmap="Greys", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(all_vals)), all_vals)
    ax.set_yticks(range(len(sampled)), sampled, fontsize=6)
    ax.set_xlabel("watershed pixel value")
    ax.set_title("Which values appear in which clips (black = present)")
    fig.savefig(fig_dir / "value_presence.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_dir / 'value_presence.png'}")


if __name__ == "__main__":
    main()