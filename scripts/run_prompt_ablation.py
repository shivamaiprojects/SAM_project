"""Measure detection rate and mask quality per text prompt across sampled frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sam3_seg.data.dataset import CholecSeg8k
from sam3_seg.data.mask_utils import binary_mask_for_class
from sam3_seg.evaluation.metrics import score_masks
from sam3_seg.models.sam3_image import Sam3ImageSegmenter, union_mask
from sam3_seg.utils.config import load_config


def sample_frames_with_class(ds, class_name, n_frames, min_fraction=0.02, seed=42):
    rng = np.random.default_rng(seed)
    hits = []
    for clip in ds.clips:
        for fp in clip.frames:
            ws = ds.load_watershed(fp)
            if binary_mask_for_class(ws, class_name).mean() >= min_fraction:
                hits.append((fp, ws))
    idx = rng.choice(len(hits), size=min(n_frames, len(hits)), replace=False)
    return [hits[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-name", default="grasper")
    ap.add_argument("--n-frames", type=int, default=40)
    ap.add_argument("--prompts", nargs="+",
                    default=["grasper", "tool", "forceps", "surgical instrument",
                             "surgical tool", "clamp", "metal instrument"])
    ap.add_argument("--config", default="configs/colab.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = CholecSeg8k(cfg["paths"]["raw_data"])
    seg = Sam3ImageSegmenter(device=cfg["model"]["device"], score_threshold=0.3,
                             bpe_path=cfg["model"].get("bpe_path"))

    frames = sample_frames_with_class(ds, args.class_name, args.n_frames)
    print(f"Sampled {len(frames)} frames containing '{args.class_name}'")

    results = {}
    for prompt in args.prompts:
        detected, ious, scores = 0, [], []
        for fp, ws in frames:
            img = ds.load_image(fp)
            gt = binary_mask_for_class(ws, args.class_name)
            dets = seg.predict(img, prompt)
            if dets:
                detected += 1
                pred = union_mask(dets, img.shape[:2])
                ious.append(score_masks(pred, gt).iou)
                scores.append(max(d.score for d in dets))
        results[prompt] = {
            "detection_rate": detected / len(frames),
            "mean_iou_when_detected": float(np.mean(ious)) if ious else 0.0,
            "mean_top_score": float(np.mean(scores)) if scores else 0.0,
            "n_detected": detected,
        }
        r = results[prompt]
        print(f"  {prompt:22s} det {r['detection_rate']:.0%}  "
              f"IoU {r['mean_iou_when_detected']:.3f}  score {r['mean_top_score']:.3f}")

    out = cfg["paths"]["outputs"] / "metrics" / f"ablation_{args.class_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()