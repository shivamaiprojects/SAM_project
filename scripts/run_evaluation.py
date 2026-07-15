"""Evaluate saved TrackingResults and write a metrics summary."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from sam3_seg.data.dataset import CholecSeg8k
from sam3_seg.data.frame_loader import load_clip_gt
from sam3_seg.evaluation.evaluator import evaluate_clip
from sam3_seg.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--config", default="configs/colab.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = CholecSeg8k(cfg["paths"]["raw_data"])
    results_dir = Path(args.results_dir) if args.results_dir \
        else cfg["paths"]["outputs"] / "tracking"

    rows = []
    for pkl in sorted(results_dir.glob("*.pkl")):
        with open(pkl, "rb") as f:
            result = pickle.load(f)
        clip_id = pkl.stem.replace("_", "/", 1).rsplit("_tool", 1)[0]
        clip = ds.get_clip(clip_id)
        gt = load_clip_gt(ds, clip)
        ev = evaluate_clip(clip_id, result, gt)

        rows.append({"clip": clip_id, "scope": "semantic (all tools)", **ev.semantic})
        for cls, sc in ev.per_class.items():
            rows.append({"clip": clip_id, "scope": cls, **sc})
        print(f"{clip_id}: semantic IoU {ev.semantic['iou']:.3f}, "
              f"Dice {ev.semantic['dice']:.3f}  (n={ev.semantic['n']})")

    df = pd.DataFrame(rows)
    out = cfg["paths"]["outputs"] / "metrics" / "benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()