"""Wrapper around SAM 3's image-level Promptable Concept Segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Detection:
    """A single detected instance of a prompted concept."""

    mask: np.ndarray
    score: float
    box: np.ndarray

    @property
    def area(self) -> int:
        return int(self.mask.sum())


class Sam3ImageSegmenter:
    """Loads SAM 3 once and answers repeated (image, text prompt) queries."""

    def __init__(
        self,
        device: str = "cuda",
        score_threshold: float = 0.3,
        resolution: int = 1008,
        bpe_path: str | None = None,
    ):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.device = device
        self.score_threshold = score_threshold
        self.model = build_sam3_image_model(bpe_path=bpe_path, device=device)
        self.processor = Sam3Processor(
            self.model,
            resolution=resolution,
            device=device,
            confidence_threshold=score_threshold,
        )
        self._state = None
        self._state_key: int | None = None

    def set_image(self, image: np.ndarray) -> None:
        """Encode an RGB image. Expensive; cached until a new image is set."""
        from PIL import Image

        key = id(image)
        if self._state_key == key:
            return
        with torch.autocast(self.device, dtype=torch.bfloat16), torch.inference_mode():
            self._state = self.processor.set_image(Image.fromarray(image))
        self._state_key = key

    def predict(self, image: np.ndarray, prompt: str) -> list[Detection]:
        """Return detections of `prompt` in `image`, sorted by descending score."""
        self.set_image(image)
        with torch.autocast(self.device, dtype=torch.bfloat16), torch.inference_mode():
            out = self.processor.set_text_prompt(prompt, self._state)

        masks = self._to_numpy(out["masks"])
        scores = self._to_numpy(out["scores"]).reshape(-1)
        boxes = self._to_numpy(out["boxes"]).reshape(-1, 4)

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        detections = [
            Detection(mask=masks[i] > 0, score=float(scores[i]), box=boxes[i])
            for i in range(len(scores))
            if scores[i] >= self.score_threshold
        ]
        return sorted(detections, key=lambda d: d.score, reverse=True)

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
        return np.asarray(x)


def union_mask(detections: list[Detection], shape: tuple[int, int]) -> np.ndarray:
    """Merge all detected instances into one boolean mask."""
    merged = np.zeros(shape, dtype=bool)
    for det in detections:
        merged |= det.mask
    return merged