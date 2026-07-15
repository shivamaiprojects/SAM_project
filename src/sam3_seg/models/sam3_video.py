"""Wrapper around SAM 3 video segmentation and tracking via the transformers API."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class FrameResult:
    frame_idx: int
    object_ids: list[int]
    masks: dict[int, np.ndarray]
    scores: dict[int, float]
    prompt_to_obj_ids: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class TrackingResult:
    prompts: list[str]
    frames: dict[int, FrameResult]
    height: int
    width: int

    def object_ids_for_prompt(self, prompt: str) -> set[int]:
        ids: set[int] = set()
        for fr in self.frames.values():
            ids.update(fr.prompt_to_obj_ids.get(prompt, []))
        return ids

    def mask_for_object(self, obj_id: int, frame_idx: int) -> np.ndarray:
        fr = self.frames.get(frame_idx)
        if fr is None or obj_id not in fr.masks:
            return np.zeros((self.height, self.width), dtype=bool)
        return fr.masks[obj_id]

    def union_mask_for_prompt(self, prompt: str, frame_idx: int) -> np.ndarray:
        fr = self.frames.get(frame_idx)
        merged = np.zeros((self.height, self.width), dtype=bool)
        if fr is None:
            return merged
        for obj_id in fr.prompt_to_obj_ids.get(prompt, []):
            if obj_id in fr.masks:
                merged |= fr.masks[obj_id]
        return merged


class Sam3VideoTracker:
    """Loads SAM 3 video model once; tracks prompted concepts through a clip."""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 model_id: str = "facebook/sam3"):
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        self.device = device
        self.dtype = dtype
        self.model = Sam3VideoModel.from_pretrained(model_id, dtype=dtype).to(device)
        self.processor = Sam3VideoProcessor.from_pretrained(model_id)

    def track(self, frames: list[np.ndarray], prompts: str | list[str],
              max_frames: int | None = None) -> TrackingResult:
        if isinstance(prompts, str):
            prompts = [prompts]

        height, width = frames[0].shape[:2]
        video = np.stack(frames)

        session = self.processor.init_video_session(
            video=video,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )
        self.processor.add_text_prompt(session, prompts)

        results: dict[int, FrameResult] = {}
        with torch.inference_mode():
            for out in self.model.propagate_in_video_iterator(
                inference_session=session,
                max_frame_num_to_track=max_frames,
                show_progress_bar=True,
            ):
                p = self.processor.postprocess_outputs(session, out)
                obj_ids = [int(x) for x in p["object_ids"].tolist()]
                masks = {
                    obj_ids[i]: p["masks"][i].cpu().numpy().astype(bool)
                    for i in range(len(obj_ids))
                }
                scores = {
                    obj_ids[i]: float(p["scores"][i]) for i in range(len(obj_ids))
                }
                results[out.frame_idx] = FrameResult(
                    frame_idx=out.frame_idx,
                    object_ids=obj_ids,
                    masks=masks,
                    scores=scores,
                    prompt_to_obj_ids={
                        k: [int(v) for v in vs]
                        for k, vs in p.get("prompt_to_obj_ids", {}).items()
                    },
                )

        return TrackingResult(prompts=prompts, frames=results,
                              height=height, width=width)