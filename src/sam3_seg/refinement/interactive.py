"""Click-based interactive refinement using SAM 3's PVS video tracker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RefinementResult:
    masks: dict[int, np.ndarray]
    clicked_frame: int
    points: list[tuple[int, int]]
    labels: list[int]


class Sam3ClickRefiner:
    """Tracks a single object from point clicks and propagates across a clip."""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 model_id: str = "facebook/sam3"):
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        self.device = device
        self.dtype = dtype
        self.model = Sam3TrackerVideoModel.from_pretrained(model_id, dtype=dtype).to(device)
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(model_id)

    def refine(self, frames: list[np.ndarray], clicked_frame: int,
               points: list[tuple[int, int]], labels: list[int],
               obj_id: int = 1) -> RefinementResult:
        height, width = frames[0].shape[:2]
        video = np.stack(frames)

        session = self.processor.init_video_session(
            video=video,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )

        input_points = [[[list(p) for p in points]]]
        input_labels = [[list(labels)]]

        self.processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=clicked_frame,
            obj_ids=obj_id,
            input_points=input_points,
            input_labels=input_labels,
            original_size=(height, width),
        )


        masks: dict[int, np.ndarray] = {}
        with torch.inference_mode():
            for out in self.model.propagate_in_video_iterator(
                session, start_frame_idx=clicked_frame
            ):
                m = self.processor.post_process_masks(
                    [out.pred_masks],
                    original_sizes=[[height, width]],
                    binarize=True,
                )[0]
                masks[out.frame_idx] = m[0].cpu().numpy().astype(bool).squeeze()

        return RefinementResult(masks=masks, clicked_frame=clicked_frame,
                                points=points, labels=labels)