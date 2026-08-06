"""Stage 3 — SAM2 image segmentation + best-seed-frame selection.

Seeding tracking from frame 0 is unreliable — some frames give SAM2 a much
better lock on the subject than others. select_seed samples several
evenly-spaced candidate frames, segments each, and keeps whichever gives SAM2
the highest confidence as the global seed for tracking.
"""

from __future__ import annotations

import cv2
import gc
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

from detector import SubjectDetector
from device import autocast, enable_fast_matmul, pick_device
from datatypes import BoundingBox, SeedFrame, VideoMetadata
from video_source import VideoSource

# Stop sampling once a candidate is this good. SAM2's IoU head is well calibrated
# at the top of its range, so anything above this is an excellent lock and the
# remaining candidates cannot meaningfully improve on it — they can only cost
# two more model forwards each.
GOOD_ENOUGH_IOU = 0.97


class SeedSegmenter:
    MODEL_ID = "facebook/sam2.1-hiera-large"

    def __init__(
        self,
        detector: SubjectDetector,
        num_candidates: int = 12,
        device: str | None = None,
        model_id: str | None = None,
    ):
        self.detector = detector
        self.num_candidates = num_candidates
        self.device = device or pick_device()
        self.model_id = model_id or self.MODEL_ID
        enable_fast_matmul()

        self.processor = Sam2Processor.from_pretrained(self.model_id)
        self.model = Sam2Model.from_pretrained(self.model_id).to(self.device).eval()

    def segment(self, frame_rgb: Image.Image, box: BoundingBox) -> tuple[np.ndarray, float]:
        inputs = self.processor(
            images=frame_rgb, input_boxes=[[box.as_list()]], return_tensors="pt"
        ).to(self.device)
        # See tracker.py: no_grad rather than inference_mode, because
        # post_process_masks touches these tensors outside the block.
        with torch.no_grad(), autocast(self.device):
            outputs = self.model(**inputs, multimask_output=False)

        masks = self.processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])
        mask = (masks[0][0, 0].to(torch.uint8) * 255).cpu().numpy()
        iou_score = float(outputs.iou_scores.item())
        return mask, iou_score

    def _candidate_indices(self, total_frames: int) -> list[int]:
        if total_frames < 1:
            raise ValueError("cannot select a seed from a video with no frames")
        # num_candidates == 1 would divide by zero on the even-spacing formula;
        # a single sample belongs in the middle, not at frame 0, because the ends
        # of a clip are where subjects are most often part-way out of frame.
        if self.num_candidates < 2:
            return [total_frames // 2]
        return sorted(
            {int(i * (total_frames - 1) / (self.num_candidates - 1)) for i in range(self.num_candidates)}
        )

    def select_seed(self, video: VideoSource, metadata: VideoMetadata) -> SeedFrame:
        best: SeedFrame | None = None
        for idx in self._candidate_indices(metadata.frame_count):
            frame_bgr = video.read_frame_at(idx)
            frame_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            box = self.detector.detect(frame_rgb)
            if box is None:
                continue

            mask, iou_score = self.segment(frame_rgb, box)
            if best is None or iou_score > best.iou_score:
                best = SeedFrame(frame_index=idx, box=box, mask=mask, iou_score=iou_score)
            if best.iou_score >= GOOD_ENOUGH_IOU:
                break

        if best is None:
            raise ValueError(
                "No candidate frame produced a detection — widen num_candidates or check the prompt."
            )
        return best

    def release(self) -> None:
        """Drop the image model once seeding is done.

        This is a second full copy of sam2.1-hiera-large — the tracker loads the
        same checkpoint again as Sam2VideoModel — and it is never used after
        select_seed returns. Freeing it hands ~0.9 GB back to the tracking
        session, which is the component that actually runs out of VRAM.

        A long-lived worker that serves many jobs should NOT call this: reloading
        the weights per request costs more than the VRAM is worth.
        """
        if getattr(self, "model", None) is None:
            return
        self.model = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
