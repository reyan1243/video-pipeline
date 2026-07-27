"""Stage 3 — SAM2 image segmentation + best-seed-frame selection.

Seeding tracking from frame 0 is unreliable — some frames give SAM2 a much
better lock on the subject than others. select_seed samples several
evenly-spaced candidate frames, segments each, and keeps whichever gives SAM2
the highest confidence as the global seed for tracking.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

from detector import SubjectDetector
from device import pick_device
from datatypes import BoundingBox, SeedFrame, VideoMetadata
from video_source import VideoSource


class SeedSegmenter:
    MODEL_ID = "facebook/sam2.1-hiera-large"

    def __init__(self, detector: SubjectDetector, num_candidates: int = 12, device: str | None = None):
        self.detector = detector
        self.num_candidates = num_candidates
        self.device = device or pick_device()

        self.processor = Sam2Processor.from_pretrained(self.MODEL_ID)
        self.model = Sam2Model.from_pretrained(self.MODEL_ID).to(self.device).eval()

    def segment(self, frame_rgb: Image.Image, box: BoundingBox) -> tuple[np.ndarray, float]:
        inputs = self.processor(
            images=frame_rgb, input_boxes=[[box.as_list()]], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)

        masks = self.processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])
        mask = masks[0][0, 0].cpu().numpy().astype(np.uint8) * 255
        iou_score = outputs.iou_scores.item()
        return mask, iou_score

    def select_seed(self, video: VideoSource, metadata: VideoMetadata) -> SeedFrame:
        total_frames = metadata.frame_count
        candidate_indices = sorted(
            {int(i * (total_frames - 1) / (self.num_candidates - 1)) for i in range(self.num_candidates)}
        )

        best: SeedFrame | None = None
        for idx in candidate_indices:
            frame_bgr = video.read_frame_at(idx)
            frame_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            box = self.detector.detect(frame_rgb)
            if box is None:
                continue

            mask, iou_score = self.segment(frame_rgb, box)
            if best is None or iou_score > best.iou_score:
                best = SeedFrame(frame_index=idx, box=box, mask=mask, iou_score=iou_score)

        if best is None:
            raise ValueError(
                "No candidate frame produced a detection — widen num_candidates or check the prompt."
            )
        return best
