"""Stage 4 — bidirectional SAM2 video tracking with auto re-seeding.

Two design decisions, both driven by real mask-degradation observed on fast
motion: (1) track in both directions from the single best seed frame instead
of only forward from frame 0, and (2) auto re-seed via a fresh detection the
moment a segment's quality craters, instead of letting a corrupted mask
propagate onward.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2VideoModel, Sam2VideoProcessor

from detector import SubjectDetector
from device import pick_device
from mask_ops import is_disrupted, mask_iou
from datatypes import (
    BoundingBox,
    SeedFrame,
    TrackerConfig,
    TrackingDiagnostic,
    TrackingResult,
    VideoMetadata,
)
from video_source import VideoSource

TRACK_OBJ_ID = 1


class MaskTracker:
    MODEL_ID = "facebook/sam2.1-hiera-large"

    def __init__(
        self,
        video: VideoSource,
        detector: SubjectDetector,
        config: TrackerConfig = TrackerConfig(),
        device: str | None = None,
    ):
        self.video = video
        self.detector = detector
        self.config = config
        self.device = device or pick_device()

        self.processor = Sam2VideoProcessor.from_pretrained(self.MODEL_ID)
        self.model = Sam2VideoModel.from_pretrained(self.MODEL_ID).to(self.device).eval()

    def track(self, seed: SeedFrame, metadata: VideoMetadata) -> TrackingResult:
        last_frame_idx = metadata.last_frame_index
        forward = self._run_direction(seed.frame_index, seed.box, "forward", last_frame_idx)
        backward = self._run_direction(seed.frame_index, seed.box, "backward", 0)

        masks = {**forward.masks, **backward.masks}
        diagnostics = forward.diagnostics + backward.diagnostics
        return TrackingResult(masks=masks, diagnostics=diagnostics)

    def _run_direction(
        self, start_frame: int, start_box: BoundingBox, direction: str, limit: int
    ) -> TrackingResult:
        masks: dict[int, np.ndarray] = {}
        diagnostics: list[TrackingDiagnostic] = []
        current_frame, current_box = start_frame, start_box

        for _ in range(self.config.max_segments_per_direction):
            segment = self._track_segment(current_frame, current_box, direction, limit)
            masks.update(segment.masks)
            diagnostics.extend(segment.diagnostics)
            if not segment.masks:
                break

            reached_limit = (
                (min(segment.masks) <= limit) if direction == "backward" else (max(segment.masks) >= limit)
            )
            if reached_limit:
                break

            resume_frame = (min(segment.masks) - 1) if direction == "backward" else (max(segment.masks) + 1)
            resume_frame_bgr = self.video.read_frame_at(resume_frame)
            resume_frame_rgb = Image.fromarray(cv2.cvtColor(resume_frame_bgr, cv2.COLOR_BGR2RGB))
            new_box = self.detector.detect(resume_frame_rgb)
            if new_box is None:
                break
            current_frame, current_box = resume_frame, new_box

        return TrackingResult(masks=masks, diagnostics=diagnostics)

    def _track_segment(
        self, start_frame_idx: int, seed_box: BoundingBox, direction: str, frame_limit: int
    ) -> TrackingResult:
        reverse = direction == "backward"
        step = -1 if reverse else 1
        session = self.processor.init_video_session(inference_device=self.device)

        masks: dict[int, np.ndarray] = {}
        diagnostics: list[TrackingDiagnostic] = []
        prev_mask, prev_area = None, None
        real_idx = start_frame_idx
        session_local_idx = 0

        with self.video.open_capture() as capture:
            while (real_idx >= frame_limit) if reverse else (real_idx <= frame_limit):
                # Forward tracking only needs an explicit seek once, to reach
                # start_frame_idx — capture.read() already advances
                # sequentially after that. Backward tracking can't move
                # backward via plain read(), so every frame needs a real seek.
                if reverse or session_local_idx == 0:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, real_idx)
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_inputs = self.processor(images=frame_rgb, device=self.device, return_tensors="pt")

                if session_local_idx == 0:
                    self.processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=0,
                        obj_ids=TRACK_OBJ_ID,
                        input_boxes=[[seed_box.as_list()]],
                        original_size=frame_inputs.original_sizes[0],
                    )

                with torch.no_grad():
                    output = self.model(
                        inference_session=session, frame=frame_inputs.pixel_values[0], reverse=reverse
                    )
                post_processed = self.processor.post_process_masks(
                    [output.pred_masks], original_sizes=frame_inputs.original_sizes
                )[0]
                mask = post_processed[0, 0].cpu().numpy().astype(np.uint8) * 255
                area = int(np.count_nonzero(mask))
                object_score = (
                    float(output.object_score_logits.item())
                    if getattr(output, "object_score_logits", None) is not None
                    else None
                )

                if prev_mask is not None:
                    iou = mask_iou(prev_mask, mask)
                    area_ratio = area / prev_area if prev_area else 1.0
                    diagnostics.append(
                        TrackingDiagnostic(
                            frame_index=real_idx,
                            direction=direction,
                            iou=iou,
                            area_ratio=area_ratio,
                            object_score=object_score,
                        )
                    )
                    if is_disrupted(iou, area_ratio, object_score, self.config):
                        break

                masks[real_idx] = mask
                prev_mask, prev_area = mask, area
                real_idx += step
                session_local_idx += 1

        return TrackingResult(masks=masks, diagnostics=diagnostics)
