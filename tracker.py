"""Stage 4 — bidirectional SAM2 video tracking with auto re-seeding.

Two design decisions, both driven by real mask-degradation observed on fast
motion: (1) track in both directions from the single best seed frame instead
of only forward from frame 0, and (2) auto re-seed via a fresh detection the
moment a segment's quality craters, instead of letting a corrupted mask
propagate onward.

A third constraint arrived later: SAM2 keeps roughly 18 MB of VRAM per frame per
tracked object for the lifetime of a session and never evicts it, so a clip that
tracks cleanly end-to-end becomes one unbounded session and exhausts a 24 GB card
somewhere around 900-1000 frames. Segments are therefore also capped by *length*,
not only by disruption — and a length-capped split resumes from the last good
mask's own bounding box rather than a fresh text detection, so the subject's
identity cannot drift at the seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2VideoModel, Sam2VideoProcessor

from detector import SubjectDetector
from device import autocast, enable_fast_matmul, load_pretrained, pick_device
from mask_ops import downscale_mask, is_disrupted, mask_iou
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

# Ceiling on the reverse read-ahead buffer. The window is in frames, but frames
# are 6 MB at 1080p and 25 MB at 4K, so a fixed frame count would swing from
# 400 MB to 1.6 GB of transient RAM. Clamp by bytes instead.
REVERSE_BUFFER_BUDGET_BYTES = 256 * 1024**2


@dataclass
class _SegmentOutcome:
    masks: dict[int, np.ndarray]
    diagnostics: list[TrackingDiagnostic] = field(default_factory=list)
    # "limit"     — reached the requested end of the direction, nothing left to do
    # "disrupted" — mask quality collapsed; needs a fresh detection
    # "capped"    — hit max_segment_frames; continue from the last good mask
    # "decode_end"— the decoder stopped handing back frames
    reason: str = "limit"
    last_mask: np.ndarray | None = None


def _box_from_mask(mask: np.ndarray) -> BoundingBox | None:
    """Tight bounding box of a mask, used to resume a length-capped segment.

    Deliberately not a re-detection: Grounding DINO would re-answer "which object
    matches this text" from scratch and can legitimately pick a different person
    in a multi-subject frame. The previous mask's own box carries the identity
    forward with no ambiguity.
    """
    ys, xs = np.nonzero(mask > 127)
    if xs.size == 0:
        return None
    return BoundingBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


class MaskTracker:
    MODEL_ID = "facebook/sam2.1-hiera-large"

    def __init__(
        self,
        video: VideoSource,
        detector: SubjectDetector,
        config: TrackerConfig = TrackerConfig(),
        device: str | None = None,
        model_id: str | None = None,
        prefer_bfloat16: bool = False,
    ):
        self.video = video
        self.detector = detector
        self.config = config
        self.device = device or pick_device()
        self.model_id = model_id or self.MODEL_ID
        enable_fast_matmul()

        self.processor = Sam2VideoProcessor.from_pretrained(self.model_id)
        self.model = load_pretrained(Sam2VideoModel, self.model_id, self.device, prefer_bfloat16)

    def track(self, seed: SeedFrame, metadata: VideoMetadata) -> TrackingResult:
        # Peak VRAM is the constraint that decides max_segment_frames, so measure
        # it rather than trusting the arithmetic. Reset here so the figure covers
        # tracking only, not the weights loaded long before.
        self.peak_bytes = 0
        self.baseline_bytes = 0
        self.longest_segment = 0
        if self.device == "cuda":
            # reset_peak_memory_stats() rebases the peak to *current* allocation,
            # not zero, so the weights already resident are counted in every
            # subsequent reading. Record them separately or the per-frame figure
            # is inflated by a constant.
            torch.cuda.reset_peak_memory_stats()
            self.baseline_bytes = torch.cuda.memory_allocated()

        frame_bytes = max(1, metadata.width * metadata.height * 3)
        window = max(1, min(self.config.reverse_window, REVERSE_BUFFER_BUDGET_BYTES // frame_bytes))

        last_frame_idx = metadata.last_frame_index
        forward = self._run_direction(seed.frame_index, seed.box, "forward", last_frame_idx, window)
        backward = self._run_direction(seed.frame_index, seed.box, "backward", 0, window)

        # Backward wins on the shared seed frame. Both directions prompt from the
        # same box on the same frame, so the two masks agree; the ordering is
        # incidental and kept as-is to avoid changing established behaviour.
        if self.device == "cuda":
            self.peak_bytes = torch.cuda.max_memory_allocated()

        masks = {**forward.masks, **backward.masks}
        diagnostics = forward.diagnostics + backward.diagnostics
        return TrackingResult(masks=masks, diagnostics=diagnostics)

    def _run_direction(
        self, start_frame: int, start_box: BoundingBox, direction: str, limit: int, window: int
    ) -> TrackingResult:
        masks: dict[int, np.ndarray] = {}
        diagnostics: list[TrackingDiagnostic] = []
        current_frame, current_box = start_frame, start_box

        for _ in range(self.config.max_segments_per_direction):
            segment = self._track_segment(current_frame, current_box, direction, limit, window)
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

            if segment.reason == "capped" and segment.last_mask is not None:
                # Purely a VRAM split, not a tracking failure — carry the object
                # forward by its own geometry and skip the detector entirely.
                carried_box = _box_from_mask(segment.last_mask)
                if carried_box is not None:
                    current_frame, current_box = resume_frame, carried_box
                    continue

            resume_frame_bgr = self.video.read_frame_at(resume_frame)
            resume_frame_rgb = Image.fromarray(cv2.cvtColor(resume_frame_bgr, cv2.COLOR_BGR2RGB))
            new_box = self.detector.detect(resume_frame_rgb)
            if new_box is None:
                break
            current_frame, current_box = resume_frame, new_box

        return TrackingResult(masks=masks, diagnostics=diagnostics)

    def _new_session(self):
        kwargs = {"inference_device": self.device}
        if self.config.cpu_offload:
            # Moves the per-frame state that grows linearly off the GPU. Costs
            # roughly 22% wall-clock; removes the frame ceiling outright.
            kwargs["inference_state_device"] = "cpu"
            kwargs["video_storage_device"] = "cpu"
        return self.processor.init_video_session(**kwargs)

    def _track_segment(
        self,
        start_frame_idx: int,
        seed_box: BoundingBox,
        direction: str,
        frame_limit: int,
        window: int,
    ) -> _SegmentOutcome:
        reverse = direction == "backward"
        session = self._new_session()

        masks: dict[int, np.ndarray] = {}
        diagnostics: list[TrackingDiagnostic] = []
        prev_mask, prev_area = None, None
        session_local_idx = 0
        reason = "decode_end"

        with self.video.open_capture() as capture:
            frames = VideoSource.iter_range(
                capture, start_frame_idx, frame_limit, reverse=reverse, window=window
            )
            for real_idx, frame_bgr in frames:
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

                # no_grad, not inference_mode: post_process_masks runs on
                # output.pred_masks *outside* this block, and inference_mode marks
                # its outputs as inference tensors that raise on any in-place
                # update outside the block. no_grad has the same memory benefit
                # here without that failure mode.
                with torch.no_grad(), autocast(self.device):
                    output = self.model(
                        inference_session=session, frame=frame_inputs.pixel_values[0], reverse=reverse
                    )
                post_processed = self.processor.post_process_masks(
                    [output.pred_masks], original_sizes=frame_inputs.original_sizes
                )[0]
                # Scale on the GPU, then transfer. Out-of-place on purpose: `*` is
                # safe whatever dtype post_process_masks returns, whereas `mul_`
                # would mutate the tensor in place if it ever stops being bool.
                mask = (post_processed[0, 0].to(torch.uint8) * 255).cpu().numpy()
                # Downscale immediately, so every downstream cost — the IoU
                # check, the stored dict, cleanup and encoding — is paid at the
                # reduced size rather than at source resolution.
                mask = downscale_mask(mask, self.config.max_mask_height)
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
                        reason = "disrupted"
                        break

                masks[real_idx] = mask
                prev_mask, prev_area = mask, area
                session_local_idx += 1

                if real_idx == frame_limit:
                    reason = "limit"
                    break
                if session_local_idx >= self.config.max_segment_frames:
                    reason = "capped"
                    break

        self.longest_segment = max(getattr(self, "longest_segment", 0), len(masks))
        return _SegmentOutcome(
            masks=masks, diagnostics=diagnostics, reason=reason, last_mask=prev_mask
        )
