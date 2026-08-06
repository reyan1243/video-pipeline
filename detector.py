"""Stage 2 — open-vocabulary subject detection via Grounding DINO.

Picks the largest-AREA detection, not the highest-confidence one: for a prompt
like "guitarist.", the highest-confidence box often matches just the
instrument, not the full person.
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from device import autocast, enable_fast_matmul, pick_device
from datatypes import BoundingBox


def _normalize_prompt(prompt: str) -> str:
    prompt = prompt.strip().lower()
    if not prompt.endswith("."):
        prompt = f"{prompt}."
    return prompt


def _largest(boxes: list[BoundingBox]) -> BoundingBox:
    return max(boxes, key=lambda box: box.area)


class SubjectDetector:
    MODEL_ID = "IDEA-Research/grounding-dino-base"

    def __init__(
        self,
        prompt: str,
        box_threshold: float = 0.2,
        text_threshold: float = 0.2,
        device: str | None = None,
    ):
        self.prompt = _normalize_prompt(prompt)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device or pick_device()
        enable_fast_matmul()

        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.MODEL_ID).to(self.device).eval()

    def detect(self, frame_rgb: Image.Image) -> BoundingBox | None:
        inputs = self.processor(images=frame_rgb, text=self.prompt, return_tensors="pt").to(self.device)
        # See tracker.py: no_grad rather than inference_mode, because
        # post_process_grounded_object_detection touches these tensors below.
        with torch.no_grad(), autocast(self.device):
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[frame_rgb.size[::-1]],
            text_labels=[[self.prompt]],
        )[0]

        boxes = [BoundingBox.from_list(box) for box in results["boxes"].tolist()]
        if not boxes:
            return None
        return _largest(boxes)
