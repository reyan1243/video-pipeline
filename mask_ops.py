"""Pure mask operations shared by the tracker and the post-tracking cleanup pass.

Kept as functions, not a class — there's no state to hold, just numpy transforms.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from datatypes import TrackerConfig


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a_bool = mask_a > 0
    b_bool = mask_b > 0
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 1.0
    return np.logical_and(a_bool, b_bool).sum() / union


def clean_mask(mask: np.ndarray, close_kernel_size: int = 45) -> np.ndarray:
    """Morphological close + fill-holes — bridges small local gaps (e.g. a waist
    notch during fast motion) that global disruption signals are too coarse to catch.
    """
    binary = mask > 127
    if int(binary.sum()) == 0:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    closed = cv2.morphologyEx(binary.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    filled = ndimage.binary_fill_holes(closed > 0).astype(np.uint8) * 255
    return filled


def clean_masks(masks: dict[int, np.ndarray], close_kernel_size: int = 45) -> dict[int, np.ndarray]:
    return {idx: clean_mask(mask, close_kernel_size) for idx, mask in masks.items()}


def is_disrupted(
    iou: float,
    area_ratio: float,
    object_score: float | None,
    config: TrackerConfig,
) -> bool:
    if iou < config.iou_drop_threshold:
        return True
    low, high = config.area_ratio_bounds
    if not (low < area_ratio < high):
        return True
    if object_score is not None and object_score < config.object_score_threshold:
        return True
    return False
