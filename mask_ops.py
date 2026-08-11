"""Pure mask operations shared by the tracker and the post-tracking cleanup pass.

Kept as functions, not a class — there's no state to hold, just numpy transforms.

Two conventions worth knowing:

* During tracking, masks are strictly binary 0/255. `mask_iou` and `is_disrupted`
  are only ever called on those, so they threshold at `> 0` and stay exact.
* After tracking, `clean_masks` may feather the edge, so masks become a full
  0..255 ramp. Nothing downstream re-thresholds them — the ramp *is* the alpha.
"""

from __future__ import annotations

import cv2
import numpy as np

from datatypes import TrackerConfig

# Default close kernel. Was 45, which reliably bridged an arm to the torso and
# then let binary_fill_holes flood the enclosed triangle solid — swallowing
# exactly the negative space that a caption behind the subject shows through.
# 7 still closes tracking speckle; hole filling is unchanged and independent.
DEFAULT_CLOSE_KERNEL_SIZE = 7

# Gaussian sigma for the edge feather, in pixels. SAM2's decoder works at 256x256
# and everything above that is interpolation, so its "soft" logits are a
# confidence artefact whose width varies with certainty and with the frame's
# aspect ratio. A fixed blur applied at full resolution is isotropic, predictable
# and tunable — which the logits are not.
DEFAULT_FEATHER_SIGMA = 1.0

_ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a_bool = mask_a > 0
    b_bool = mask_b > 0
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a_bool, b_bool).sum() / union)


def fill_holes(solid: np.ndarray) -> np.ndarray:
    """Fill enclosed background regions. Equivalent to scipy's binary_fill_holes.

    Measured at 1080x1920: scipy 59.0 ms/frame vs 3.6 ms here — 16x, and the
    outputs are bit-identical. On a 1378-frame clip that is 81s versus 5s, which
    made mask cleanup as expensive as the entire GPU tracking pass.

    Works by flooding the background inward from outside the image and keeping
    whatever the flood could not reach. The 1px zero border is what makes the
    seed provably background — flooding from (0, 0) of the raw mask would fail
    the moment the subject touches a corner.
    """
    padded = cv2.copyMakeBorder(solid, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    flooded = padded.copy()
    cv2.floodFill(flooded, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flooded)
    return cv2.bitwise_or(padded, holes)[1:-1, 1:-1]


def downscale_mask(mask: np.ndarray, max_height: int) -> np.ndarray:
    """Cap mask height, preserving aspect ratio.

    Nearly free in quality terms: SAM2's decoder emits 256x256 logits and
    everything above that is interpolation, so a 1080x1920 mask carries no more
    real information than a 540x960 one. Costs scale with pixels, though —
    halving the height quarters cleanup, encode, transfer and RAM.
    """
    height, width = mask.shape[:2]
    if max_height <= 0 or height <= max_height:
        return mask
    scale = max_height / height
    target = (max(2, int(round(width * scale))), max_height)
    return cv2.resize(mask, target, interpolation=cv2.INTER_AREA)


def feather_mask(mask: np.ndarray, sigma: float = DEFAULT_FEATHER_SIGMA) -> np.ndarray:
    """Erode 1px, then blur — a soft edge biased *inward*.

    The bias is deliberate. Text tucked a pixel under a shoulder reads as
    correct; text creeping a pixel over it reads as a bug.
    """
    if sigma <= 0:
        return mask
    eroded = cv2.erode(mask, _ERODE_KERNEL)
    radius = max(3, int(round(sigma * 6)) | 1)  # odd kernel, ~3 sigma each side
    return cv2.GaussianBlur(eroded, (radius, radius), sigma)


def clean_mask(
    mask: np.ndarray,
    close_kernel_size: int = DEFAULT_CLOSE_KERNEL_SIZE,
    feather_sigma: float = DEFAULT_FEATHER_SIGMA,
) -> np.ndarray:
    """Morphological close + fill-holes, then an optional edge feather.

    The close bridges small local gaps (e.g. a waist notch during fast motion)
    that global disruption signals are too coarse to catch.
    """
    binary = mask > 127
    if not binary.any():
        return mask

    solid = binary.astype(np.uint8) * 255
    if close_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
        )
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel)
    return feather_mask(fill_holes(solid), feather_sigma)


def clean_masks(
    masks: dict[int, np.ndarray],
    close_kernel_size: int = DEFAULT_CLOSE_KERNEL_SIZE,
    feather_sigma: float = DEFAULT_FEATHER_SIGMA,
) -> dict[int, np.ndarray]:
    """Clean every mask **in place**.

    A dict comprehension here would hold the old and new dicts simultaneously,
    doubling peak RAM at the worst possible moment — 3.2 GB instead of 1.6 GB on
    a minute of 720p. Mutating in place keeps peak at one full set plus one frame.
    Returns the same dict so the call site reads unchanged.
    """
    for idx in list(masks):
        masks[idx] = clean_mask(masks[idx], close_kernel_size, feather_sigma)
    return masks


def temporal_median(masks: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """3-tap median across adjacent frames, in place.

    SAM2's per-frame masks jitter a pixel or two even on a static subject, and on
    an edge that a caption runs behind, that jitter is the most visible artefact
    there is — far more than per-frame IoU suggests. Costs a frame of lag on fast
    motion, which is why it is opt-in.
    """
    indices = sorted(masks)
    if len(indices) < 3:
        return masks

    previous = None
    current = masks[indices[0]]
    for position, idx in enumerate(indices):
        following = masks[indices[position + 1]] if position + 1 < len(indices) else current
        # `previous`/`current`/`following` are pre-smoothing originals: the
        # window is saved before each write, so smoothing never feeds itself.
        window = np.stack([current if previous is None else previous, current, following])
        masks[idx] = np.median(window, axis=0).astype(np.uint8)
        previous, current = current, following
    return masks


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
