"""Shared data types passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_list(cls, box: list[float]) -> "BoundingBox":
        x0, y0, x1, y1 = box
        return cls(x0, y0, x1, y1)


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int  # empirically-read count — never cv2.CAP_PROP_FRAME_COUNT, which over-reports

    @property
    def last_frame_index(self) -> int:
        return self.frame_count - 1


@dataclass(frozen=True)
class SeedFrame:
    frame_index: int
    box: BoundingBox
    mask: np.ndarray
    iou_score: float


@dataclass(frozen=True)
class TrackingDiagnostic:
    frame_index: int
    direction: Literal["forward", "backward"]
    iou: float
    area_ratio: float
    object_score: float | None


@dataclass
class TrackingResult:
    masks: dict[int, np.ndarray]  # frame_idx -> uint8 0/255 mask
    diagnostics: list[TrackingDiagnostic] = field(default_factory=list)

    def frame_indices(self) -> list[int]:
        return sorted(self.masks)

    def coverage(self, total_frames: int) -> float:
        if total_frames == 0:
            return 0.0
        return len(self.masks) / total_frames


@dataclass(frozen=True)
class TrackerConfig:
    object_score_threshold: float = 0.0
    iou_drop_threshold: float = 0.4
    area_ratio_bounds: tuple[float, float] = (0.3, 1 / 0.3)
    # Raised from 8: a busy clip exhausts a small budget and the direction stops
    # early, which the exporter then reports as incomplete coverage.
    max_segments_per_direction: int = 20
    # SAM2 retains ~18 MB of VRAM per frame per tracked object for the lifetime of
    # a session and never evicts it, so an undisrupted run OOMs a 24 GB card at
    # roughly 900-1000 frames. Splitting into bounded sessions caps peak VRAM at
    # `max_segment_frames` regardless of clip length; each split re-seeds from the
    # last good mask, so masks are unaffected.
    max_segment_frames: int = 600
    # Move the per-frame session state to host RAM instead. Removes the ceiling
    # entirely at roughly +22% wall-clock. Prefer max_segment_frames.
    cpu_offload: bool = False
    # Read this many frames per seek when tracking backwards. Video cannot be
    # decoded in reverse, so each seek costs a decode from the preceding
    # keyframe; batching amortises that over a window instead of paying it per
    # frame. Larger = fewer seeks, more transient RAM.
    reverse_window: int = 64


@dataclass(frozen=True)
class ExportResult:
    format: Literal["webm_alpha", "fill_matte", "matte"]
    frame_count: int
    fps: float
    # Actual encoded size. yuv420p requires even dimensions, so an odd-sized
    # source is padded by one pixel on the right/bottom; the origin is preserved
    # so alignment holds, but the consumer should scale using these, not the
    # source dimensions.
    width: int = 0
    height: int = 0
    # None for "matte", which deliberately emits only the silhouette — the
    # consumer already has the source video, so shipping a cut-out copy of it
    # and a byte-identical "background" is pure egress.
    person_path: Path | None = None
    background_path: Path | None = None
    matte_path: Path | None = None
