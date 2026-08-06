"""Stage 1 — video loading. Every method opens its own fresh capture and always releases it."""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from datatypes import VideoMetadata


@contextmanager
def _open_capture(path: Path) -> Iterator[cv2.VideoCapture]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    try:
        yield capture
    finally:
        capture.release()


class VideoSource:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def open_capture(self):
        """Context manager yielding a raw cv2.VideoCapture, released on exit.

        For callers (e.g. the tracker) that need to seek+read many frames from
        one open handle rather than paying an open/close cost per frame.
        """
        return _open_capture(self.path)

    def load_metadata(self) -> VideoMetadata:
        with _open_capture(self.path) as capture:
            fps = capture.get(cv2.CAP_PROP_FPS)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # cv2.CAP_PROP_FRAME_COUNT over-reports vs. actually-decodable frames on
            # some containers — count by reading, not by trusting the header.
            # grab() decodes exactly as read() does but skips the copy into a numpy
            # array, which is the expensive half when the pixels are discarded.
            read_count = 0
            while capture.grab():
                read_count += 1

        if read_count == 0:
            raise ValueError(f"no decodable frames in {self.path}")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(
                f"{self.path} reports an unusable frame rate ({fps!r}). Re-encode to a "
                "constant frame rate first, e.g. `ffmpeg -i in.mp4 -r 30 out.mp4` — "
                "guessing here would silently desync the matte from the source."
            )

        return VideoMetadata(path=self.path, fps=fps, width=width, height=height, frame_count=read_count)

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        with _open_capture(self.path) as capture:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield index, frame
                index += 1

    @staticmethod
    def iter_range(
        capture: cv2.VideoCapture,
        start_index: int,
        limit_index: int,
        reverse: bool,
        window: int = 64,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (index, frame) from `start_index` toward `limit_index`, inclusive.

        Forward is a plain sequential read after one seek.

        Backward is the reason this exists. Video cannot be decoded in reverse, so
        naively seeking per frame makes each `capture.set(CAP_PROP_POS_FRAMES, i)`
        decode from the preceding keyframe — around 125 hidden frame-decodes per
        requested frame at a typical 250-frame GOP. Reading a window forward into
        memory and then walking it backwards pays one seek per `window` frames
        instead of one per frame. The model sees identical pixels in identical
        order; only the disk access pattern changes.
        """
        if reverse:
            cursor = start_index
            while cursor >= limit_index:
                low = max(limit_index, cursor - window + 1)
                capture.set(cv2.CAP_PROP_POS_FRAMES, low)

                buffered: list[np.ndarray] = []
                for _ in range(cursor - low + 1):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    buffered.append(frame)
                if not buffered:
                    return

                for offset in range(len(buffered) - 1, -1, -1):
                    yield low + offset, buffered[offset]
                cursor = low - 1
        else:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
            index = start_index
            while index <= limit_index:
                ok, frame = capture.read()
                if not ok:
                    return
                yield index, frame
                index += 1

    def read_frame_at(self, frame_index: int) -> np.ndarray:
        with _open_capture(self.path) as capture:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
        if not ok:
            raise IndexError(f"could not read frame {frame_index} from {self.path}")
        return frame
