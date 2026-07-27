"""Stage 1 — video loading. Every method opens its own fresh capture and always releases it."""

from __future__ import annotations

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
            read_count = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                read_count += 1

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

    def read_frame_at(self, frame_index: int) -> np.ndarray:
        with _open_capture(self.path) as capture:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
        if not ok:
            raise IndexError(f"could not read frame {frame_index} from {self.path}")
        return frame
