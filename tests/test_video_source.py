import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from video_source import VideoSource

FRAME_COUNT = 20
WIDTH, HEIGHT = 32, 24
FPS = 10.0


def _make_synthetic_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    try:
        for i in range(FRAME_COUNT):
            frame = np.full((HEIGHT, WIDTH, 3), fill_value=i % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class TestVideoSource(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.video_path = Path(self._tmpdir.name) / "synthetic.mp4"
        _make_synthetic_video(self.video_path)
        self.source = VideoSource(self.video_path)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_frame_count_matches_independent_manual_count(self):
        # Independent of VideoSource entirely: open our own capture and count by
        # reading, not by trusting CAP_PROP_FRAME_COUNT — pins the "empirical
        # count wins" invariant mechanically rather than by inspection.
        capture = cv2.VideoCapture(str(self.video_path))
        manual_count = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            manual_count += 1
        capture.release()

        metadata = self.source.load_metadata()
        self.assertEqual(metadata.frame_count, manual_count)
        self.assertEqual(metadata.width, WIDTH)
        self.assertEqual(metadata.height, HEIGHT)

    def test_read_frame_at_matches_iter_frames(self):
        frames = dict(self.source.iter_frames())
        frame_5 = self.source.read_frame_at(5)
        np.testing.assert_array_equal(frame_5, frames[5])

    def test_read_frame_at_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            self.source.read_frame_at(FRAME_COUNT + 100)


if __name__ == "__main__":
    unittest.main()
