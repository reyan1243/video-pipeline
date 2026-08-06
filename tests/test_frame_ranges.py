"""Pins VideoSource.iter_range — the reverse-reading optimisation.

Backward tracking used to seek per frame, and each `CAP_PROP_POS_FRAMES` seek
decodes from the preceding keyframe (~125 hidden frame-decodes per frame at a
typical 250-frame GOP). iter_range reads a window forward and walks it backwards
instead. The whole point is that the model sees *identical frames in identical
order* — only the disk access pattern changes — so that is exactly what these
tests assert, alongside the seek count that motivated the change.

Uses a stub capture rather than a real file: encoding a fixture here would depend
on the local OpenCV build having an mp4v encoder, which headless builds routinely
lack.
"""

import unittest

import numpy as np

from video_source import VideoSource


class FakeCapture:
    """Minimal cv2.VideoCapture stand-in. Frame N is an array full of value N."""

    def __init__(self, frame_count: int):
        self.frame_count = frame_count
        self.position = 0
        self.seek_count = 0

    def set(self, _prop, value):
        self.position = int(value)
        self.seek_count += 1
        return True

    def read(self):
        if self.position >= self.frame_count:
            return False, None
        frame = np.full((2, 2, 3), self.position % 256, dtype=np.uint8)
        self.position += 1
        return True, frame


def _collect(capture, start, limit, reverse, window=8):
    return [
        (idx, int(frame[0, 0, 0]))
        for idx, frame in VideoSource.iter_range(capture, start, limit, reverse=reverse, window=window)
    ]


class TestForwardRange(unittest.TestCase):
    def test_yields_ascending_indices_with_matching_frames(self):
        collected = _collect(FakeCapture(50), start=10, limit=19, reverse=False)
        self.assertEqual([idx for idx, _ in collected], list(range(10, 20)))
        # Frame content must match its index, or masks land on the wrong frames.
        self.assertTrue(all(idx == value for idx, value in collected))

    def test_forward_seeks_once(self):
        capture = FakeCapture(50)
        _collect(capture, start=10, limit=40, reverse=False)
        self.assertEqual(capture.seek_count, 1)

    def test_stops_at_end_of_stream(self):
        collected = _collect(FakeCapture(5), start=0, limit=99, reverse=False)
        self.assertEqual([idx for idx, _ in collected], [0, 1, 2, 3, 4])


class TestReverseRange(unittest.TestCase):
    def test_yields_descending_indices_with_matching_frames(self):
        collected = _collect(FakeCapture(50), start=30, limit=11, reverse=True)
        self.assertEqual([idx for idx, _ in collected], list(range(30, 10, -1)))
        self.assertTrue(all(idx == value for idx, value in collected))

    def test_reaches_frame_zero(self):
        collected = _collect(FakeCapture(50), start=5, limit=0, reverse=True)
        self.assertEqual([idx for idx, _ in collected], [5, 4, 3, 2, 1, 0])

    def test_single_frame_range(self):
        collected = _collect(FakeCapture(50), start=7, limit=7, reverse=True)
        self.assertEqual(collected, [(7, 7)])

    def test_seeks_once_per_window_not_once_per_frame(self):
        capture = FakeCapture(200)
        frames = _collect(capture, start=99, limit=0, reverse=True, window=8)
        self.assertEqual(len(frames), 100)
        # The old implementation issued one seek per frame. Ceil(100/8) == 13.
        self.assertEqual(capture.seek_count, 13)
        self.assertLess(capture.seek_count, len(frames))

    def test_window_larger_than_range_is_one_seek(self):
        capture = FakeCapture(200)
        _collect(capture, start=20, limit=10, reverse=True, window=64)
        self.assertEqual(capture.seek_count, 1)

    def test_ordering_is_identical_across_window_sizes(self):
        baseline = _collect(FakeCapture(200), start=80, limit=3, reverse=True, window=1)
        for window in (2, 7, 8, 64, 500):
            self.assertEqual(
                _collect(FakeCapture(200), start=80, limit=3, reverse=True, window=window),
                baseline,
                f"window={window} changed what the model would see",
            )


if __name__ == "__main__":
    unittest.main()
