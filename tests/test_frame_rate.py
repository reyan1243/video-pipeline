"""Pins the exact-frame-rate probe.

OpenCV's CAP_PROP_FPS is a float and is measurably wrong on real files: on a
1080x1920 clip whose true rate is 30000/1001 (29.970030) it reported 29.991793.
Encoding the matte at that value makes it run 0.07% fast, which is a full frame
of drift against the source by the 46-second mark — the mask visibly leading the
subject. Nothing downstream can recover it, so the rate has to come from the
container, exactly, as a rational.
"""

import shutil
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from video_source import _probe_frame_rate

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not on PATH")
class TestProbeFrameRate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _make(self, rate: str) -> Path:
        path = self.tmp / f"clip_{rate.replace('/', '_')}.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
             "-i", f"testsrc2=s=64x64:r={rate}:d=1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
            check=True, capture_output=True,
        )
        return path

    def test_reads_ntsc_rate_exactly(self):
        # The case that actually bit us. As a float this is 29.9700299700...,
        # which is why it must stay a Fraction all the way to the encoder.
        self.assertEqual(_probe_frame_rate(self._make("30000/1001")), Fraction(30000, 1001))

    def test_reads_integer_rate(self):
        self.assertEqual(_probe_frame_rate(self._make("25")), Fraction(25, 1))

    def test_missing_file_returns_none_rather_than_raising(self):
        # Callers fall back to OpenCV's value; a probe failure must not abort a
        # job that would otherwise succeed.
        self.assertIsNone(_probe_frame_rate(self.tmp / "does-not-exist.mp4"))

    def test_non_video_returns_none(self):
        junk = self.tmp / "junk.mp4"
        junk.write_bytes(b"not a video")
        self.assertIsNone(_probe_frame_rate(junk))


if __name__ == "__main__":
    unittest.main()
