"""End-to-end checks on the `matte` export path.

These run without torch and without a real source video: matte export never
decodes the input (it walks `tracking.masks` directly), so the only external
dependency is ffmpeg. Frames are decoded back out with ffmpeg rather than
OpenCV so the assertions do not depend on the local OpenCV build's codec
support, which headless wheels routinely lack.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from exporter import LayerExporter
from datatypes import TrackingResult, VideoMetadata
from video_source import VideoSource

FPS = 10.0
FRAME_COUNT = 12

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _metadata(path: Path, width: int, height: int) -> VideoMetadata:
    return VideoMetadata(
        path=path, fps=FPS, width=width, height=height, frame_count=FRAME_COUNT
    )


def _masks(width: int, height: int, feathered: bool = False) -> dict[int, np.ndarray]:
    masks = {}
    for idx in range(FRAME_COUNT):
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 255
        if feathered:
            # One ramp column, standing in for the gaussian edge feather.
            mask[height // 4 : 3 * height // 4, width // 4 - 1] = 128
        masks[idx] = mask
    return masks


def _decode_gray(path: Path, width: int, height: int) -> np.ndarray:
    """Decode frame 0 back to a raw gray array via ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.uint8)[: width * height].reshape(height, width)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg not on PATH")
class TestMatteExport(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.source = self.tmp / "input.mp4"
        self.source.write_bytes(b"")  # never opened by the matte path
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _export(self, width, height, masks=None):
        exporter = LayerExporter(VideoSource(self.source), output_dir=self.tmp / "out")
        tracking = TrackingResult(masks=masks or _masks(width, height))
        return exporter.export(tracking, _metadata(self.source, width, height), person_format="matte")

    def test_writes_a_non_empty_matte(self):
        result = self._export(64, 64)
        self.assertEqual(result.format, "matte")
        self.assertIsNotNone(result.matte_path)
        self.assertGreater(result.matte_path.stat().st_size, 0)

    def test_emits_only_the_matte(self):
        # The whole point of this format: no cut-out copy of the source, no
        # byte-identical "background".
        result = self._export(64, 64)
        self.assertIsNone(result.person_path)
        self.assertIsNone(result.background_path)
        self.assertFalse((self.tmp / "out" / "person_fill.mp4").exists())

    def test_silhouette_survives_the_encode(self):
        result = self._export(64, 64)
        decoded = _decode_gray(result.matte_path, 64, 64)
        self.assertGreater(int(decoded[32, 32]), 200)  # inside the subject
        self.assertLess(int(decoded[2, 2]), 55)  # outside it

    def test_soft_edge_survives_the_encode(self):
        # If the encoder crushed a feathered matte back to binary, the feather
        # work upstream would be silently wasted.
        result = self._export(64, 64, masks=_masks(64, 64, feathered=True))
        decoded = _decode_gray(result.matte_path, 64, 64)
        intermediate = np.count_nonzero((decoded > 40) & (decoded < 215))
        self.assertGreater(intermediate, 0)

    def test_odd_dimensions_are_padded_and_reported(self):
        # yuv420p requires even dimensions; padding keeps the origin so
        # alignment holds, and the result reports what was actually written.
        result = self._export(63, 65)
        self.assertEqual((result.width, result.height), (64, 66))

    def test_even_dimensions_are_untouched(self):
        result = self._export(64, 64)
        self.assertEqual((result.width, result.height), (64, 64))

    def test_incomplete_coverage_is_rejected_with_an_actionable_message(self):
        masks = _masks(64, 64)
        del masks[5]
        exporter = LayerExporter(VideoSource(self.source), output_dir=self.tmp / "out")
        with self.assertRaises(ValueError) as caught:
            exporter.export(
                TrackingResult(masks=masks), _metadata(self.source, 64, 64), person_format="matte"
            )
        message = str(caught.exception)
        self.assertIn("91.7%", message)  # 11/12
        self.assertIn("frame 5", message)

    def test_unknown_format_is_rejected(self):
        exporter = LayerExporter(VideoSource(self.source), output_dir=self.tmp / "out")
        with self.assertRaises(ValueError):
            exporter.export(
                TrackingResult(masks=_masks(64, 64)),
                _metadata(self.source, 64, 64),
                person_format="rgba",
            )


if __name__ == "__main__":
    unittest.main()
