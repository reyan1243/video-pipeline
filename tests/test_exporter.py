import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from exporter import LayerExporter
from datatypes import TrackingResult, VideoMetadata
from video_source import VideoSource

FRAME_COUNT = 10
WIDTH, HEIGHT = 64, 64
FPS = 10.0


def _make_synthetic_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    try:
        for i in range(FRAME_COUNT):
            frame = np.full((HEIGHT, WIDTH, 3), fill_value=(i * 20) % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _centered_square_masks() -> dict[int, np.ndarray]:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[20:44, 20:44] = 255  # centered 24x24 square
    return {i: mask.copy() for i in range(FRAME_COUNT)}


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not on PATH")
class TestLayerExporter(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmpdir.name)
        self.video_path = tmp_path / "synthetic.mp4"
        _make_synthetic_video(self.video_path)
        self.video = VideoSource(self.video_path)
        self.metadata = VideoMetadata(
            path=self.video_path, fps=FPS, width=WIDTH, height=HEIGHT, frame_count=FRAME_COUNT
        )
        self.output_dir = tmp_path / "out"
        self.exporter = LayerExporter(self.video, self.output_dir)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_background_is_byte_identical_copy(self):
        tracking = TrackingResult(masks=_centered_square_masks())
        result = self.exporter.export(tracking, self.metadata, person_format="fill_matte")
        self.assertEqual(result.background_path.read_bytes(), self.video_path.read_bytes())

    def test_missing_coverage_raises(self):
        masks = _centered_square_masks()
        del masks[3]
        tracking = TrackingResult(masks=masks)
        with self.assertRaises(ValueError):
            self.exporter.export(tracking, self.metadata, person_format="fill_matte")

    def test_fill_matte_matches_mask(self):
        tracking = TrackingResult(masks=_centered_square_masks())
        result = self.exporter.export(tracking, self.metadata, person_format="fill_matte")

        matte_capture = cv2.VideoCapture(str(result.matte_path))
        ok, matte_frame = matte_capture.read()
        matte_capture.release()
        self.assertTrue(ok)
        matte_gray = matte_frame[:, :, 0] if matte_frame.ndim == 3 else matte_frame
        self.assertGreater(int(matte_gray[30, 30]), 200)  # inside square
        self.assertLess(int(matte_gray[5, 5]), 50)  # outside square

        fill_capture = cv2.VideoCapture(str(result.person_path))
        ok, fill_frame = fill_capture.read()
        fill_capture.release()
        self.assertTrue(ok)
        self.assertTrue(np.all(fill_frame[5, 5] == 0))  # outside square is black

    def test_webm_alpha_has_transparency_matching_mask(self):
        tracking = TrackingResult(masks=_centered_square_masks())
        result = self.exporter.export(tracking, self.metadata, person_format="webm_alpha")
        self.assertIsNone(result.matte_path)

        decode_cmd = [
            # -c:v libvpx-vp9 forces the real libvpx decoder, which correctly
            # reads the WebM alpha side-block. ffmpeg's built-in "vp9" decoder
            # silently ignores it and reports everything as opaque — the same
            # trap awaits any downstream consumer of person.webm.
            "ffmpeg", "-c:v", "libvpx-vp9", "-i", str(result.person_path),
            "-pix_fmt", "bgra", "-f", "rawvideo", "-",
        ]
        raw = subprocess.run(decode_cmd, capture_output=True, check=True).stdout
        frame_bytes = WIDTH * HEIGHT * 4
        first_frame = np.frombuffer(raw[:frame_bytes], dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)

        self.assertGreater(int(first_frame[30, 30, 3]), 200)  # inside square: opaque
        self.assertLess(int(first_frame[5, 5, 3]), 50)  # outside square: transparent


if __name__ == "__main__":
    unittest.main()
