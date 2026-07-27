"""Final stage: exports two stackable output videos (person layer + background
layer). Compositing anything on top of them (text, captions, etc.) is out of
scope here — this stage ends at handing back a usable person cutout and the
background plate.

Default is "fill_matte", not "webm_alpha", for a real reason: ffmpeg ships two
separate VP9 decoders, and the one every plain `ffmpeg -i file.webm` (or any
wrapper shelling out to the system ffmpeg) reaches for by default — the
native "vp9" decoder — does not implement WebM's alpha side-channel at all.
It silently reports every frame as fully opaque instead of erroring. Only the
"libvpx-vp9" decoder wrapper reads it correctly, which requires the consumer
to explicitly pass `-c:v libvpx-vp9` (or equivalent) — not something fixable
from the encode side, since decoder choice happens entirely in the
consumer's code, not the bitstream itself. "fill_matte" sidesteps this:
plain H.264 RGB + a grayscale matte, decodable by anything, with nothing to
silently drop.
Only pass person_format="webm_alpha" if the downstream consumer has
confirmed it decodes VP9 alpha correctly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Literal

import cv2

from datatypes import ExportResult, TrackingResult, VideoMetadata
from video_source import VideoSource

PersonFormat = Literal["webm_alpha", "fill_matte"]


class LayerExporter:
    def __init__(self, video: VideoSource, output_dir: str | Path, ffmpeg_bin: str = "ffmpeg"):
        self.video = video
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_bin = ffmpeg_bin
        if shutil.which(ffmpeg_bin) is None:
            raise FileNotFoundError(f"{ffmpeg_bin!r} not found on PATH")

    def export(
        self,
        tracking: TrackingResult,
        metadata: VideoMetadata,
        person_format: PersonFormat = "fill_matte",
    ) -> ExportResult:
        missing = sorted(set(range(metadata.frame_count)) - set(tracking.masks))
        if missing:
            raise ValueError(f"tracking coverage incomplete — missing {len(missing)} frames: {missing[:10]}...")

        background_path = self.output_dir / f"background{self.video.path.suffix}"
        shutil.copy2(self.video.path, background_path)

        if person_format == "webm_alpha":
            person_path = self._write_webm_alpha(tracking, metadata)
            matte_path = None
        else:
            person_path, matte_path = self._write_fill_matte(tracking, metadata)

        return ExportResult(
            format=person_format,
            person_path=person_path,
            matte_path=matte_path,
            background_path=background_path,
            frame_count=metadata.frame_count,
            fps=metadata.fps,
        )

    def _write_webm_alpha(self, tracking: TrackingResult, metadata: VideoMetadata) -> Path:
        person_path = self.output_dir / "person.webm"
        command = [
            self.ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-pixel_format", "bgra",
            "-video_size", f"{metadata.width}x{metadata.height}",
            "-framerate", str(metadata.fps),
            "-i", "-",
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            # VP9 alt-ref frames corrupt the alpha plane if left on — must be disabled.
            "-auto-alt-ref", "0",
            str(person_path),
        ]
        with subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        ) as process:
            assert process.stdin is not None
            for frame_idx, frame_bgr in self.video.iter_frames():
                bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
                bgra[:, :, 3] = tracking.masks[frame_idx]
                process.stdin.write(bgra.tobytes())
            process.stdin.close()

            return_code = process.wait()
            stderr = process.stderr.read().decode() if process.stderr else ""
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed (exit {return_code}): {stderr}")
        return person_path

    def _write_fill_matte(self, tracking: TrackingResult, metadata: VideoMetadata) -> tuple[Path, Path]:
        fill_path = self.output_dir / "person_fill.mp4"
        matte_path = self.output_dir / "person_matte.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        size = (metadata.width, metadata.height)

        fill_writer = cv2.VideoWriter(str(fill_path), fourcc, metadata.fps, size)
        matte_writer = cv2.VideoWriter(str(matte_path), fourcc, metadata.fps, size, isColor=False)
        try:
            for frame_idx, frame_bgr in self.video.iter_frames():
                mask = tracking.masks[frame_idx]
                fill = frame_bgr.copy()
                fill[mask == 0] = 0
                fill_writer.write(fill)
                matte_writer.write(mask)
        finally:
            fill_writer.release()
            matte_writer.release()

        return fill_path, matte_path
