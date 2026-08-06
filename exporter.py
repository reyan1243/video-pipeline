"""Final stage: writes the subject matte, and optionally stackable person /
background layers. Compositing anything on top of them (text, captions, etc.) is
out of scope here — this stage ends at handing back a usable cutout.

Three formats, in order of preference:

"matte" (default)
    A single grayscale H.264 video: white subject, black background. This is all
    a compositor actually needs, because the subject's colour *is* the source
    video the consumer already has. Emitting a cut-out copy of that video plus a
    byte-identical "background" is pure egress — roughly 2.5 MB/min for the matte
    versus two full videos. It is also the only format with no dark rim (below).

"fill_matte"
    The original layout: RGB with non-subject pixels zeroed, plus a separate
    matte, plus a copy of the input. Kept for existing consumers. Note the rim:
    zeroing creates a maximal-contrast black step exactly on the silhouette, and
    lossy 4:2:0 encoding then bleeds that black 1-3 px *into* the subject via
    chroma subsampling and DCT ringing. Raising the encoder from mp4v (MPEG-4
    Part 2, no quality control) to H.264 CRF 18 shrinks it substantially, but it
    is inherent to the format. Use "matte" if the rim matters.

"webm_alpha"
    Single VP9 file with a real alpha channel. Correct for a browser or Electron
    consumer, which decodes it properly. Not for anything shelling out to ffmpeg:
    ffmpeg ships two VP9 decoders and the default native one does not implement
    WebM's alpha side-channel at all — it silently reports every frame as fully
    opaque instead of erroring. Only `-c:v libvpx-vp9` reads it correctly, and
    that choice lives in the consumer's code, not in the bitstream.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np

from datatypes import ExportResult, TrackingResult, VideoMetadata
from video_source import VideoSource

PersonFormat = Literal["webm_alpha", "fill_matte", "matte"]

# CRF 18 with `-tune grain` measured 3.2 dB better than plain CRF 18 on a
# hard-edged matte — and better than CRF 15 — for +27% size. `grain` disables the
# psychovisual and deblocking behaviour that smears sharp binary edges, which is
# exactly the failure mode a silhouette suffers from.
_MATTE_ENCODE_ARGS = (
    "-c:v", "libx264",
    "-crf", "18",
    "-tune", "grain",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
)

_FILL_ENCODE_ARGS = (
    "-c:v", "libx264",
    "-crf", "18",
    "-preset", "medium",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
)

# yuv420p needs even dimensions. Pad rather than scale so the top-left origin —
# and therefore alignment with the source — is preserved exactly. ExportResult
# carries the resulting size so the consumer never has to guess.
_EVEN_PAD = ("-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2")


class _RawVideoEncoder:
    """Pipe raw frames into ffmpeg, with the stderr pipe actually drained.

    Draining matters: ffmpeg writes progress to stderr continuously, and if that
    pipe fills (64 KB) ffmpeg blocks writing to it, therefore stops reading stdin,
    therefore the parent blocks in write() — and the wait() that would have read
    stderr is never reached. That is a permanent deadlock on any clip long enough
    to produce 64 KB of log, which the previous implementation was exposed to.
    """

    def __init__(
        self,
        ffmpeg_bin: str,
        path: Path,
        width: int,
        height: int,
        fps: float,
        input_pixel_format: str,
        output_args: Sequence[str],
    ):
        self.path = path
        self.command = [
            ffmpeg_bin,
            "-y",
            "-loglevel", "error",
            "-nostats",
            "-f", "rawvideo",
            "-pixel_format", input_pixel_format,
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", "-",
            *output_args,
            str(path),
        ]
        self._process: subprocess.Popen | None = None
        self._stderr: list[bytes] = []
        self._stderr_thread: threading.Thread | None = None

    def __enter__(self) -> "_RawVideoEncoder":
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        return self

    def _drain_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr.append(line)

    def write(self, frame: np.ndarray) -> None:
        assert self._process is not None and self._process.stdin is not None
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(f"ffmpeg exited early writing {self.path}: {self._stderr_text()}") from exc

    def _stderr_text(self) -> str:
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
        return b"".join(self._stderr).decode(errors="replace").strip()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self._process is not None
        if self._process.stdin is not None and not self._process.stdin.closed:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
        return_code = self._process.wait()
        message = self._stderr_text()
        if self._process.stderr is not None and not self._process.stderr.closed:
            self._process.stderr.close()
        if exc_type is None:
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed (exit {return_code}) writing {self.path}: {message}")
            # A zero exit with no output means the encoder accepted every frame and
            # wrote nothing — silently shipping an empty file is worse than failing.
            if not self.path.exists() or self.path.stat().st_size == 0:
                raise RuntimeError(f"ffmpeg produced an empty file at {self.path}: {message}")


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
        person_format: PersonFormat = "matte",
    ) -> ExportResult:
        if person_format not in ("matte", "fill_matte", "webm_alpha"):
            raise ValueError(
                f"unknown person_format {person_format!r} — expected 'matte', 'fill_matte' or 'webm_alpha'"
            )

        self._assert_complete_coverage(tracking, metadata)

        width, height = self._padded_size(metadata)

        if person_format == "matte":
            matte_path = self._write_matte(tracking, metadata)
            return ExportResult(
                format="matte",
                matte_path=matte_path,
                frame_count=metadata.frame_count,
                fps=metadata.fps,
                width=width,
                height=height,
            )

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
            width=width,
            height=height,
        )

    @staticmethod
    def _padded_size(metadata: VideoMetadata) -> tuple[int, int]:
        return metadata.width + (metadata.width % 2), metadata.height + (metadata.height % 2)

    @staticmethod
    def _assert_complete_coverage(tracking: TrackingResult, metadata: VideoMetadata) -> None:
        missing = sorted(set(range(metadata.frame_count)) - set(tracking.masks))
        if not missing:
            return
        coverage = tracking.coverage(metadata.frame_count)
        raise ValueError(
            f"tracking covered {coverage:.1%} of frames — {len(missing)} of "
            f"{metadata.frame_count} are missing, starting at frame {missing[0]}. "
            "Every frame needs a mask before export. Try a more specific prompt, raise "
            "max_segments_per_direction, or split the clip where the subject leaves frame."
        )

    def _write_matte(self, tracking: TrackingResult, metadata: VideoMetadata) -> Path:
        matte_path = self.output_dir / "person_matte.mp4"
        with _RawVideoEncoder(
            self.ffmpeg_bin,
            matte_path,
            metadata.width,
            metadata.height,
            metadata.fps,
            "gray",
            (*_EVEN_PAD, *_MATTE_ENCODE_ARGS),
        ) as encoder:
            for frame_idx in range(metadata.frame_count):
                encoder.write(np.ascontiguousarray(tracking.masks[frame_idx]))
        return matte_path

    def _write_webm_alpha(self, tracking: TrackingResult, metadata: VideoMetadata) -> Path:
        person_path = self.output_dir / "person.webm"
        with _RawVideoEncoder(
            self.ffmpeg_bin,
            person_path,
            metadata.width,
            metadata.height,
            metadata.fps,
            "bgra",
            (
                "-c:v", "libvpx-vp9",
                "-pix_fmt", "yuva420p",
                # VP8 + alpha + alt-ref is a hard error in ffmpeg; for VP9 the two
                # streams no longer desync, but this stays set for older players.
                "-auto-alt-ref", "0",
                # Without an explicit rate, libvpx defaults to a 256 kbit/s target
                # — severe mush at 720p, on the format documented as high fidelity.
                "-crf", "30",
                "-b:v", "0",
                "-row-mt", "1",
                "-deadline", "good",
                "-cpu-used", "2",
            ),
        ) as encoder:
            for frame_idx, frame_bgr in self.video.iter_frames():
                bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
                bgra[:, :, 3] = tracking.masks[frame_idx]
                encoder.write(bgra)
        return person_path

    def _write_fill_matte(self, tracking: TrackingResult, metadata: VideoMetadata) -> tuple[Path, Path]:
        fill_path = self.output_dir / "person_fill.mp4"
        matte_path = self.output_dir / "person_matte.mp4"

        with _RawVideoEncoder(
            self.ffmpeg_bin,
            fill_path,
            metadata.width,
            metadata.height,
            metadata.fps,
            "bgr24",
            (*_EVEN_PAD, *_FILL_ENCODE_ARGS),
        ) as fill_encoder, _RawVideoEncoder(
            self.ffmpeg_bin,
            matte_path,
            metadata.width,
            metadata.height,
            metadata.fps,
            "gray",
            (*_EVEN_PAD, *_MATTE_ENCODE_ARGS),
        ) as matte_encoder:
            for frame_idx, frame_bgr in self.video.iter_frames():
                mask = tracking.masks[frame_idx]
                fill = frame_bgr.copy()
                # Documented contract of this format. See the module docstring for
                # why it costs a dark rim, and prefer "matte" if that matters.
                fill[mask == 0] = 0
                fill_encoder.write(np.ascontiguousarray(fill))
                matte_encoder.write(np.ascontiguousarray(mask))

        return fill_path, matte_path
