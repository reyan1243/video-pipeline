"""Orchestrator wiring all stages together, plus a thin CLI entrypoint.

This is also the seam a future FastAPI service wraps: one call in, the subject
matte out.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Importing torch + transformers takes tens of seconds on a cold container
# filesystem, and it happens below, before any of our code can run. Announce it
# first or the CLI looks hung for its slowest single step. Guarded so importing
# this module as a library stays silent.
if __name__ == "__main__":
    print(
        f"[{time.strftime('%H:%M:%S')}] importing torch + transformers "
        "(slow on a cold container, cached afterwards)",
        file=sys.stderr,
        flush=True,
    )

from detector import SubjectDetector
from exporter import LayerExporter, PersonFormat
from mask_ops import DEFAULT_CLOSE_KERNEL_SIZE, DEFAULT_FEATHER_SIGMA, clean_masks, temporal_median
from segmenter import SeedSegmenter
from tracker import MaskTracker
from datatypes import ExportResult, SeedFrame, TrackerConfig, TrackingResult, VideoMetadata
from video_source import VideoSource

MODEL_IDS = {
    "large": "facebook/sam2.1-hiera-large",
    "base-plus": "facebook/sam2.1-hiera-base-plus",
    "small": "facebook/sam2.1-hiera-small",
    "tiny": "facebook/sam2.1-hiera-tiny",
}


@dataclass(frozen=True)
class PipelineResult:
    metadata: VideoMetadata
    seed: SeedFrame
    tracking: TrackingResult
    export: ExportResult


class VideoLayerPipeline:
    def __init__(
        self,
        video_path: str | Path,
        prompt: str,
        output_dir: str | Path,
        num_seed_candidates: int = 12,
        tracker_config: TrackerConfig | None = None,
        mask_close_kernel_size: int = DEFAULT_CLOSE_KERNEL_SIZE,
        feather_sigma: float = DEFAULT_FEATHER_SIGMA,
        temporal_smoothing: bool = False,
        person_format: PersonFormat = "matte",
        device: str | None = None,
        model_size: str = "large",
        release_seed_model: bool = True,
        prefer_bfloat16: bool = False,
        verbose: bool = False,
    ):
        if model_size not in MODEL_IDS:
            raise ValueError(f"unknown model_size {model_size!r} — expected one of {sorted(MODEL_IDS)}")
        model_id = MODEL_IDS[model_size]

        self.verbose = verbose
        self.video = VideoSource(video_path)

        # ~2.7 GB of weights move disk -> GPU here, and it used to happen in
        # total silence before run() printed anything, which reads as a hang.
        # Note sam2 is loaded TWICE: once as Sam2Model to score seed frames,
        # once as Sam2VideoModel to track. Same checkpoint, two objects.
        loaded_at = time.time()
        self._log("loading grounding-dino")
        self.detector = SubjectDetector(prompt=prompt, device=device, prefer_bfloat16=prefer_bfloat16)
        self._log(f"loading {model_id} (image)")
        self.segmenter = SeedSegmenter(
            self.detector,
            num_candidates=num_seed_candidates,
            device=device,
            model_id=model_id,
            prefer_bfloat16=prefer_bfloat16,
        )
        self._log(f"loading {model_id} (video)")
        self.tracker = MaskTracker(
            self.video,
            self.detector,
            config=tracker_config or TrackerConfig(),
            device=device,
            model_id=model_id,
            prefer_bfloat16=prefer_bfloat16,
        )
        self._log(f"models ready in {time.time() - loaded_at:.1f}s")

        self.exporter = LayerExporter(self.video, output_dir=output_dir)
        self.mask_close_kernel_size = mask_close_kernel_size
        self.feather_sigma = feather_sigma
        self.temporal_smoothing = temporal_smoothing
        self.person_format = person_format
        self.release_seed_model = release_seed_model

    def _log(self, message: str) -> None:
        # A long run used to print nothing at all until it finished, which makes
        # a 30-minute job indistinguishable from a hang.
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)

    def run(self) -> PipelineResult:
        self._log("reading metadata")
        metadata = self.video.load_metadata()
        self._log(f"{metadata.frame_count} frames at {metadata.fps:.3g} fps, {metadata.width}x{metadata.height}")

        self._log("selecting seed frame")
        seed = self.segmenter.select_seed(self.video, metadata)
        self._log(f"seed frame {seed.frame_index} (iou {seed.iou_score:.3f})")

        if self.release_seed_model:
            # Frees a second full copy of the SAM2 checkpoint before the tracking
            # session — which is the stage that actually runs out of VRAM.
            self.segmenter.release()

        self._log("tracking")
        tracking = self.tracker.track(seed, metadata)
        peak = getattr(self.tracker, "peak_bytes", 0)
        longest = getattr(self.tracker, "longest_segment", 0) or 1
        self._log(f"tracked {len(tracking.masks)}/{metadata.frame_count} frames")
        if peak:
            # Reported per frame of the LONGEST segment, since that is what sets
            # the ceiling — a session's state is freed when the segment ends.
            self._log(
                f"peak VRAM {peak / 1024**3:.2f} GiB over a {longest}-frame segment "
                f"({peak / 1024**2 / longest:.1f} MiB/frame)"
            )

        self._log("cleaning masks")
        clean_masks(tracking.masks, self.mask_close_kernel_size, self.feather_sigma)
        if self.temporal_smoothing:
            self._log("temporal smoothing")
            temporal_median(tracking.masks)

        self._log(f"exporting ({self.person_format})")
        export = self.exporter.export(tracking, metadata, person_format=self.person_format)
        self._log("done")
        return PipelineResult(metadata=metadata, seed=seed, tracking=tracking, export=export)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Detect, segment, and track a subject; export a matte (and optional layers)."
    )
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--prompt", required=True, help='e.g. "person" or "guitarist"')
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--person-format",
        choices=["matte", "fill_matte", "webm_alpha"],
        default="matte",
        help="matte: grayscale silhouette only (default, smallest, no edge rim)",
    )
    parser.add_argument("--num-seed-candidates", type=int, default=12)

    quality = parser.add_argument_group("matte quality")
    quality.add_argument(
        "--mask-close-kernel-size",
        type=int,
        default=DEFAULT_CLOSE_KERNEL_SIZE,
        help=f"morphological close kernel (default {DEFAULT_CLOSE_KERNEL_SIZE}); large values "
        "bridge arm-to-torso gaps and fill them solid",
    )
    quality.add_argument(
        "--feather-sigma",
        type=float,
        default=DEFAULT_FEATHER_SIGMA,
        help=f"gaussian edge feather in pixels (default {DEFAULT_FEATHER_SIGMA}, 0 disables)",
    )
    quality.add_argument(
        "--temporal-smoothing",
        action="store_true",
        help="3-tap median across frames; removes edge jitter, costs a frame of lag",
    )

    tracking = parser.add_argument_group("tracking")
    tracking.add_argument("--max-segments-per-direction", type=int, default=TrackerConfig().max_segments_per_direction)
    tracking.add_argument(
        "--max-segment-frames",
        type=int,
        default=TrackerConfig().max_segment_frames,
        help="caps VRAM: SAM2 retains ~18 MB per frame per session and never evicts",
    )
    tracking.add_argument(
        "--cpu-offload",
        action="store_true",
        help="hold session state in host RAM; removes the frame ceiling, ~22%% slower",
    )
    tracking.add_argument(
        "--max-mask-height",
        type=int,
        default=TrackerConfig().max_mask_height,
        help="cap matte height, preserving aspect (0 = source). SAM2 decodes masks at "
        "256x256 internally, so 960 on a 1080p source costs almost no real detail and "
        "quarters cleanup, encode and RAM",
    )
    tracking.add_argument(
        "--bf16-weights",
        action="store_true",
        help="load weights in bfloat16 on CUDA — roughly halves the ~2.7GB startup "
        "load and the VRAM they occupy; activations already run bf16 under autocast",
    )
    tracking.add_argument(
        "--model-size",
        choices=sorted(MODEL_IDS),
        default="large",
        help="smaller is faster and measurably less accurate (default: large)",
    )

    args = parser.parse_args(argv)

    pipeline = VideoLayerPipeline(
        video_path=args.video_path,
        prompt=args.prompt,
        output_dir=args.output_dir,
        num_seed_candidates=args.num_seed_candidates,
        tracker_config=TrackerConfig(
            max_segments_per_direction=args.max_segments_per_direction,
            max_segment_frames=args.max_segment_frames,
            cpu_offload=args.cpu_offload,
            max_mask_height=args.max_mask_height,
        ),
        mask_close_kernel_size=args.mask_close_kernel_size,
        feather_sigma=args.feather_sigma,
        temporal_smoothing=args.temporal_smoothing,
        person_format=args.person_format,
        model_size=args.model_size,
        prefer_bfloat16=args.bf16_weights,
        verbose=True,
    )
    result = pipeline.run()

    print(f"Tracked {len(result.tracking.masks)}/{result.metadata.frame_count} frames")
    if result.export.matte_path:
        print(f"Matte:            {result.export.matte_path}")
    if result.export.person_path:
        print(f"Person layer:     {result.export.person_path}")
    if result.export.background_path:
        print(f"Background layer: {result.export.background_path}")


if __name__ == "__main__":
    main()
