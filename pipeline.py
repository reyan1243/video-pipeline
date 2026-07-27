"""Orchestrator wiring all stages together, plus a thin CLI entrypoint.

This is also the seam a future FastAPI service wraps: one call in, two
video files out.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from detector import SubjectDetector
from exporter import LayerExporter, PersonFormat
from mask_ops import clean_masks
from segmenter import SeedSegmenter
from tracker import MaskTracker
from datatypes import ExportResult, SeedFrame, TrackerConfig, TrackingResult, VideoMetadata
from video_source import VideoSource


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
        tracker_config: TrackerConfig = TrackerConfig(),
        mask_close_kernel_size: int = 45,
        person_format: PersonFormat = "fill_matte",
        device: str | None = None,
    ):
        self.video = VideoSource(video_path)
        self.detector = SubjectDetector(prompt=prompt, device=device)
        self.segmenter = SeedSegmenter(self.detector, num_candidates=num_seed_candidates, device=device)
        self.tracker = MaskTracker(self.video, self.detector, config=tracker_config, device=device)
        self.exporter = LayerExporter(self.video, output_dir=output_dir)
        self.mask_close_kernel_size = mask_close_kernel_size
        self.person_format = person_format

    def run(self) -> PipelineResult:
        metadata = self.video.load_metadata()
        seed = self.segmenter.select_seed(self.video, metadata)
        tracking = self.tracker.track(seed, metadata)
        tracking.masks = clean_masks(tracking.masks, self.mask_close_kernel_size)
        export = self.exporter.export(tracking, metadata, person_format=self.person_format)
        return PipelineResult(metadata=metadata, seed=seed, tracking=tracking, export=export)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Detect, segment, and track a subject; export person + background layers."
    )
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--prompt", required=True, help='e.g. "person" or "guitarist"')
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--person-format", choices=["webm_alpha", "fill_matte"], default="fill_matte")
    parser.add_argument("--num-seed-candidates", type=int, default=12)
    parser.add_argument("--mask-close-kernel-size", type=int, default=45)
    args = parser.parse_args(argv)

    pipeline = VideoLayerPipeline(
        video_path=args.video_path,
        prompt=args.prompt,
        output_dir=args.output_dir,
        num_seed_candidates=args.num_seed_candidates,
        mask_close_kernel_size=args.mask_close_kernel_size,
        person_format=args.person_format,
    )
    result = pipeline.run()

    print(f"Tracked {len(result.tracking.masks)}/{result.metadata.frame_count} frames")
    print(f"Person layer:     {result.export.person_path}")
    if result.export.matte_path:
        print(f"Person matte:     {result.export.matte_path}")
    print(f"Background layer: {result.export.background_path}")


if __name__ == "__main__":
    main()
