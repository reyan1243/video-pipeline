"""End-to-end smoke test: real models, real (short) clip, no assertions on
exact numbers — prints diagnostics for a human to sanity-check. Runs against
the bundled fixture clip by default so it works out of the box; pass a path
to run against a real video instead.

Usage:
    python -m smoke_test [video_path]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from pipeline import VideoLayerPipeline

DEFAULT_VIDEO_PATH = Path(__file__).parent / "tests" / "fixtures" / "short_clip.mp4"
OUTPUT_DIR = Path("smoke_test_output")
PROMPT = "person"


def main() -> None:
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    if not video_path.exists():
        sys.exit(f"missing input video: {video_path}")

    print(f"video: {video_path}")
    print(f"prompt: {PROMPT!r}")
    print(f"output dir: {OUTPUT_DIR}")
    print()

    t0 = time.time()
    pipeline = VideoLayerPipeline(
        video_path=video_path,
        prompt=PROMPT,
        output_dir=OUTPUT_DIR,
        person_format="fill_matte",
    )
    print(f"[{time.time() - t0:.1f}s] models loaded, device={pipeline.detector.device}")

    result = pipeline.run()
    elapsed = time.time() - t0

    metadata = result.metadata
    tracking = result.tracking
    seed = result.seed
    export = result.export

    print()
    print("=== metadata ===")
    print(f"fps={metadata.fps} width={metadata.width} height={metadata.height} frame_count={metadata.frame_count}")

    print()
    print("=== seed ===")
    print(f"frame_index={seed.frame_index} iou_score={seed.iou_score:.3f} box={seed.box}")

    print()
    print("=== tracking ===")
    coverage = tracking.coverage(metadata.frame_count)
    print(f"tracked {len(tracking.masks)}/{metadata.frame_count} frames (coverage={coverage:.1%})")
    missing = sorted(set(range(metadata.frame_count)) - set(tracking.masks))
    if missing:
        print(f"missing frames ({len(missing)}): {missing[:20]}{'...' if len(missing) > 20 else ''}")
    if tracking.diagnostics:
        ious = [d.iou for d in tracking.diagnostics]
        print(f"consecutive-frame IoU: mean={sum(ious) / len(ious):.3f} min={min(ious):.3f}")
    disruptions_forward = sum(1 for d in tracking.diagnostics if d.direction == "forward")
    disruptions_backward = sum(1 for d in tracking.diagnostics if d.direction == "backward")
    print(f"diagnostic samples: forward={disruptions_forward} backward={disruptions_backward}")

    print()
    print("=== export ===")
    print(f"format={export.format}")
    print(f"person_path={export.person_path}")
    if export.matte_path:
        print(f"matte_path={export.matte_path}")
    print(f"background_path={export.background_path}")

    print()
    print(f"total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
