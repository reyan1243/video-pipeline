"""RunPod Serverless handler.

Design notes, each of which costs real money if ignored:

* **Models load once per worker, at import.** RunPod bills worker start time —
  container init *and* loading weights into GPU memory — so constructing the
  pipeline per request pays ~1.8 GB of disk reads on the meter every single job.
  The stages are built here as module-level singletons and their per-request
  fields (`prompt`, `num_candidates`, `video`) are reassigned; they are plain
  attributes, so this is safe.

* **The seed model is deliberately NOT released.** `SeedSegmenter.release()` is
  right for a one-shot CLI run but wrong here: a warm worker would have to reload
  the checkpoint on its next job, which costs more than the VRAM is worth.

* **Guards run before any GPU work.** Duration, download size, and projected host
  RAM are all checked first so an oversized job is rejected for free instead of
  being OOM-killed after 30 minutes of billed compute.

* **Idempotent by content hash.** RunPod's heartbeat is 10s and a lapse requeues
  the job, so a handler genuinely can run twice for one submission. Output keys
  derive from sha256(source bytes + params), and an existing object short-circuits
  the whole run — which also makes retries and duplicate submissions free.

* **Expected failures return `ok: False`, not `error`.** Returning a truthy
  `"error"` key marks the whole job **FAILED** — wire-identical to an uncaught
  exception — which loses the structured detail a caller needs to tell "your clip
  is too long" from "the worker crashed". Anything the caller can act on comes
  back as `{"ok": False, "code": ..., "reason": ...}` with the job COMPLETED.
  Only genuine crashes produce FAILED.

* **No `refresh_worker`.** It wipes worker state and forces a model reload,
  defeating everything above.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

import boto3
import runpod
import torch
from boto3.s3.transfer import TransferConfig
from botocore.client import Config
from botocore.exceptions import ClientError

sys.path.insert(0, "/app")

from datatypes import TrackerConfig  # noqa: E402
from detector import SubjectDetector, _normalize_prompt  # noqa: E402
from exporter import LayerExporter  # noqa: E402
from mask_ops import (  # noqa: E402
    DEFAULT_CLOSE_KERNEL_SIZE,
    DEFAULT_FEATHER_SIGMA,
    clean_masks,
    temporal_median,
)
from segmenter import SeedSegmenter  # noqa: E402
from tracker import MaskTracker  # noqa: E402
from video_source import VideoSource  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "")
S3_REGION = os.environ.get("S3_REGION", "auto")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY_ID", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_ACCESS_KEY", "")
S3_PREFIX = os.environ.get("S3_PREFIX", "mattes")
URL_TTL_SECONDS = int(os.environ.get("URL_TTL_SECONDS", 86400))

MAX_DURATION_SECONDS = float(os.environ.get("MAX_DURATION_SECONDS", 300))
MAX_DOWNLOAD_BYTES = int(os.environ.get("MAX_DOWNLOAD_BYTES", 2 * 1024**3))
# Host RAM ceiling for the mask dictionary: frame_count * width * height bytes.
MAX_MASK_BYTES = int(os.environ.get("MAX_MASK_BYTES", 8 * 1024**3))
MAX_SEGMENTS_PER_DIRECTION = int(os.environ.get("MAX_SEGMENTS_PER_DIRECTION", 20))
MAX_SEGMENT_FRAMES = int(os.environ.get("MAX_SEGMENT_FRAMES", 600))
# Cap matte height (0 = source resolution). SAM2 decodes masks at 256x256
# internally, so 960 on a 1080p source discards almost no real detail while
# quartering mask cleanup, encoding and RAM — which together outweigh tracking.
MAX_MASK_HEIGHT = int(os.environ.get("MAX_MASK_HEIGHT", 0))
MODEL_SIZE = os.environ.get("MODEL_SIZE", "large")

MODEL_IDS = {
    "large": "facebook/sam2.1-hiera-large",
    "base-plus": "facebook/sam2.1-hiera-base-plus",
    "small": "facebook/sam2.1-hiera-small",
    "tiny": "facebook/sam2.1-hiera-tiny",
}

# --------------------------------------------------------------------------
# Cold start — once per worker
# --------------------------------------------------------------------------

_t0 = time.time()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_ID = MODEL_IDS.get(MODEL_SIZE, MODEL_IDS["large"])

_TRACKER_CONFIG = TrackerConfig(
    max_segments_per_direction=MAX_SEGMENTS_PER_DIRECTION,
    max_segment_frames=MAX_SEGMENT_FRAMES,
    max_mask_height=MAX_MASK_HEIGHT,
)

_detector = SubjectDetector(prompt="person", device=DEVICE)
_segmenter = SeedSegmenter(_detector, device=DEVICE, model_id=_MODEL_ID)
_tracker = MaskTracker(
    video=None,  # replaced per request
    detector=_detector,
    config=_TRACKER_CONFIG,
    device=DEVICE,
    model_id=_MODEL_ID,
)

print(
    f"[cold start] {_MODEL_ID} loaded in {time.time() - _t0:.1f}s on {DEVICE}",
    flush=True,
)

_s3 = None
if S3_BUCKET:
    _s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT or None,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _download(url: str, dest: Path) -> str:
    """Stream to disk, enforcing the size cap, returning the sha256."""
    request = urllib.request.Request(url, headers={"User-Agent": "video-pipeline-worker"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"video exceeds the {MAX_DOWNLOAD_BYTES} byte limit")

        written = 0
        with dest.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"video exceeds the {MAX_DOWNLOAD_BYTES} byte limit")
                digest.update(chunk)
                handle.write(chunk)
    return digest.hexdigest()


def _probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    parsed = json.loads(result.stdout)
    stream = parsed["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration": float(parsed["format"]["duration"]),
    }


def _trim(source: Path, dest: Path, start: float | None, end: float | None) -> Path:
    """Cut the requested range before any GPU work.

    This is the main cost lever the API exposes: billing is per GPU-second, so a
    caller who only needs 20s out of a 3-minute clip should pay for 20s. Re-encodes
    rather than stream-copying because `-c copy` cuts on keyframes, and a matte
    that is off by a few frames is worse than useless.
    """
    if start is None and end is None:
        return source
    command = ["ffmpeg", "-y", "-v", "error", "-i", str(source)]
    if start is not None:
        command += ["-ss", f"{start:.3f}"]
    if end is not None:
        command += ["-to", f"{end:.3f}"]
    command += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-an", str(dest)]
    subprocess.run(command, check=True, capture_output=True)
    return dest


def _output_key(content_hash: str, params: dict) -> str:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    combined = hashlib.sha256(f"{content_hash}:{canonical}".encode()).hexdigest()[:32]
    return f"{S3_PREFIX}/{combined}/person_matte.mp4"


def _presign(key: str) -> str:
    return _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=URL_TTL_SECONDS,
    )


def _already_done(key: str) -> bool:
    if _s3 is None:
        return False
    try:
        _s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError:
        return False


# Cloudflare R2 rejects multipart parts below 5 MiB. boto3's default chunk size is
# fine, but the SDK's own rp_upload helper uses 25 KiB and fails against R2 — this
# is set explicitly so nobody "simplifies" it back to the helper later.
_TRANSFER = TransferConfig(
    multipart_threshold=8 * 1024**2,
    multipart_chunksize=8 * 1024**2,
)


def _upload(local: Path, key: str) -> str:
    if _s3 is None:
        raise RuntimeError("S3 is not configured — set S3_BUCKET and credentials")
    _s3.upload_file(
        str(local), S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"}, Config=_TRANSFER
    )
    url = _presign(key)
    if not url.startswith("https://"):
        raise RuntimeError(f"presigned URL is not https: {url!r}")
    return url


def _rejected(code: str, reason: str, **extra) -> dict:
    """An expected, caller-actionable failure.

    Deliberately not keyed "error": a truthy "error" marks the job FAILED, which
    is indistinguishable from a crash. This keeps the job COMPLETED and hands the
    caller something it can branch on.
    """
    return {"ok": False, "code": code, "reason": reason, **extra}


def _cleanup(workdir: Path) -> None:
    shutil.rmtree(workdir, ignore_errors=True)
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Handler
# --------------------------------------------------------------------------


def handler(job):
    job_input = job.get("input") or {}

    video_url = job_input.get("video_url")
    if not video_url:
        return _rejected("missing_input", "missing required input 'video_url'")

    prompt = str(job_input.get("prompt", "person"))
    num_seed_candidates = int(job_input.get("num_seed_candidates", 12))
    close_kernel = int(job_input.get("mask_close_kernel_size", DEFAULT_CLOSE_KERNEL_SIZE))
    feather_sigma = float(job_input.get("feather_sigma", DEFAULT_FEATHER_SIGMA))
    smoothing = bool(job_input.get("temporal_smoothing", False))
    max_mask_height = int(job_input.get("max_mask_height", MAX_MASK_HEIGHT))
    start_time = job_input.get("start_time")
    end_time = job_input.get("end_time")
    start_time = float(start_time) if start_time is not None else None
    end_time = float(end_time) if end_time is not None else None

    params = {
        "prompt": prompt,
        "candidates": num_seed_candidates,
        "close": close_kernel,
        "feather": feather_sigma,
        "smoothing": smoothing,
        "max_mask_height": max_mask_height,
        "start": start_time,
        "end": end_time,
        "model": _MODEL_ID,
    }

    workdir = Path(tempfile.mkdtemp(prefix="vp-"))
    started = time.time()

    try:
        source = workdir / "input.mp4"
        content_hash = _download(video_url, source)

        key = _output_key(content_hash, params)
        if _already_done(key):
            # Same bytes, same settings — a duplicate submission or a requeue after
            # a heartbeat lapse. Returning the existing object costs nothing.
            return {
                "ok": True,
                "matte_key": key,
                "matte_url": _presign(key),
                "cached": True,
                "processing_seconds": round(time.time() - started, 1),
                "url_expires_in_seconds": URL_TTL_SECONDS,
            }

        clip = _trim(source, workdir / "clip.mp4", start_time, end_time)

        info = _probe(clip)
        if info["duration"] > MAX_DURATION_SECONDS:
            return _rejected(
                "too_long",
                f"clip is {info['duration']:.1f}s, limit is {MAX_DURATION_SECONDS:.0f}s. "
                "Pass start_time/end_time to process only the range you need, or split it.",
                duration=info["duration"],
            )

        runpod.serverless.progress_update(job, "reading metadata")
        video = VideoSource(clip)
        metadata = video.load_metadata()

        projected = metadata.frame_count * metadata.width * metadata.height
        if projected > MAX_MASK_BYTES:
            return _rejected(
                "too_large",
                f"job would need ~{projected / 1024**3:.1f} GB of mask memory "
                f"(limit {MAX_MASK_BYTES / 1024**3:.1f} GB). Downscale to 720p, or "
                "use start_time/end_time to process a shorter range.",
                frames=metadata.frame_count,
                resolution=f"{metadata.width}x{metadata.height}",
            )

        # TrackerConfig is frozen, so a per-request override rebuilds it. The
        # tracker reads self.config on every frame, so reassigning is enough.
        _tracker.config = replace(_TRACKER_CONFIG, max_mask_height=max_mask_height)

        _detector.prompt = _normalize_prompt(prompt)
        _segmenter.num_candidates = num_seed_candidates
        _tracker.video = video

        runpod.serverless.progress_update(job, "selecting seed frame")
        seed = _segmenter.select_seed(video, metadata)

        runpod.serverless.progress_update(job, f"tracking {metadata.frame_count} frames")
        tracking = _tracker.track(seed, metadata)

        coverage = tracking.coverage(metadata.frame_count)
        if coverage < 1.0:
            missing = metadata.frame_count - len(tracking.masks)
            return _rejected(
                "incomplete_coverage",
                f"tracking covered {coverage:.1%} of frames ({missing} missing). "
                "Every frame needs a mask. Try a more specific prompt, or split the "
                "clip where the subject leaves frame.",
                coverage=coverage,
            )

        runpod.serverless.progress_update(job, "cleaning masks")
        clean_masks(tracking.masks, close_kernel, feather_sigma)
        if smoothing:
            temporal_median(tracking.masks)

        runpod.serverless.progress_update(job, "encoding matte")
        exporter = LayerExporter(video, output_dir=workdir / "out")
        export = exporter.export(tracking, metadata, person_format="matte")

        runpod.serverless.progress_update(job, "uploading")
        return {
            "ok": True,
            # The key is the durable reference — store this. The URL is a
            # convenience for testing and expires after URL_TTL_SECONDS; callers
            # should re-presign from the key rather than persist the URL.
            "matte_key": key,
            "matte_url": _upload(Path(export.matte_path), key),
            "cached": False,
            "frames": metadata.frame_count,
            "fps": metadata.fps,
            "width": export.width,
            "height": export.height,
            "seed_frame": seed.frame_index,
            "seed_iou": round(seed.iou_score, 3),
            "coverage": 1.0,
            "prompt": prompt,
            "processing_seconds": round(time.time() - started, 1),
            "url_expires_in_seconds": URL_TTL_SECONDS,
        }

    except ValueError as exc:
        # Guard rejections (oversize download, unusable fps, no detection) — all
        # caller-actionable, so COMPLETED with ok=False rather than FAILED.
        return _rejected("invalid_input", str(exc))
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode(errors="replace")[-500:] if exc.stderr else ""
        return _rejected("ffmpeg_failed", f"ffmpeg failed: {detail}")
    except Exception as exc:  # noqa: BLE001 — surface the real failure to the caller
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "processing_seconds": round(time.time() - started, 1),
        }
    finally:
        _cleanup(workdir)


runpod.serverless.start({"handler": handler})
