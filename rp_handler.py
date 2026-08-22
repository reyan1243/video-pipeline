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
import threading
import time
import urllib.error
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

from arch_guard import compiled_archs, covers  # noqa: E402
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
# bfloat16 weights halve the ~2.7GB that moves disk -> GPU at worker start.
# RunPod bills that start time, so this is a direct cold-start saving.
BF16_WEIGHTS = os.environ.get("BF16_WEIGHTS", "").strip().lower() in {"1", "true", "yes"}

MODEL_IDS = {
    "large": "facebook/sam2.1-hiera-large",
    "base-plus": "facebook/sam2.1-hiera-base-plus",
    "small": "facebook/sam2.1-hiera-small",
    "tiny": "facebook/sam2.1-hiera-tiny",
}

# --------------------------------------------------------------------------
# Cold start — once per worker
# --------------------------------------------------------------------------

# Set by the contract tests, which stub torch and exercise the CPU-only paths.
# Never set on a worker: a serverless run that falls back to CPU does not
# degrade, it burns the full 900s execution timeout on the meter and returns
# nothing.
_ALLOW_CPU = os.environ.get("ALLOW_CPU", "").strip().lower() in {"1", "true", "yes"}


def _select_device() -> str:
    """`"cuda"`, or a loud failure naming the GPU that torch cannot drive.

    A worker that lands on a GPU outside torch's compiled arch list dies with
    `no kernel image is available for execution on the device` — or, worse,
    `torch.cuda.is_available()` swallows the init failure, returns False, and
    the worker runs the whole pipeline on CPU until the execution timeout kills
    it. Both look like a queue stall from the caller's side.

    Checking here turns either one into a startup crash naming the card and the
    arch list, which is the pair of facts needed to fix it. CUDA cubins are
    forward-compatible within a major version, so an sm_86 binary runs on sm_89
    but nothing in sm_9x or sm_12x.
    """
    try:
        available = torch.cuda.is_available()
    except RuntimeError as exc:  # CUDA init itself blew up
        raise RuntimeError(f"CUDA present but unusable: {exc}") from exc

    if not available:
        if _ALLOW_CPU:
            return "cpu"
        raise RuntimeError(
            "no usable CUDA device; refusing to fall back to CPU — every job "
            "would run to the execution timeout and be billed for it"
        )

    major, minor = torch.cuda.get_device_capability()
    archs = compiled_archs()
    if not covers(archs, (major, minor)):
        raise RuntimeError(
            f"{torch.cuda.get_device_name()} is sm_{major}{minor}, which torch "
            f"{torch.__version__} (cuda {torch.version.cuda}) has no kernels for. "
            f"Compiled for: {archs}. Either deselect this GPU type on the endpoint "
            f"or rebuild on a base image covering sm_{major}{minor}."
        )

    print(f"gpu: {torch.cuda.get_device_name()} (sm_{major}{minor})", flush=True)
    return "cuda"


_t0 = time.time()
DEVICE = _select_device()
_MODEL_ID = MODEL_IDS.get(MODEL_SIZE, MODEL_IDS["large"])

_TRACKER_CONFIG = TrackerConfig(
    max_segments_per_direction=MAX_SEGMENTS_PER_DIRECTION,
    max_segment_frames=MAX_SEGMENT_FRAMES,
    max_mask_height=MAX_MASK_HEIGHT,
)

_detector = SubjectDetector(prompt="person", device=DEVICE, prefer_bfloat16=BF16_WEIGHTS)
_segmenter = SeedSegmenter(
    _detector, device=DEVICE, model_id=_MODEL_ID, prefer_bfloat16=BF16_WEIGHTS
)
_tracker = MaskTracker(
    video=None,  # replaced per request
    detector=_detector,
    config=_TRACKER_CONFIG,
    device=DEVICE,
    model_id=_MODEL_ID,
    prefer_bfloat16=BF16_WEIGHTS,
)

print(
    f"[cold start] {_MODEL_ID} on {DEVICE} in {time.time() - _t0:.1f}s "
    f"(bf16_weights={BF16_WEIGHTS}, max_mask_height={MAX_MASK_HEIGHT or 'source'}). "
    "This is paid once per worker, not per job.",
    flush=True,
)

# The stages are module-level singletons whose per-request fields are mutated,
# so two jobs on one worker would corrupt each other. RunPod sends one job per
# worker by default (no concurrency_modifier is set), making this belt-and-braces
# — but the failure it prevents is silent wrong output, not a crash.
_JOB_LOCK = threading.Lock()

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
    with _OPENER.open(request, timeout=120) as response:
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


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects on the source fetch.

    The worker is handed a URL by our backend and fetches it with no further
    checks, so a redirect is the one way that URL could point somewhere it was
    never signed for — including an address inside the worker's own network.
    Presigned R2 URLs never redirect, so refusing costs nothing. Mirrors the
    `maxRedirects: 0` already used on untrusted fetches in apps/api.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(newurl, code, f"redirect refused ({code})", headers, fp)


_OPENER = urllib.request.build_opener(_NoRedirect)


def _remote_identity(url: str) -> str | None:
    """Identify the source object without downloading it.

    Uses a one-byte ranged GET rather than HEAD: a presigned URL is signed for a
    specific method, so HEAD against a `get_object` presign fails signature
    validation. A Range request is still a GET, so the signature holds, and the
    response carries ETag and the total size in Content-Range.

    Returns None whenever the origin does not cooperate — the caller then falls
    back to hashing the downloaded bytes.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "video-pipeline-worker", "Range": "bytes=0-0"}
    )
    try:
        with _OPENER.open(request, timeout=30) as response:
            etag = (response.headers.get("ETag") or "").strip().strip('"')
            content_range = response.headers.get("Content-Range") or ""
    except Exception:  # noqa: BLE001 — any failure just means "cannot identify"
        return None

    if not etag:
        return None
    total = content_range.rsplit("/", 1)[-1] if "/" in content_range else ""
    return f"{etag}:{total}"


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
    streams = parsed.get("streams") or []
    if not streams:
        raise ValueError(
            "no video stream found — the URL points at audio, an image, or a file "
            "that is not a video"
        )
    stream = streams[0]
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


def _put_presigned(local: Path, url: str, content_type: str = "video/mp4") -> None:
    """Upload via a presigned PUT supplied by the caller.

    This is the preferred path, and it exists because the destination is not the
    worker's to choose. Projects carry their own `storage_provider` (R2, S3 or
    B2) and R2 uploads can fail over to S3, so a bucket hardcoded here would
    scatter mattes away from the projects they belong to.

    Letting the API sign the destination also means the worker holds no storage
    credentials whatsoever — it can write exactly one object, the one it was
    asked to produce, and nothing else in any bucket.

    The Content-Type must match whatever the URL was signed with, or the origin
    rejects the signature.
    """
    payload = local.read_bytes()
    request = urllib.request.Request(
        url,
        data=payload,
        method="PUT",
        headers={"Content-Type": content_type, "Content-Length": str(len(payload))},
    )
    with _OPENER.open(request, timeout=300) as response:
        if response.status not in (200, 201, 204):
            raise RuntimeError(f"upload rejected with HTTP {response.status}")


def _cleanup(workdir: Path) -> None:
    shutil.rmtree(workdir, ignore_errors=True)
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Handler
# --------------------------------------------------------------------------


def handler(job):
    with _JOB_LOCK:
        return _handle(job)


def _handle(job):
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
    # Where the matte goes. When the caller supplies a presigned PUT we use it
    # and touch no credentials; the env-configured bucket is the standalone
    # fallback for local testing.
    upload_url = job_input.get("upload_url")
    upload_key = job_input.get("matte_key")
    upload_content_type = str(job_input.get("upload_content_type", "video/mp4"))

    start_time = job_input.get("start_time")
    end_time = job_input.get("end_time")
    start_time = float(start_time) if start_time is not None else None
    end_time = float(end_time) if end_time is not None else None

    if start_time is not None and start_time < 0:
        return _rejected("invalid_range", "start_time cannot be negative")
    if end_time is not None and start_time is not None and end_time <= start_time:
        return _rejected(
            "invalid_range", f"end_time ({end_time}) must be after start_time ({start_time})"
        )

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

        # Identify the source before fetching it. A cache hit then costs one
        # ranged request instead of a full download — tens of seconds of billed
        # transfer on a 67 MB clip, for a result we already hold.
        identity = _remote_identity(video_url)
        key = _output_key(identity, params) if identity else None

        if key is not None and _already_done(key):
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

        content_hash = _download(video_url, source)
        if key is None:
            # The origin gave us nothing to identify it by, so fall back to
            # hashing the bytes: same guarantee, just paid for after the fact.
            key = _output_key(content_hash, params)
            if _already_done(key):
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

        if upload_url:
            _put_presigned(Path(export.matte_path), upload_url, upload_content_type)
            # The caller chose the key and owns the bucket, so it can presign a
            # GET itself — we have no credentials to do so and should not.
            stored_key, stored_url = upload_key, None
        else:
            stored_key, stored_url = key, _upload(Path(export.matte_path), key)

        return {
            "ok": True,
            # The key is the durable reference — store this. The URL is a
            # convenience for testing and expires after URL_TTL_SECONDS; callers
            # should re-presign from the key rather than persist the URL.
            "matte_key": stored_key,
            "matte_url": stored_url,
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

    except urllib.error.HTTPError as exc:
        hint = (
            " The presigned URL may have expired while the job sat in the queue — "
            "sign it for longer than the job's TTL."
            if exc.code in (401, 403)
            else ""
        )
        return _rejected("source_unavailable", f"could not fetch the video: HTTP {exc.code}.{hint}")
    except urllib.error.URLError as exc:
        return _rejected("source_unavailable", f"could not reach the video URL: {exc.reason}")
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
