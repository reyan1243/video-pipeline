# Optimisation plan — cost down, quality up, nothing compromised

Branch: `perf/cost-and-matte-quality`

## Target use case

Produce a **matte** (per-frame silhouette of the subject) so captions can be
composited *between* the background and the subject. The consumer is the Kalakar
web app (Remotion), which already has the source video loaded — so the only
artifact it actually needs is the matte.

## Baseline

| | Value |
|---|---|
| Measured throughput | ~16 s GPU per 1 s of 30fps video |
| Cost @ RunPod 4090 ($0.00031/s) | **~$0.298 per minute of video** |
| Theoretical floor (SAM2-large, 30fps in / ~25fps out) | ~1.3 s GPU per 1 s video |

The gap is **not** model compute. It is precision, disk I/O and encoding waste.

## The one rule

Every change lands in exactly one bucket, and the bucket decides whether it is
on by default:

- **A — Free.** Provably identical mask output. Pure waste removal. **Default on.**
- **B — Better.** Measurably improves the matte for this use case. **Default on.**
- **C — Trade-off.** Could change quality either way. **Opt-in flag, default off.**
- **D — Reliability.** No effect on a successful run's output. **Default on.**

Nothing in C is enabled by default, so a default run can only get faster or
better, never worse.

---

## Bucket A — Free speed (bit-identical masks)

| # | Change | File | Why it is free |
|---|---|---|---|
| A1 | Chunked reverse frame reader — one seek per 64-frame window instead of one seek per frame | `video_source.py`, `tracker.py:110` | Model sees the same frames in the same reverse order |
| A2 | Count frames with `capture.grab()` instead of `capture.read()` | `video_source.py:46` | `grab()` decodes but skips the numpy copy; count is identical |
| A3 | Threshold + scale the mask on GPU, transfer `uint8` | `tracker.py:132` | Same values, less PCIe traffic and no full-res sync |
| A4 | Free the seed-only SAM2 image model after seeding | `segmenter.py`, `pipeline.py` | Model is never used again after `select_seed` |
| A5 | `clean_masks` mutates in place | `mask_ops.py:38`, `pipeline.py:54` | Same result, halves peak RAM (no double dict) |
| A6 | One `ffmpeg` subprocess for encoding instead of two `cv2.VideoWriter`s in the Python loop | `exporter.py` | Encoding moves off the Python thread and is multithreaded |
| A7 | Early-exit seed search once a candidate scores ≥ 0.97 IoU | `segmenter.py:53` | Only stops when an excellent seed is already in hand |

## Bucket B — Better matte

| # | Change | File | Effect |
|---|---|---|---|
| B1 | Stop zeroing non-subject pixels in the fill | `exporter.py:117` | Removes the dark rim (measured up to 73 luma) caused by lossy encoding of a hard black step |
| B2 | Drop the `> 127` re-binarise; add a controlled Gaussian feather (default σ = 1.0 px, `0` disables) | `mask_ops.py` | Replaces a stair-stepped edge with a uniform 1 px soft edge |
| B3 | Close-kernel default 45 → 7 | `mask_ops.py`, `pipeline.py` | A 45 px close + fill-holes bridges arm-to-torso gaps and fills them solid — exactly the negative space captions need |
| B4 | Encode the matte H.264 CRF 18 `-tune grain`, not `mp4v` | `exporter.py:108` | `-tune grain` measured 3.2 dB better than plain CRF 18 on hard edges; `mp4v` is MPEG-4 Part 2 and has no quality knob |
| B5 | New default format `matte` — one grayscale file, no redundant fill/background copy | `exporter.py` | ~2.5 MB/min instead of two full videos plus a byte copy of the input |

## Bucket C — Opt-in trade-offs (all default off)

| # | Flag | Default | Trade |
|---|---|---|---|
| C1 | `--model-size {large,base-plus,small,tiny}` | `large` | tiny is ~1.6× faster, ~5 J&F points worse |
| C2 | `--max-height N` | `0` (off) | Matte rendered at reduced height; SAM2 resizes to 1024² internally so model quality is near-identical, but output resolution drops |
| C3 | `--temporal-smoothing` | off | 3-tap median across frames kills edge jitter, costs a frame of lag on fast motion |
| C4 | `--no-autocast` | autocast **on** for CUDA | On CUDA, bf16 is required for the pipeline to run at all (see D1) and is what SAM2's reference implementation uses; the flag is an escape hatch |

## Bucket D — Reliability

| # | Change | File |
|---|---|---|
| D1 | Wrap inference in `torch.autocast(bfloat16)` on CUDA — fixes the `mat1 and mat2 must have the same dtype` crash that makes CUDA unusable today | `tracker.py:128`, `segmenter.py:38`, `detector.py:49` |
| D2 | Cap segment length + `--cpu-offload` — removes the undocumented ~900-frame (33 s) VRAM cliff | `tracker.py:96` |
| D3 | `max_segments_per_direction` 8 → 20, exposed on the CLI | `datatypes.py:81`, `pipeline.py` |
| D4 | Explicit coverage check with an actionable message instead of a raw `ValueError` listing frame indices | `exporter.py:51` |
| D5 | Detect encoder failure instead of silently producing a 0-byte file | `exporter.py:111` |
| D6 | Fix the `webm_alpha` stderr deadlock (`-nostats`, drained pipe) | `exporter.py:89` |
| D7 | Guard `num_seed_candidates < 2`, `frame_count == 0`, non-finite fps | `segmenter.py:49`, `video_source.py` |
| D8 | Pin `torch==2.6.0` / `torchvision==0.21.0`, `transformers>=4.56,<5` | `requirements.txt` |
| D9 | Correct README/LIMITATIONS: codec is not H.264; document the VRAM ceiling | `README.md`, `docs/LIMITATIONS.md` |

---

## Expected outcome

| | GPU-s per min of video | Cost/min @ 4090 |
|---|---|---|
| Today | ~960 | ~$0.298 |
| After A + D1 | ~80 (estimate) | ~$0.025 |
| With C1 `tiny` (opt-in) | ~50 (estimate) | ~$0.016 |

The 960 s figure is measured. The 80 s and 50 s figures are **estimates derived
from published SAM2 throughput** and must be confirmed by a real run before they
are quoted anywhere.

## Verification

- `tests/` runs offline for everything that does not need torch — mask ops,
  video source, exporter, datatypes.
- New tests cover: feathering, in-place cleanup, reverse-window frame ordering,
  encoder failure detection, and the coverage guard.
- The GPU path (autocast, VRAM ceiling, throughput) **cannot** be verified on
  this machine and needs a RunPod run.
