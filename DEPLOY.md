# Step 1 — Deploy the masking endpoint

Everything verified against docs.runpod.io in August 2026, and against real runs
on a GPU pod. Where RunPod's own docs contradict each other, the conflict is
called out rather than hidden.

**What this endpoint does:** takes a video URL, returns **one** grayscale matte
(~3.4 MB/min) uploaded to your R2 bucket. Nothing else — no person layer, no
background copy. Both were measured as redundant: the subject's colour is the
source video the browser already has.

---

## 0. Before you start

| Need | Where |
|---|---|
| RunPod API key | Console → Settings → API Keys |
| **Spend limit** | Console → Settings → Billing. **Set this first** — the account default is $80/hour. |
| R2 bucket + access key/secret | Cloudflare dashboard. Reuse the app's bucket; scope the key to the `mattes/` prefix. |
| R2 account id | The `<ACCOUNT_ID>` in `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` |

---

## 1. Connect GitHub

RunPod builds the image itself — no Docker, no multi-GB upload from your laptop.

Console → **Settings → Connections → GitHub → Connect** → grant access to
`kalakar-es/video-pipeline`. Private repos work.

> Only one GitHub account can be connected per RunPod account, and it is not
> shared with teammates.

This repo is compatible with the hosted build: the base image
(`python:3.11-slim`) is public, no `--build-arg` is needed, and nothing requires
a GPU at build time — the three things that would otherwise disqualify it.

---

## 2. Create the endpoint

**Serverless → New Endpoint → the `GitHub` tile.**

*(Older doc pages still say "Import Git Repository". They are stale — the
six-tile flow shipped 2026-07-01.)*

| Field | Value |
|---|---|
| Repository | `kalakar-es/video-pipeline` |
| Branch | `perf/cost-and-matte-quality` |
| **Dockerfile Path** | **`Dockerfile.serverless`** |

⚠️ **The Dockerfile path is not optional.** The repo also contains a plain
`Dockerfile` for the old FastAPI app; leaving this blank builds the wrong image.

---

## 3. Endpoint configuration

Six of these differ from the defaults. Each one is a real failure otherwise.

| Field | Value | Why |
|---|---|---|
| Endpoint type | **Queue** | Load-balancing endpoints cap at ~5.5 min of processing; our jobs run 2–5 min |
| GPU | **RTX 4090** → L40S → A6000 | Measured: a faster GPU costs the *same per job* (price scales with speed), so this is about availability, not speed. Three types = three pools to draw from. |
| Active workers | **0** | Scale to zero. A single always-on worker is ~$790/mo. |
| Max workers | **3** | Queue capacity is `max_workers × 100` |
| GPUs per worker | **1** | Tracking is sequential; a second GPU idles |
| **Idle timeout** | **60 s** *(default 5)* | 5s means a full cold start between nearly every job in a burst |
| **Execution timeout** | **900 s** *(default 600)* | Headroom over the worst case |
| Job TTL | **86400 s** | Clock starts at *submission* and includes queue time |
| **Container disk** | **50 GB** *(default 20)* | Ephemeral scratch; the image alone is ~9 GB |
| FlashBoot | **on** | Free. Snapshots the warmed worker → ~7s restarts instead of ~70s |
| Data centers | **all** | Restricting shrinks the GPU pool |
| Network volume | **none** | Pins the endpoint to one datacenter |

### On "load the models at deploy"

There is no deploy-time warm hook, and you do not need one. Models load at
**module import** in `rp_handler.py`, so a worker loads them **once when it
starts** and then serves job after job with them resident:

```
cold worker : ~50s imports + ~20s model load, then N jobs at full speed
warm worker : 0s — the job starts immediately
FlashBoot   : ~7s, restoring an already-warmed process
```

Measured on a real 46s clip: **97s of actual work**, versus 70s of one-time
startup. Keeping a worker permanently warm (Active workers = 1) would remove
that 70s from the first request of an idle period — for ~$790/month. Not worth
it until traffic is steady; revisit above ~25% utilisation.

This is also why the handler deliberately does **not** call
`segmenter.release()` and does **not** set `refresh_worker` — both would throw
the resident models away and force a reload on the next job.

---

## 4. Environment variables

**The worker does not need storage credentials in production.** Your projects
carry their own `storage_provider` (R2/S3/B2) and R2 uploads can fail over to
S3, so the destination is the API's decision, not the worker's. The API presigns
a PUT and passes it as `upload_url`; the worker writes that one object and can
reach nothing else in any bucket.

The variables below are the **standalone fallback**, used only when no
`upload_url` is supplied — i.e. local testing.

| Variable | Value |
|---|---|
| `S3_BUCKET` | your R2 bucket |
| `S3_ENDPOINT` | `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` |
| `S3_ACCESS_KEY_ID` | R2 access key |
| `S3_SECRET_ACCESS_KEY` | R2 secret |
| `S3_REGION` | `auto` |
| `S3_PREFIX` | `mattes` |
| `BF16_WEIGHTS` | `1` — measured 22.0s → 16.8s model load, and halves resident VRAM |

Optional, all with sane defaults: `MAX_DURATION_SECONDS` (300),
`MAX_MASK_HEIGHT` (0 = source resolution), `MODEL_SIZE` (`large`),
`MAX_SEGMENT_FRAMES` (600), `URL_TTL_SECONDS` (86400).

> After the first deploy, check the worker log for a credential starting with
> `{{`. RunPod's `{{ RUNPOD_SECRET_name }}` substitution is documented for Pods
> only; if it does not apply to Serverless you receive the literal string.

Then **Deploy**, and watch the **Builds** tab: Pending → Building → Uploading →
Testing → Completed. First build is ~15–25 min (it downloads ~2.5 GB of torch
and ~1.8 GB of weights). Limits are 30 min for the build step and 80 GB image.

---

## 5. Verify

```bash
export RUNPOD_API_KEY="..."
export ENDPOINT_ID="..."

# Health — should report zero workers, zero jobs
curl -s https://api.runpod.ai/v2/$ENDPOINT_ID/health \
  -H "authorization: Bearer $RUNPOD_API_KEY"

# Submit a 10s slice; keeps the first (cold) run to a few cents
curl -s -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/run \
  -H "authorization: Bearer $RUNPOD_API_KEY" -H "content-type: application/json" \
  -d '{"input":{"video_url":"https://<presigned-r2-url>","prompt":"person",
        "start_time":0,"end_time":10},
       "policy":{"executionTimeout":900000,"ttl":3600000}}'
# -> {"id":"...","status":"IN_QUEUE"}

curl -s https://api.runpod.ai/v2/$ENDPOINT_ID/status/<JOB_ID> \
  -H "authorization: Bearer $RUNPOD_API_KEY"
```

Job submission is `api.runpod.**ai**`; account/billing is `api.runpod.**io**`.
Mixing them gives 404s that look like a bad endpoint id.

**`policy` is milliseconds. The console is seconds.** Off by 1000 here is a
silent 10-minutes-vs-10-seconds timeout.

### Success

```json
{
  "status": "COMPLETED",
  "output": {
    "ok": true,
    "matte_key": "mattes/9f2c…/person_matte.mp4",
    "matte_url": "https://…?X-Amz-Signature=…",
    "cached": false,
    "frames": 300, "fps": 29.97, "width": 1080, "height": 1920,
    "seed_frame": 0, "seed_iou": 0.992, "coverage": 1.0,
    "processing_seconds": 41.2
  }
}
```

**Store `matte_key`, not `matte_url`.** The URL expires after
`URL_TTL_SECONDS`; the key is permanent and you re-presign from it.

`cached: true` means this exact video and settings were processed before and the
existing object was returned — no GPU time, no charge. Keys are
`sha256(video bytes + params)`, which also makes RunPod's heartbeat requeue
(which genuinely can run a handler twice) harmless.

### Expected failure — still `COMPLETED`

```json
{"ok": false, "code": "too_long", "reason": "clip is 412.0s, limit is 300s…"}
```

A truthy `"error"` key marks the whole RunPod job **FAILED**, indistinguishable
from a crash — so caller-actionable problems come back as `ok: false` instead.
**Check `output.ok`.** `status: "FAILED"` means the worker genuinely died.

| `code` | Meaning |
|---|---|
| `missing_input` | no `video_url` |
| `invalid_range` | `start_time` negative, or `end_time` <= `start_time` |
| `source_unavailable` | the URL 4xx'd or was unreachable. **A 401/403 usually means the presigned URL expired while the job sat in the queue** — sign it for longer than the job TTL. |
| `too_long` | over `MAX_DURATION_SECONDS` — pass `start_time`/`end_time` |
| `too_large` | over the mask-RAM guard — downscale or shorten |
| `incomplete_coverage` | subject lost — better prompt, or split the clip |
| `invalid_input` | unfetchable URL, oversized file, unusable frame rate |
| `ffmpeg_failed` | trim or probe failed |

### Cancel

```bash
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/cancel/<JOB_ID> \
  -H "authorization: Bearer $RUNPOD_API_KEY"
```

POST, not DELETE. You are billed for GPU time already consumed either way, so
cancel early when a user abandons.

---

## 6. Inputs

| Field | Default | Notes |
|---|---|---|
| `video_url` | — | **required**, presigned GET. Redirects are refused. |
| `upload_url` | — | presigned PUT for the matte. Supply this and the worker needs no credentials. |
| `matte_key` | — | the key `upload_url` points at; echoed back for the job row |
| `upload_content_type` | `video/mp4` | must match what `upload_url` was signed with |
| `prompt` | `"person"` | open-vocabulary: `"dog"`, `"woman in red jacket"` |
| `start_time` / `end_time` | whole clip | **the cost lever** — you pay per GPU-second |
| `num_seed_candidates` | `12` | stops early at IoU ≥ 0.97; usually costs 1 |
| `mask_close_kernel_size` | `7` | larger values swallow arm-to-torso gaps |
| `feather_sigma` | `1.0` | edge softness in px; `0` = hard edge |
| `temporal_smoothing` | `false` | measured jitter is already 0.2–0.65px, so leave off |
| `max_mask_height` | `0` | `960` on a 1080p source quarters cleanup/encode for ~no quality loss |

---

## 7. Expected cost

Measured on a 46s 1080×1920 clip, warm worker:

| | |
|---|---|
| Work per job | **97s** (86s tracking, 11s everything else) |
| Throughput | 2.1× realtime |
| **Cost** | **~$0.039 per minute of video** (RTX 4090 @ $0.00031/s) |
| Cold start | +70s when no worker is warm, ~7s with a FlashBoot hit |

Tracking is 89% of the work and is model-bound. The only remaining lever is
`MODEL_SIZE=base-plus` (~1.4× faster, measurably less accurate) — a real quality
trade, worth eyeballing on one clip before adopting.

---

## 8. Testing without redeploying

The same handler runs locally and loads models once, so you can iterate against
it as many times as you like for the cost of the processing only:

```bash
export S3_BUCKET=… S3_ENDPOINT=… S3_ACCESS_KEY_ID=… S3_SECRET_ACCESS_KEY=… S3_REGION=auto
python rp_handler.py --rp_serve_api          # loads once, listens on :8000

curl -X POST http://localhost:8000/runsync \
  -H "content-type: application/json" \
  -d '{"input":{"video_url":"https://…","prompt":"person"}}'
```

This is the exact code RunPod runs, so it also validates your R2 credentials and
the upload path before anything is deployed.

Never put `--test_input` or `--rp_serve_api` in the production `CMD` — the SDK
treats either as "run locally", so the worker would process one fake job and
exit instead of polling the queue. `test_input.json` is in `.dockerignore` for
the same reason.

---

## 9. Things that will bite you

1. **Blank Dockerfile Path** builds the old FastAPI image.
2. **Idle timeout 5s** burns a cold start between nearly every job.
3. **`policy` is ms, console is seconds.**
4. **`output` present ≠ job finished** — progress updates put a string in `output`. Gate on `status`.
5. **404 on `/status` is terminal**, not transient — TTL expired or the 30-min result-retention window lapsed.
6. **Cold start is billed.** Docs are authoritative; several vendor blogs claim otherwise.
7. **Updates need a GitHub release**, not a push. RunPod's launch blog says otherwise and is stale.
8. **Container disk is ephemeral** and wiped on restart.
