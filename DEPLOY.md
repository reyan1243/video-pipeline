# Deploying to RunPod Serverless

Everything here was checked against docs.runpod.io in **August 2026**. Where the
docs contradict themselves, the conflict is called out rather than hidden.

---

## 0. What you need first

| Thing | Why |
|---|---|
| RunPod account + **API key** | Settings → API Keys |
| Cloudflare **R2 bucket** + access key/secret | The worker uploads the matte and returns a presigned URL. R2 has zero egress fees, which matters because end users download these files. |
| GitHub account connected to RunPod *(Path A only)* | So RunPod can build the image for you |
| Docker running *(Path B only)* | Only if you take the fallback path |

The `/run` payload cap is **10 MB** and results are retained only **30 minutes**,
so the video goes in as a **URL** and the matte comes back as a **URL**. Neither
ever travels through the RunPod API as bytes.

---

## Path A — Let RunPod build it (recommended, no Docker on your machine)

RunPod clones the repo, builds the image on its own infrastructure, and deploys
it. You never build or push a 9 GB image from a laptop.

**This repo is compatible with the hosted build.** The three things that
disqualify a project from Path A don't apply here: our base image
(`python:3.11-slim`) is public, we need no `--build-arg`, and nothing needs a GPU
at build time (weights are only *downloaded*, not compiled).

1. **Connect GitHub** — RunPod console → **Settings** → **Connections** →
   **GitHub** → **Connect**. Choose *All repositories* or *Only select
   repositories* and pick `kalakar-es/video-pipeline`. Private repos work.
   *(Only one GitHub account can be connected per RunPod account, and it isn't
   shared with teammates.)*
2. **Serverless** → **New Endpoint** → the **GitHub** tile.
   *(Older doc pages still say "Import Git Repository". Those pages are stale —
   the six-tile flow shipped 2026-07-01.)*
3. Select `kalakar-es/video-pipeline`, then set:
   - **Branch**: `perf/cost-and-matte-quality`
   - **Dockerfile Path**: `Dockerfile.serverless` ← **required.** The repo also
     contains a plain `Dockerfile` for the old FastAPI app; leaving this blank
     builds the wrong thing.
4. Fill in the endpoint settings from §2 below → **Deploy**.
5. Watch the **Builds** tab: Pending → Building → Uploading → Testing → Completed.

**Build limits:** 30 minutes for the docker build step, 2.5 h total, 80 GB image.
Our build downloads ~2.5 GB of torch plus ~1.8 GB of weights, so it should fit —
but if it times out, take Path B.

> **Updates do not auto-deploy on push.** You must cut a **GitHub release** to
> trigger a rebuild. RunPod's own launch blog says every push rebuilds; that is
> out of date, the docs are correct.

---

## Path B — Build and push yourself (fallback)

```bash
cd /Users/apple/Desktop/Kalakar/video_pipeline

# --platform is MANDATORY. Without it an Apple Silicon Mac produces an arm64
# image that pushes fine and then dies at worker start with "exec format error".
docker buildx build --platform linux/amd64 --push \
  -f Dockerfile.serverless \
  -t docker.io/YOUR_DOCKERHUB_USER/video-matte:v1 .
```

Then **New Endpoint** → **Docker** tile → Container Image
`docker.io/YOUR_DOCKERHUB_USER/video-matte:v1`.

Never tag `:latest` — mutable tags mean different workers silently run different
code. Use `v1`, `v2`, …

---

## 1. Environment variables

Set these on the endpoint (Path A) or the template (Path B).

| Variable | Required | Value |
|---|---|---|
| `S3_BUCKET` | yes | your R2 bucket name |
| `S3_ENDPOINT` | yes | `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` |
| `S3_ACCESS_KEY_ID` | yes | R2 access key |
| `S3_SECRET_ACCESS_KEY` | yes | R2 secret |
| `S3_REGION` | no | `auto` (default) — correct for R2 |
| `S3_PREFIX` | no | `mattes` |
| `URL_TTL_SECONDS` | no | `86400` — presigned URL lifetime |
| `MAX_DURATION_SECONDS` | no | `300` |
| `MAX_MASK_BYTES` | no | `8589934592` (8 GB host-RAM guard) |
| `MAX_SEGMENT_FRAMES` | no | `600` — caps VRAM growth |
| `MODEL_SIZE` | no | `large` \| `base-plus` \| `small` \| `tiny` |

After the first deploy, **verify the credentials actually arrived**. RunPod's
`{{ RUNPOD_SECRET_name }}` substitution is documented for Pods only; if it
doesn't apply to Serverless you receive the *literal string* as your key. Check
the worker log for a value starting with `{{`.

---

## 2. Endpoint settings

| Field | Value | Why |
|---|---|---|
| Endpoint type | **Queue** | Load-balancing endpoints cap at ~5.5 min of processing |
| GPU configuration | **RTX 4090 → L40S → A6000** (3 in priority order) | One type alone means waiting on one pool |
| Active workers | **0** | This is what makes it scale to zero |
| Max workers | **3–5** | Never 1 — the queue cap is `max_workers × 100` |
| GPUs per worker | **1** | |
| Idle timeout | **60 s** *(default is 5 s)* | 5 s means a full cold start between nearly every job in a burst |
| Execution timeout | **900 s** *(default is 600 s)* | 50% headroom over a 10-minute job |
| Job TTL | **86400 s** | Clock starts at **submission** and includes queue time |
| FlashBoot | **on** | Free; snapshots the worker for ~7 s warm restarts |
| Container disk | **50 GB** *(default 20 GB)* | Ephemeral scratch; the image alone is ~9 GB |
| Data centers | **all** | Restricting shrinks the GPU pool |
| Network volume | **none** | Pins the endpoint to one datacenter |

---

## 3. Test the handler locally (no GPU needed)

```bash
# Fill in a real URL first
$EDITOR test_input.json

python rp_handler.py --test_input "$(cat test_input.json)"
```

Exit code 0 = pass. Never put `--test_input` in the production `CMD` — the SDK
treats it as "run locally", so the worker would run one fake job and exit instead
of polling the queue. `test_input.json` is in `.dockerignore` for the same reason.

---

## 4. Calling it

Job submission is on `api.runpod.**ai**`. Account/billing is `api.runpod.**io**`.
Mixing them up produces 404s that look like a bad endpoint ID.

### Submit

```bash
export RUNPOD_API_KEY="..."
export ENDPOINT_ID="..."

curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/run \
  -H "authorization: Bearer $RUNPOD_API_KEY" \
  -H "content-type: application/json" \
  -d '{
        "input": {
          "video_url": "https://your-r2/clip.mp4",
          "prompt": "person",
          "start_time": 0,
          "end_time": 30
        },
        "policy": { "executionTimeout": 900000, "ttl": 3600000 }
      }'
# -> {"id":"eaebd6e7-...","status":"IN_QUEUE"}
```

`input`, `webhook` and `policy` are **top-level siblings**. **`policy` is in
milliseconds** while the console is in seconds — an off-by-1000 here is a silent
10-minutes-vs-10-seconds bug.

| Input field | Default | Notes |
|---|---|---|
| `video_url` | — | **required**, must be publicly fetchable by the worker |
| `prompt` | `"person"` | open-vocabulary: `"dog"`, `"woman in red jacket"` |
| `start_time` / `end_time` | whole clip | **the main cost lever** — you are billed per GPU-second, so process only the range you need |
| `num_seed_candidates` | `12` | raise if detection is flaky |
| `mask_close_kernel_size` | `7` | large values swallow arm-to-torso gaps |
| `feather_sigma` | `1.0` | edge softness in px; `0` = hard edge |
| `temporal_smoothing` | `false` | removes edge jitter, costs a frame of lag |

### Poll

```bash
curl https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JOB_ID \
  -H "authorization: Bearer $RUNPOD_API_KEY"
```

```jsonc
{"id":"...","status":"IN_QUEUE"}
{"id":"...","status":"IN_PROGRESS","output":"tracking 900 frames"}   // progress, NOT the result
{"id":"...","status":"COMPLETED","output":{ ... }}
{"id":"...","status":"FAILED","error":"..."}
```

**Gate on `status === "COMPLETED"`, never on the presence of `output`** —
progress updates put a string in `output` mid-flight. A **404 is terminal**
(TTL expired or the 30-minute retention window lapsed), not a retryable blip.

### Success

```json
{
  "ok": true,
  "matte_url": "https://...person_matte.mp4?X-Amz-Signature=...",
  "cached": false,
  "frames": 900,
  "fps": 30.0,
  "width": 720,
  "height": 1280,
  "seed_frame": 412,
  "seed_iou": 0.981,
  "coverage": 1.0,
  "processing_seconds": 41.2,
  "url_expires_in_seconds": 86400
}
```

`cached: true` means the identical video and settings were processed before and
the existing file was returned — no GPU time, no charge.

### Expected failure — still `COMPLETED`

```json
{ "ok": false, "code": "too_long", "reason": "clip is 412.0s, limit is 300s. ..." }
```

Returning a truthy `"error"` key makes RunPod mark the job **FAILED**, which is
indistinguishable from a crash — so caller-actionable problems come back as
`ok: false` instead. **Check `output.ok`.**

| `code` | Meaning |
|---|---|
| `missing_input` | no `video_url` |
| `too_long` | over `MAX_DURATION_SECONDS` — pass `start_time`/`end_time` |
| `too_large` | over the mask-RAM guard — downscale or shorten |
| `incomplete_coverage` | subject was lost — better prompt, or split the clip |
| `invalid_input` | unfetchable URL, oversized file, unusable frame rate |
| `ffmpeg_failed` | trim or probe failed |

`status: "FAILED"` means the worker genuinely crashed.

### Cancel

```bash
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/cancel/$JOB_ID \
  -H "authorization: Bearer $RUNPOD_API_KEY"
```

POST, not DELETE. Rate-limited to 100 req/10 s — 10× tighter than `/run`. Call
this when a user abandons a job; you are billed for consumed time either way, so
the sooner the better.

---

## 5. Webhooks

Add `"webhook": "https://you.example/rp-hook/<UNGUESSABLE_TOKEN>"` alongside
`input`. RunPod POSTs on completion.

**Treat it as a latency optimisation, never as a delivery guarantee:** there are
**3 attempts total, 10 s apart, no backoff, no dead-letter**, and the payload is
**not signed** — no HMAC header exists. The body shape is undocumented.

Safe receiver design:

1. Put a high-entropy token in the URL path; reject anything else.
2. Read **only** `body.id`, ignore every other field.
3. Confirm you submitted that id.
4. `GET /status/<id>` for authoritative state.
5. Return 200 immediately.
6. **Keep a polling fallback** — three failed deliveries drop the job silently.

---

## 6. Reference client

```ts
const RUNPOD = `https://api.runpod.ai/v2/${process.env.RUNPOD_ENDPOINT_ID}`
const headers = {
  authorization: `Bearer ${process.env.RUNPOD_API_KEY}`,
  'content-type': 'application/json',
}

export async function generateMatte(
  videoUrl: string,
  { prompt = 'person', startTime, endTime, onProgress }: {
    prompt?: string; startTime?: number; endTime?: number
    onProgress?: (stage: string) => void
  } = {}
) {
  const submit = await fetch(`${RUNPOD}/run`, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      input: { video_url: videoUrl, prompt, start_time: startTime, end_time: endTime },
      policy: { executionTimeout: 900_000, ttl: 3_600_000 }, // milliseconds
    }),
  })
  if (!submit.ok) throw new Error(`submit failed: ${submit.status}`)
  const { id } = await submit.json()

  while (true) {
    await new Promise((r) => setTimeout(r, 5000))
    const response = await fetch(`${RUNPOD}/status/${id}`, { headers })

    // Terminal: the job was deleted (TTL expired, or retention lapsed).
    if (response.status === 404) throw new Error('job expired')

    const job = await response.json()

    // `output` is populated by progress updates too — only COMPLETED is done.
    if (job.status === 'IN_PROGRESS' && typeof job.output === 'string') {
      onProgress?.(job.output)
    }

    if (job.status === 'COMPLETED') {
      if (job.output?.ok === false) {
        throw Object.assign(new Error(job.output.reason), { code: job.output.code })
      }
      return job.output as { matte_url: string; width: number; height: number }
    }

    if (['FAILED', 'TIMED_OUT', 'CANCELLED'].includes(job.status)) {
      throw new Error(`job ${job.status}: ${job.error ?? ''}`)
    }
  }
}
```

---

## 7. Things that will bite you

1. **Missing `--platform linux/amd64`** (Path B) → `exec format error` at worker start.
2. **Execution timeout** — console default 600 s; if you create via the REST API and omit it, the default is **300 s**.
3. **Idle timeout 5 s** — burns a cold start between nearly every job. Raise to 60 s.
4. **`policy` is milliseconds, the console is seconds.**
5. **`output` present ≠ finished.** Gate on `status`.
6. **404 on `/status` is terminal**, not transient.
7. **Container disk is ephemeral and 20 GB by default** — the image alone is ~9 GB.
8. **Do not use `refresh_worker`** — it wipes worker state and forces a full model reload, undoing the singleton loading that makes warm jobs fast.
9. **Set a spend limit.** The account default is **$80/hour**.
10. **Cold start is billed** — container init *and* loading weights into GPU memory. Docs are authoritative here; several vendor blog posts claim otherwise.
