# Limitations

## Requirements to run

- A GPU is effectively required for practical runtime. CPU inference is
  impractically slow. Apple Silicon (MPS) works but is roughly 4x slower
  than a CUDA GPU (measured: ~35 min vs. ~9 min for the same tracking pass
  on a ~12s clip).
- `ffmpeg` must be on `PATH` (used for the `webm_alpha` export format; not
  needed for `fill_matte`).
- ~1.8GB of model weights (`grounding-dino-base` + `sam2.1-hiera-large`) are
  downloaded on first run if not already cached locally.
- Runtime scales with clip length (roughly linear, based on measurements so
  far). No hard duration cap is enforced by the code.

## Not supported

- Multiple subjects in one run — a prompt tracks a single detection (the
  largest-area match), not every match in frame.
- Prompts other than "person" are untested against real objects.
- Background removal/inpainting — the background output is always the
  untouched original video.
- Text/caption rendering or compositing — only the two raw video layers
  (person, background) are produced.
- Synchronous request/response — the API is submit-then-poll only; a single
  job can take 30-50+ minutes.
- Authentication, rate limiting, and automatic cleanup of old job
  directories.
- Horizontal scaling — job state is in-memory per process, and only one job
  runs at a time per process.
- GPU inside the Docker image — CPU-only by default. CUDA requires swapping
  in a CUDA-enabled `torch` build and running the container with
  `--gpus all`; not set up or verified.
- Small image: the default pip `torch` wheel bundles full CUDA runtime
  libraries even for CPU-only use, making the built image ~9.3GB. Using
  PyTorch's CPU-only wheel index would shrink this significantly; not done
  by default since it would need re-verifying against the GPU path too.

## Known quality limitations

- If the prompt doesn't match anything in any of the sampled seed-candidate
  frames (12 by default), the run fails outright — there's no fallback
  subject.
- The morphological mask-cleanup pass mitigates small occlusion gaps but
  doesn't guarantee none — one frame in the validated test clip still
  showed a visible hole where an object occluded the subject.
- Validated end-to-end on exactly one real clip so far (12s, vertical
  720×1280, single subject). No evidence yet on other resolutions, aspect
  ratios, lighting conditions, or multi-subject scenes.
- API error responses include raw exception text, not sanitized for public
  exposure.

## Output format notes

- `fill_matte` (default): two H.264 videos — person on black plus a
  grayscale matte. Works with any downstream tool via a standard track-matte
  / luma-matte operation (a built-in effect in Premiere Pro and DaVinci
  Resolve, or `ffmpeg -i fill.mp4 -i matte.mp4 -filter_complex alphamerge`
  in code).
- `webm_alpha` (opt-in): one file with a real alpha channel, but most
  default VP9 decoders — including plain `ffmpeg -i file.webm` — silently
  report it as fully opaque instead of erroring. Only use this format if the
  consumer has confirmed it decodes VP9 alpha correctly (requires forcing
  `-c:v libvpx-vp9` or equivalent).
