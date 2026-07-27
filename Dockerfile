FROM python:3.12-slim

# ffmpeg is required by exporter.py (both output formats shell out to it);
# the rest of the video/ML stack is pure pip dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
# --default-timeout / --retries: the large ML wheels here (torch, nvidia-*)
# make a plain pip install prone to failing on a single stalled chunk read on
# a slow connection. The cache mount matters just as much: without it, a
# failed install on attempt N discards everything and attempt N+1 re-downloads
# from zero — on a genuinely flaky connection that's a guaranteed retry loop.
# With it, already-downloaded wheels persist across build attempts.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=120 --retries 5 -r requirements.txt

COPY . .

# Job uploads + outputs live here — mount a volume in production so results
# survive a container restart (job state itself is in-memory and won't).
ENV VIDEO_PIPELINE_DATA_DIR=/data/jobs
VOLUME ["/data/jobs"]

EXPOSE 8000

# python -m (not the bare `uvicorn` entry point) guarantees the working
# directory is on sys.path, so the flat top-level modules import correctly.
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
