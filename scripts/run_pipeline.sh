#!/usr/bin/env bash
# Thin passthrough to the CLI, for running the pipeline directly without the
# API — e.g. ./scripts/run_pipeline.sh clip.mp4 --prompt person --output-dir out/
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m pipeline "$@"
