#!/usr/bin/env bash
# Builds the Docker image and runs it, mounting ./data/jobs so job
# uploads/outputs survive a container restart. CPU-only by default — see
# README.md for the GPU-enabled variant.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE_NAME="video-pipeline"

docker build -t "$IMAGE_NAME" .
mkdir -p data/jobs
docker run --rm -p 8000:8000 -v "$(pwd)/data/jobs:/data/jobs" "$IMAGE_NAME"
