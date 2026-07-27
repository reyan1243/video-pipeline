#!/usr/bin/env bash
# Runs the FastAPI service locally with the project's venv.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
