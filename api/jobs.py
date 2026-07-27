"""Background job tracking for the FastAPI service.

Jobs run one at a time per process (see _PIPELINE_LOCK) — a deliberate
simplification for a single-instance initial deployment, not a scale limit
baked into the design. Job state lives in memory, so it doesn't survive a
restart and doesn't coordinate across instances either way — scaling to
concurrent processing means running multiple instances behind a load
balancer, or later, swapping this for a real task queue + shared store.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pipeline import VideoLayerPipeline
from datatypes import ExportResult

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "failed"]

# ponytail: one process-wide lock serializes pipeline runs so concurrent
# submissions don't load multiple copies of the models into memory/GPU at
# once. Upgrade path: a real task queue (or multiple replicas) if throughput
# needs to scale past one job at a time.
_PIPELINE_LOCK = threading.Lock()


@dataclass
class Job:
    id: str
    status: JobStatus
    created_at: float
    prompt: str
    person_format: str
    num_seed_candidates: int
    mask_close_kernel_size: int
    video_path: Path
    output_dir: Path
    error: str | None = None
    result: ExportResult | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        prompt: str,
        person_format: str,
        num_seed_candidates: int,
        mask_close_kernel_size: int,
        video_path: Path,
        output_dir: Path,
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            status="queued",
            created_at=time.time(),
            prompt=prompt,
            person_format=person_format,
            num_seed_candidates=num_seed_candidates,
            mask_close_kernel_size=mask_close_kernel_size,
            video_path=video_path,
            output_dir=output_dir,
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in changes.items():
                setattr(job, key, value)


def run_job(job_id: str, store: JobStore) -> None:
    job = store.get(job_id)
    if job is None:
        return

    with _PIPELINE_LOCK:
        store.update(job_id, status="running")
        try:
            pipeline = VideoLayerPipeline(
                video_path=job.video_path,
                prompt=job.prompt,
                output_dir=job.output_dir,
                num_seed_candidates=job.num_seed_candidates,
                mask_close_kernel_size=job.mask_close_kernel_size,
                person_format=job.person_format,
            )
            result = pipeline.run()
            store.update(job_id, status="done", result=result.export)
        except Exception as exc:
            logger.exception("job %s failed", job_id)
            store.update(job_id, status="failed", error=str(exc))
