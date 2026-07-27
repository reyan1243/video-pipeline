"""FastAPI wrapper around VideoLayerPipeline — the actual client deliverable.

One call in (a video + a prompt), two files out (person + background
layers). Processing happens in a background task; poll GET /jobs/{id} for
status, then download the finished artifacts.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.jobs import JobStore, run_job

DATA_DIR = Path(os.environ.get("VIDEO_PIPELINE_DATA_DIR", "data/jobs"))
VALID_PERSON_FORMATS = ("webm_alpha", "fill_matte")

app = FastAPI(title="video-layer-service API", version="0.1.0")
store = JobStore()


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    error: str | None = None
    person_url: str | None = None
    matte_url: str | None = None
    background_url: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs", response_model=JobCreatedResponse)
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile,
    prompt: str = Form(...),
    person_format: str = Form("fill_matte"),
    num_seed_candidates: int = Form(12),
    mask_close_kernel_size: int = Form(45),
) -> JobCreatedResponse:
    if person_format not in VALID_PERSON_FORMATS:
        raise HTTPException(400, f"person_format must be one of {VALID_PERSON_FORMATS}, got {person_format!r}")

    job_dir = DATA_DIR / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
    video_path = job_dir / f"input{suffix}"
    with video_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    job = store.create(
        prompt=prompt,
        person_format=person_format,
        num_seed_candidates=num_seed_candidates,
        mask_close_kernel_size=mask_close_kernel_size,
        video_path=video_path,
        output_dir=job_dir / "output",
    )
    background_tasks.add_task(run_job, job.id, store)
    return JobCreatedResponse(job_id=job.id, status=job.status)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")

    person_url = matte_url = background_url = None
    if job.status == "done" and job.result is not None:
        person_url = f"/jobs/{job_id}/download/person"
        background_url = f"/jobs/{job_id}/download/background"
        if job.result.matte_path is not None:
            matte_url = f"/jobs/{job_id}/download/matte"

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        error=job.error,
        person_url=person_url,
        matte_url=matte_url,
        background_url=background_url,
    )


@app.get("/jobs/{job_id}/download/{artifact}")
def download_artifact(job_id: str, artifact: str) -> FileResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status != "done" or job.result is None:
        raise HTTPException(409, f"job status is {job.status!r}, not done")

    path_by_artifact = {
        "person": job.result.person_path,
        "matte": job.result.matte_path,
        "background": job.result.background_path,
    }
    if artifact not in path_by_artifact:
        raise HTTPException(404, f"unknown artifact {artifact!r}")
    path = path_by_artifact[artifact]
    if path is None:
        raise HTTPException(404, f"{artifact!r} was not produced for this job's format")

    return FileResponse(path)
