from uuid import uuid4
from fastapi import APIRouter
from fast_api.models import RunJobRequest

from fast_api.jobs_dao import (
    create_job,
    get_job,
    list_jobs,
)


router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
)


@router.post("/run")
async def run_job(req: RunJobRequest):
    job_id = str(uuid4())

    create_job(
        job_id  = job_id,
        concept = req.concept,
    )

    return {
        "job_id": job_id,
        "status": "queued",
    }


@router.get("/list")
async def jobs_list():
    return list_jobs()


@router.get("/{job_id}")
async def job_status(job_id: str):
    row = get_job(job_id)

    if row is None:
        return {
            "status": "not_found",
        }

    return dict(row)
