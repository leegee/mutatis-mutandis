import asyncio
from uuid import uuid4

from fastapi import FastAPI

from .jobs_db import ( init_db, create_job, get_job )
from .worker import worker_loop

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://localhost:3443",
        "http://localhost:3443",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():

    init_db()

    asyncio.create_task(
        worker_loop()
    )


@app.post("/jobs/run")
async def run_job(concept: str):

    job_id = str(uuid4())

    create_job(
        job_id=job_id,
        concept=concept,
    )

    return {
        "job_id": job_id,
        "status": "queued",
    }


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):

    row = get_job(job_id)

    if row is None:
        return {
            "status": "not_found",
        }

    return {
        "job_id": row[0],
        "concept": row[1],
        "status": row[2],
        "stage": row[3],
        "attempts": row[4],
        "created_at": row[5],
        "started_at": row[6],
        "finished_at": row[7],
        "last_heartbeat": row[8],
        "error": row[9],
    }
