import asyncio
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fast_api.worker import worker_loop
from fast_api.routes.jobs import router as jobs_router
from fast_api.routes.concepts import router as concepts_router
from fast_api.jobs_dao import ( init_db, create_job, get_job, list_jobs  )
from fast_api.event_bus import job_streams

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://localhost:3443",
        "http://localhost:3443",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(concepts_router)
app.include_router(jobs_router)



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
        job_id  = job_id,
        concept = concept,
    )
    return {
        "job_id": job_id,
        "status": "queued",
    }


@app.get("/jobs")
async def list_all_jobs():
    rows = list_jobs()

    return [
        {
            "job_id": row[0],
            "concept": row[1],
            "status": row[2],
        }
        for row in rows
    ]

@app.get("/jobs/{job_id}/events")
async def events(job_id: str):
    async def stream():
        q = job_streams.setdefault(job_id, asyncio.Queue())

        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=5)
                yield f"data: {json.dumps(event)}\n\n"

            except asyncio.TimeoutError:
                job = get_job(job_id)

                if job and job[2] in ("done", "error"):
                    break

    return StreamingResponse(stream(), media_type="text/event-stream")


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

