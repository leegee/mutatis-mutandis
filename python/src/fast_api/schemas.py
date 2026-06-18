from pydantic import BaseModel


class JobResponse(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    concept: str
    status: str
    stage: str | None = None
    error: str | None = None
