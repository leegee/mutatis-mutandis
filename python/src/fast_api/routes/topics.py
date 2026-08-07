# fast_api/routes/topics.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4

from fast_api.jobs_dao import create_job   # adjust import if needed
from lib.corpus_logging import logger

router = APIRouter(prefix="/topic", tags=["topics"])

class TopicAnalysisRequest(BaseModel):
    concept: str
    documents: List[str]
    label: Optional[str] = None
    min_topic_size: int = 5
    use_sentence_chunking: bool = True


@router.post("/")
async def create_topic_analysis(request: TopicAnalysisRequest):
    """Queue a BERTopic + MacBERTh analysis job."""
    if not request.documents or len(request.documents) < 3:
        raise HTTPException(status_code=400, detail="At least 3 documents are required")

    job_id = str(uuid4())

    create_job(
        job_id=job_id,
        concept=request.concept,
        job_type="topic_analysis",
        payload={
            "documents": request.documents,
            "label": request.label,
            "min_topic_size": request.min_topic_size,
            "use_sentence_chunking": request.use_sentence_chunking,
        }
    )

    logger.info(f"Topic analysis job queued: {job_id} | concept: {request.concept}")

    return {
        "job_id": job_id,
        "status": "queued",
        "concept": request.concept,
        "document_count": len(request.documents)
    }
