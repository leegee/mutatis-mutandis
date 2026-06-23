from uuid import uuid4
from fastapi import APIRouter, HTTPException

from lib.eebo_config import CORPUS_TIER2_DB_PATH
from fast_api.jobs_dao import create_job
from fast_api.models import CreateConceptRequest, CreateConceptAndRunRequest, RunJobRequest
from fast_api.connections import get_corpus_tier2_conn
from tier2_0_concept_events import sqlite3_connection as events_sqlite3_connection

# con = events_sqlite3_connection(CORPUS_TIER2_DB_PATH)

router = APIRouter(
    prefix="/concepts",
    tags=["concepts"],
)


def load_concepts():
    conn = get_corpus_tier2_conn()

    rows = conn.execute("""
        SELECT concept, form, is_false_positive
        FROM concept_forms
        ORDER BY concept, form
    """).fetchall()

    result = {}

    for concept, form, is_fp in rows:
        bucket = result.setdefault(concept, {
            "forms": [],
            "false_positives": [],
        })

        if is_fp:
            bucket["false_positives"].append(form)
        else:
            bucket["forms"].append(form)

    return result


    conn = get_corpus_tier2_conn()
    rows = conn.execute("""
        SELECT concept, form
        FROM concept_forms
        ORDER BY concept, form
    """).fetchall()

    result = {}
    for concept, form in rows:
        result.setdefault(
            concept,
            {
                "forms": [],
                "false_positives": [],
            },
        )["forms"].append(form)

    return result


def save_concepts(data):
    conn = get_corpus_tier2_conn()

    try:
        conn.execute("BEGIN")

        for concept, payload in data.items():
            concept = concept.strip().upper()

            forms = payload.get("forms") or []
            fps = payload.get("false_positives") or []

            # ensure concept exists
            conn.execute(
                """
                INSERT INTO concepts (concept, n_events)
                VALUES (?, COALESCE((SELECT n_events FROM concepts WHERE concept=?), 0))
                ON CONFLICT(concept)
                DO UPDATE SET n_events = excluded.n_events
                """,
                (concept, concept),
            )

            # normal forms
            for form in set(forms):
                form = form.strip()
                if not form:
                    continue

                conn.execute(
                    """
                    INSERT INTO concept_forms (concept, form, is_false_positive)
                    VALUES (?, ?, 0)
                    ON CONFLICT(concept, form)
                    DO UPDATE SET is_false_positive = 0
                    """,
                    (concept, form),
                )

            # false positives override anything else
            for fp in set(fps):
                fp = fp.strip()
                if not fp:
                    continue

                conn.execute(
                    """
                    INSERT INTO concept_forms (concept, form, is_false_positive)
                    VALUES (?, ?, 1)
                    ON CONFLICT(concept, form)
                    DO UPDATE SET is_false_positive = 1
                    """,
                    (concept, fp),
                )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


@router.get("/list")
async def concepts_list():
    return load_concepts()


@router.post("/create_and_run")
async def create_and_run(req: CreateConceptAndRunRequest):
    concept_name = req.concept.upper().strip()

    if not concept_name:
        raise HTTPException(status_code=400, detail="Concept name required")

    concepts = load_concepts()

    # upsert — merge new forms/fps into existing ones
    existing = concepts.get(concept_name, {"forms": [], "false_positives": []})

    concepts[concept_name] = {
        "forms": sorted(set(existing["forms"]) | set(req.forms)),
        "false_positives": sorted(set(existing["false_positives"]) | set(req.false_positives)),
    }

    save_concepts(concepts)

    job_id = str(uuid4())
    create_job(job_id=job_id, concept=concept_name)

    return {
        "ok": True,
        "concept": concept_name,
        "job_id": job_id,
        "status": "queued",
    }


@router.post("/create")
async def create_concept(req: CreateConceptRequest):

    concepts = load_concepts()

    concept_name = req.concept.upper().strip()

    if not concept_name:
        raise HTTPException(
            status_code=400,
            detail="Concept name required",
        )

    if concept_name in concepts:
        raise HTTPException(
            status_code=409,
            detail=f"Concept already exists: {concept_name}",
        )

    concepts[concept_name] = {
        "forms": sorted(set(req.forms)),
        "false_positives": sorted(set(req.false_positives)),
    }

    save_concepts(concepts)

    return {
        "ok": True,
        "concept": concept_name,
    }


@router.post("/run/tier2")
async def run_tier2(req: RunJobRequest):
    job_id = str(uuid4())
    create_job(job_id=job_id, concept=req.concept)
    return {"job_id": job_id, "status": "queued"}


@router.post("/run/tier3")
async def run_tier3(req: RunJobRequest):
    job_id = str(uuid4())
    create_job(job_id=job_id, concept=req.concept)
    return {"job_id": job_id, "status": "queued"}