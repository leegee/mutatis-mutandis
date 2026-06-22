# fast_api/runner.py
from __future__ import annotations

import traceback

from lib.eebo_logging import setEmit, EmitFn
from lib.eebo_db import get_connection
from lib.eebo_config import CORPUS_TIER2_DB_PATH
from fast_api.jobs_dao import ( update_stage, mark_done, mark_error, )
from fast_api.event_bus import job_streams
from tier2_0_concept_events import ( run_tier2_service, load_doc_metadata, sqlite3_connection as events_sqlite3_connection )
from tier3_0_plots import ( run_tier3_service )


def make_emit(job_id: str) -> EmitFn:
    def emit(message: str, payload: str):
        q = job_streams.get(job_id)
        if q is None:
            return
        q.put_nowait({
            "message": message,
            "payload": json.loads(payload) if payload else None,
        })
    return emit


def run_job(*, job_id, concept, index, lookup):
    emit = make_emit(job_id)
    logger = setEmit(emit, "[tier2]",  {"concept": concept})

    logger.info("fast_api.runner Enter")

    try:
        conn = get_connection()
        doc_meta = load_doc_metadata(conn)

        # load the actual saved forms for this concept
        tier2_conn = events_sqlite3_connection(CORPUS_TIER2_DB_PATH)
        rows = tier2_conn.execute(
            "SELECT form, is_false_positive FROM concept_forms WHERE concept = ?",
            (concept,)
        ).fetchall()
        tier2_conn.close()

        forms           = [r[0] for r in rows if not r[1]]
        false_positives = [r[0] for r in rows if r[1]]

        update_stage(job_id, "tier2")

        kwargs = dict(
            db_path         = CORPUS_TIER2_DB_PATH,
            lookup          = lookup,
            index           = index,
            doc_meta        = doc_meta,
            false_positives = false_positives,
            concepts_to_run = [(concept, {"forms": forms})],
            emit            = emit,
        )
        logger.info(f"[fast_api.runner] calling run_tier2_service with keys: {list(kwargs.keys())}")

        run_tier2_service(**kwargs)
        logger.info("[fast_api.runner] returned from run_tier2_service with keys")

        update_stage(job_id, "tier3")

        logger.info("[fast_api.runner] calling run_tier3_service")
        run_tier3_service(
            db_path         = CORPUS_TIER2_DB_PATH,
            index           = index,
            lookup          = lookup,
            concept         = concept,
            false_positives = false_positives,
        )

        mark_done(job_id)

        logger.info(f"[fast_api.runner] Done {job_id}")

    except Exception:
        mark_error(job_id, traceback.format_exc())
        logger.exception(f"[fast_api.runner] run  failed for job_id={job_id}")
