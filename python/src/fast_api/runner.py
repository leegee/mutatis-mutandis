import traceback

from lib.eebo_logging import logger
from lib.eebo_db import get_connection
from lib.eebo_logging import logger

from tier2_0_concept_events import (
    run_tier2_service,
    load_doc_metadata,
)

from tier3_0_plots import (
    run_tier3_service,
)

from .jobs_dao import (
    update_stage,
    mark_done,
    mark_error,
)


def run_job(
    *,
    job_id,
    concept,
    index,
    lookup,
):
    logger.info("fast_api.runner Enter")
    try:
        conn = get_connection()

        doc_meta = load_doc_metadata(conn)

        update_stage(job_id, "tier2")

        run_tier2_service(
            index=index,
            doc_meta=doc_meta,
            concepts_to_run=[
                (
                    concept,
                    { "forms": [concept], }
                )
            ],
            lookup=lookup,
        )

        update_stage(job_id, "tier3")

        run_tier3_service(
            index=index,
            lookup=lookup,
            concept=concept,
        )

        mark_done(job_id)

    except Exception:
        mark_error(
            job_id,
            traceback.format_exc(),
        )

        logger.exception(
            "[fastapi] job failed"
        )
