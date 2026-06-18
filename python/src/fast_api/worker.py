import asyncio

from .jobs_db import (
    claim_next_job,
)

from .runner import run_job
from .state import (
    init_state,
    STATE,
)


async def worker_loop():

    init_state()

    while True:

        job = claim_next_job()

        if job is None:
            await asyncio.sleep(1)
            continue

        await asyncio.to_thread(
            run_job,
            job_id=job["job_id"],
            concept=job["concept"],
            index=STATE.index,
            lookup=STATE.lookup,
        )
