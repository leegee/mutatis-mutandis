import asyncio

from fast_api.jobs_dao import ( claim_next_job, )
from fast_api.state import ( init_state, STATE, )
from fast_api.runner import run_job

async def worker_loop():
    init_state()

    while True:
        job = claim_next_job()

        if job is None:
            await asyncio.sleep(1)
            continue

        await asyncio.to_thread(
            lambda: run_job(
                job_id  = job["job_id"],
                concept = job["concept"],
                index   = STATE.index,
                lookup  = STATE.get_tier1_zarr_lookup(),
            )
        )
