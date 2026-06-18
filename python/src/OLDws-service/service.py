# service.py

"""
FastAPI WebSocket service exposing tier2 and tier3 pipelines.

Endpoints
---------
ws://host/ws/tier2
ws://host/ws/tier3

Each endpoint accepts a single JSON message on connect that carries
the run parameters, streams progress events back as JSON, and closes
when the job completes or errors.

Inbound message shapes
----------------------
Tier 2:
{
    "forms":           ["prerogative", "prerogatives"],   // either
    "false_positives": ["positive"],                      // or
}

Tier 3:
{
    "concept":         "PREROGATIVE",   //
    "false_positives": [],              // optional
    "mode":            "full"           // "full" | "clustering"
}

Outbound message shapes
-----------------------
Progress:
{ "event": "<event_name>", "data": { ... } }

Completion:
{ "event": "done" }

Error:
{ "event": "error", "detail": "<message>" }
"""

import asyncio
import json
import traceback
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from lib.eebo_config import ZARR_ROOT, FAISS_TIER1_INDEX
from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_db import get_connection
from lib.eebo_logging import logger

from tier2_1_concept_analysis import run_tier2_service
from tier2_2_umap import run_tier3_service
from tier2_0_concept_events import ZarrEventLookup

app = FastAPI(title="EEBO Pipeline Service")

# One thread per blocking job is enough; both pipelines are CPU/IO bound
# and are not designed for concurrency within a single run.
_executor = ThreadPoolExecutor(max_workers=4)


def _make_emitter(websocket: WebSocket, loop: asyncio.AbstractEventLoop):
    """
    Returns a synchronous emit(event, data) callable safe to call from a
    worker thread.  Schedules a coroutine on the event loop that owns the
    WebSocket.
    """
    async def _send(payload: str):
        await websocket.send_text(payload)

    def emit(event: str, data: dict):
        payload = json.dumps({"event": event, "data": data})
        asyncio.run_coroutine_threadsafe(_send(payload), loop)

    return emit


async def _run_in_thread(fn, *args, **kwargs):
    """Run a blocking callable in the thread pool, awaiting its completion."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


# Tier 2
@app.websocket("/ws/tier2")
async def ws_tier2(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    try:
        raw = await websocket.receive_text()
        params = json.loads(raw)

        forms = (
            {f.strip() for f in params["forms"]}
            if params.get("forms")
            else None
        )
        false_positives = (
            {f.strip() for f in params["false_positives"]}
            if params.get("false_positives")
            else None
        )
        clear = bool(params.get("clear", False))

        emit = _make_emitter(websocket, loop)

        index = EeboFaissIndex.load(FAISS_TIER1_INDEX)
        conn  = get_connection()

        try:
            await _run_in_thread(
                run_tier2_service,
                conn=conn,
                index=index,
                forms=forms,
                false_positives=false_positives,
                emit=emit,
            )
        finally:
            conn.close()

        await websocket.send_text(json.dumps({"event": "done"}))

    except WebSocketDisconnect:
        logger.info("[ws/tier2] client disconnected")
    except Exception as exc:
        logger.error(f"[ws/tier2] error: {exc}\n{traceback.format_exc()}")
        try:
            await websocket.send_text(
                json.dumps({"event": "error", "detail": str(exc)})
            )
        except Exception:
            pass
    finally:
        await websocket.close()


# Tier 3
@app.websocket("/ws/tier3")
async def ws_tier3(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    try:
        raw = await websocket.receive_text()
        params = json.loads(raw)

        concept         = params.get("concept") or None
        false_positives = params.get("false_positives") or []
        mode            = params.get("mode", "full")

        if mode not in ("full", "clustering"):
            await websocket.send_text(
                json.dumps({"event": "error", "detail": f"Unknown mode: {mode!r}"})
            )
            await websocket.close()
            return

        emit = _make_emitter(websocket, loop)

        lookup = ZarrEventLookup(ZARR_ROOT / "tier1")
        index  = EeboFaissIndex.load(FAISS_TIER1_INDEX)

        await _run_in_thread(
            run_tier3_service,
            index=index,
            lookup=lookup,
            concept=concept,
            false_positives=false_positives,
            mode=mode,
            emit=emit,
        )

        await websocket.send_text(json.dumps({"event": "done"}))

    except WebSocketDisconnect:
        logger.info("[ws/tier3] client disconnected")
    except Exception as exc:
        logger.error(f"[ws/tier3] error: {exc}\n{traceback.format_exc()}")
        try:
            await websocket.send_text(
                json.dumps({"event": "error", "detail": str(exc)})
            )
        except Exception:
            pass
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=False)
