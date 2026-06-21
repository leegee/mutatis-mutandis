# fast_api/event_bus.py
import asyncio

job_streams: dict[str, asyncio.Queue] = {}
