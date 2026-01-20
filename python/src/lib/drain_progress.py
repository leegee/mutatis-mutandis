"""
Progress-drain helper for supervising asynchronous workers.

This module implements a "drain-until-quiescent" pattern:

When work is performed asynchronously, progress signals (e.g. shared counters)
may lag behind worker completion. A naive loop that exits as soon as all workers
report "ready" can miss late-arriving progress updates.

The drain loop continues updating progress until:
  (a) all workers have finished, AND
  (b) no unobserved progress remains.

This guarantees correctness of progress reporting for long-running,
multiprocess jobs.
"""

from typing import Iterable, Callable, Any


def drain_progress(
    *,
    total: int,
    counter: Any,
    workers: Iterable[Any],
    pbar_factory: Callable[[int], Any],
    poll: Callable[[], None],
) -> None:
    """
    Drain progress updates until workers are finished and progress is quiescent.

    Args:
        total:
            Total expected progress count.
        counter:
            Shared counter object with a `.value` attribute.
        workers:
            Iterable of async worker handles exposing `.ready()`.
        pbar_factory:
            Callable that returns a progress-bar-like object when given `total`.
        poll:
            Callable invoked once per loop iteration to avoid busy-waiting
            (e.g. time.sleep).
    """
    with pbar_factory(total) as pbar:
        last_count = 0

        def update():
            nonlocal last_count
            current = counter.value
            delta = current - last_count
            if delta > 0:
                pbar.update(delta)
                last_count = current

        while True:
            update()
            if all(w.ready() for w in workers):
                break
            poll()

        # Final drain in case workers finished before last counter increment
        update()
