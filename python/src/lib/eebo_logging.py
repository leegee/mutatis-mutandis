# eebo_logging.py

import json
import logging
from typing import Any, Callable

EmitFn = Callable[[str, str], None]

# Base logger used by the application
logger = logging.getLogger("eebo")


class EeboLogger(logging.LoggerAdapter):

    def __init__(
        self,
        logger: logging.Logger,
        emit: EmitFn | None = None,
        tag: str = "",
        context: dict[str, Any] | None = None,
    ):
        super().__init__(logger, {})
        self._emit = emit
        self._tag = tag
        self._context = context or {}

    def process(self, msg, kwargs):
        if self._tag:
            msg = f"{self._tag} {msg}"

        kwargs.setdefault("extra", {}).update(self._context)

        return msg, kwargs

    def log(self, level, msg, *args, **kwargs):
        if self._emit:
            try:
                rendered = msg % args if args else str(msg)
            except Exception:
                rendered = str(msg)

            payload = dict(self._context)
            payload.update(kwargs.get("extra", {}))

            self._emit(
                rendered,
                json.dumps(payload, default=str),
            )

        super().log(level, msg, *args, **kwargs)


def setEmit(
    emit: EmitFn,
    tag: str = "",
    context: dict[str, Any] | None = None,
) -> EeboLogger:
    return EeboLogger(
        logger,
        emit=emit,
        tag=tag,
        context=context,
    )
