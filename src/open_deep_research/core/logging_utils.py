"""Structured logging utilities for Open Deep Research.

This module centralises logging and switches the project from ad-hoc
``print``/``logging`` usage to a single **structlog** pipeline that produces
either colourful console logs (local dev) or JSON (production/CI).

Usage
-----
```python
from open_deep_research.core.logging_utils import get_logger

logger = get_logger(__name__)
logger.info("Message", extra_field="value")
```
"""

from __future__ import annotations

import logging
import os
import sys
from typing import cast

import structlog

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def init_logger(level: str | int = "INFO", json: bool | None = None) -> None:  # noqa: D401
    """Initialise structlog + stdlib logging.

    This should be idempotent; calling it a second time is a no-op.

    Args:
        level: Log level name or numeric value.
        json: If *True* force JSON logs, if *False* force colourful console
            logs, if *None* auto-detect (JSON in CI, console otherwise).
    """
    # Avoid repeated configuration when modules import this helper multiple
    # times.
    if getattr(init_logger, "_configured", False):
        return

    # ---------------------------------------------------------------------
    # Determine output format
    # ---------------------------------------------------------------------
    if json is None:
        # Heuristic: if running in CI or env LOG_JSON=true we emit JSON.
        json = (
            bool(os.environ.get("CI"))
            or os.environ.get("LOG_JSON", "false").lower() == "true"
        )

    # Map string levels to numeric constants provided by ``logging``.
    if isinstance(level, str):
        level = level.upper()
        level = getattr(logging, level, logging.INFO)

    # Basic stdlib logging config – structlog will send output here.
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stdout,
    )

    # Choose renderer
    if json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        # Human-readable colourful output for local development.
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Finalise structlog processors list.
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,  # Inject contextvars (trace-id)…
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        renderer,
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    init_logger._configured = True  # type: ignore[attr-defined]


# Immediately configure logging on import for convenience.
init_logger()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_logger(name: str | None = None) -> structlog.BoundLogger:  # noqa: D401 – simple accessor
    """Return a structured :class:`structlog.BoundLogger`.

    Parameters
    ----------
    name:
        Optional module/qualifier. When provided it is bound as the *logger
        name* field so downstream log aggregation can filter on it.
    """
    logger = structlog.get_logger()
    if name:
        # Bind once so subsequent calls inherit the context.
        logger = logger.bind(logger_name=name)
    return cast("structlog.BoundLogger", logger)


# Convenience: helper to inject a trace-id into the current context.
def set_trace_id(trace_id: str) -> None:
    """Attach *trace_id* to all subsequent log records on this context."""
    from structlog.contextvars import bind_contextvars

    bind_contextvars(trace_id=trace_id)


# Generic helper to bind arbitrary key-value pairs to the current log context
# so they appear on every subsequent log call executed in the same async task
# or thread.  Example::
#
#     bind_log_context(task="research", node="generate_queries")
#     logger.info("starting")  # will include task=node fields


def bind_log_context(**kwargs: str) -> None:  # noqa: D401
    """Bind arbitrary logging context using structlog's contextvars backend."""

    from structlog.contextvars import bind_contextvars

    if kwargs:
        bind_contextvars(**kwargs)
