"""Structured logging utilities for Open Deep Research.

Provides a Rich-formatted logger so we can replace ad-hoc print() calls with
`logger.debug/info/warning/error` while keeping colourful, readable output.
"""

from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler

# Configure root logger ONLY once – other modules just call get_logger()
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s » %(message)s"

# If basicConfig() has already been called by another library we skip re-init
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format=_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(markup=True, rich_tracebacks=True)],
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:  # noqa: D401 – simple utility
    """Return a Rich-enabled logger.

    Example
    -------
    >>> logger = get_logger(__name__)
    >>> logger.debug("Something happened: %s", details)
    """
    return logging.getLogger(name if name else "open_deep_research") 