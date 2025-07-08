"""Wrapper for Tavily tools with singleton client reuse and concurrency protection.

This module centralises creation of Tavily Search / Extract LangChain tools so the rest
of the codebase can simply import `tavily_search_tool` (and, if needed,
`tavily_extract_tool`).  We also expose an `acquire_tavily_semaphore` async context
manager to keep concurrent requests within rate-limit bounds (100 RPM on dev keys).
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langchain_core.tools import BaseTool
from langchain_tavily import TavilyExtract, TavilySearch

# ---------------------------------------------------------------------------
# Concurrency guard
# ---------------------------------------------------------------------------
# Dev keys: 100 requests / minute.  We conservatively allow up to 40 concurrent
# requests; adjust via env var if needed.
_MAX_CONCURRENT = int(os.getenv("TAVILY_MAX_CONCURRENT", "40"))
_tavily_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)


@asynccontextmanager
async def acquire_tavily_semaphore() -> AsyncGenerator[None, None]:
    """Async context-manager that acquires the Tavily semaphore.

    Usage::
        async with acquire_tavily_semaphore():
            result = await tavily_search_tool.ainvoke({"query": "…"})
    """
    await _tavily_semaphore.acquire()
    try:
        yield
    finally:
        _tavily_semaphore.release()


# ---------------------------------------------------------------------------
# Singleton tool instances
# ---------------------------------------------------------------------------
# NOTE: These tool objects are stateless; holding one instance is sufficient.
# You can still customise per-call parameters when invoking with `tool.ainvoke`.

# Default configuration follows Tavily best-practices: advanced depth, no raw
# content by default (encourage two-step extraction for large docs).

_tavily_search = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_raw_content=False,
)

_tavily_extract = TavilyExtract()

# Re-export for external import.
# The leading underscore variables stay internal; aliases are exported.

# LangChain `BaseTool` subclasses
TavilySearchTool: BaseTool = _tavily_search  # type: ignore[assignment]
TavilyExtractTool: BaseTool = _tavily_extract  # type: ignore[assignment]

# Convenience names (backwards-compat with existing code expectations)
# These match previous camelCase naming.

tavily_search_tool = TavilySearchTool
tavily_extract_tool = TavilyExtractTool 