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
from typing import TYPE_CHECKING, Any, AsyncGenerator, cast

# Only import for type checking to avoid unnecessary runtime dependency.
if TYPE_CHECKING:  # pragma: no cover – typing-only import
    from langchain_core.tools import BaseTool  # noqa: WPS433 – type-only import
else:  # pragma: no cover – fallback alias for runtime
    BaseTool = Any  # type: ignore

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

# Allow runtime configuration of `max_results` and search depth via environment
# variables so users can easily trade off credit usage vs. recall.  The defaults
# favour cost-effectiveness ("standard" depth and 5 results).


# Ensure we always pass a valid literal ("basic" or "advanced") to TavilySearch.
# Historically, some configurations used "standard" which was renamed to "basic" in
# newer Tavily versions.  We also normalise case and provide a sensible fallback.

_TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# Raw env var (case-insensitive)
_raw_depth = os.getenv("TAVILY_SEARCH_DEPTH", "advanced").lower()

# Mapping for backwards-compatibility and validation
_depth_aliases = {
    "standard": "basic",  # legacy name → new literal
    "basic": "basic",
    "advanced": "advanced",
}

# Use mapped value if valid; otherwise default to "advanced" to maintain functionality.
_TAVILY_SEARCH_DEPTH = _depth_aliases.get(_raw_depth, "advanced")

import os

# First try env var; if not set load via pydantic *settings* which reads .env
_tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
if not _tavily_api_key:
    try:
        from open_deep_research.core.settings import settings as _app_settings

        _tavily_api_key = (_app_settings.tavily_api_key or "").strip()
    except Exception:
        _tavily_api_key = ""

# Ensure the key is set as an environment variable so ``langchain_tavily`` internals
# that rely on ``os.getenv('TAVILY_API_KEY')`` continue to work even when we pass
# the key programmatically or load it from the `.env` file.
if _tavily_api_key:
    os.environ["TAVILY_API_KEY"] = _tavily_api_key

_tavily_search = TavilySearch(
    tavily_api_key=_tavily_api_key or None,  # pass None to let LC fallback to env
    max_results=_TAVILY_MAX_RESULTS,
    search_depth=_TAVILY_SEARCH_DEPTH,
    include_raw_content=False,
)

_tavily_extract = TavilyExtract()

# Re-export for external import.
# The leading underscore variables stay internal; aliases are exported.

# LangChain `BaseTool` subclasses (cast to silence type-checkers without ignore)
TavilySearchTool = cast("BaseTool", _tavily_search)
TavilyExtractTool = cast("BaseTool", _tavily_extract)

# Convenience names (backwards-compat with existing code expectations)
# These match previous camelCase naming.

tavily_search_tool = TavilySearchTool
tavily_extract_tool = TavilyExtractTool
