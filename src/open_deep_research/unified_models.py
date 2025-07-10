from __future__ import annotations

"""Shared typed payloads for the forthcoming unified workflow.

These Pydantic models allow node adapters to communicate in a
statically-safe manner while continuing to interoperate with the legacy
(dict-based) workflow and multi-agent implementations.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Re-export Section so downstream modules don’t need to import twice.
# ---------------------------------------------------------------------------
from open_deep_research.pydantic_state import Section  # noqa: F401

__all__ = [
    "PlannerResult",
    "ResearchResult",
    "RouterDecision",
]


class PlannerResult(BaseModel):
    """Typed result returned by the *planning* step.

    Attributes
    ----------
    sections
        List of section objects that form the report skeleton.
    metadata
        Optional additional information emitted by the underlying node
        (e.g., token count, model name, debugging info).  Stored as an
        untyped mapping to remain forward-compatible.
    """

    sections: List[Section]
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class ResearchResult(BaseModel):
    """Result of researching a set of sections."""

    completed_sections: List[Section]
    source_str: str = ""
    search_metadata: Optional[Dict[str, Any]] = None


class RouterDecision(BaseModel):
    """Explicit routing decision used by conditional edges."""

    next_node: str
    reason: Optional[str] = None
