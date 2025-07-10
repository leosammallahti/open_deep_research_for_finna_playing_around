"""Test stubs for offline (token-free) execution.

This module provides minimal replacements for expensive components so the
entire workflow can run inside pytest without hitting external APIs.

* FakeChatModel  – drop-in LangChain chat-model stub (supports .ainvoke & helpers)
* fake_search()  – deterministic replacement for select_and_execute_search()
"""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace
from typing import Any, Dict, List, cast

__all__ = ["FakeChatModel", "fake_search"]


class FakeChatModel:  # noqa: D101 – test stub
    def __init__(self, role: str | None = None):
        self.role = role or "stub"

    # ------------------------------------------------------------------
    # LangChain-compatible interface
    # ------------------------------------------------------------------
    async def ainvoke(self, messages: List[Any]) -> SimpleNamespace:  # noqa: D401 – async stub
        return self._dispatch(messages)

    def invoke(self, messages: List[Any]) -> SimpleNamespace:  # noqa: D401 – sync stub
        return self._dispatch(messages)

    def _dispatch(self, messages: List[Any]) -> SimpleNamespace:  # noqa: D401 – helper
        """Return canned responses based on the last user prompt."""
        last = messages[-1]
        content = (
            last["content"] if isinstance(last, dict) else getattr(last, "content", "")
        )

        # Very naive heuristics – good enough for tests
        if "sections" in str(content).lower():
            payload = cast(
                Dict[str, Any],
                {
                    "sections": [
                        {
                            "name": "Introduction",
                            "description": "Overview",
                            "research": True,
                        },
                        {
                            "name": "Conclusion",
                            "description": "Summary and outlook",
                            "research": False,
                        },
                    ],
                },
            )
            return SimpleNamespace(content=json.dumps(payload))

        if re.search(r"generate search queries", str(content), re.I):
            payload = {"queries": [{"search_query": "stub query"}]}
            return SimpleNamespace(content=json.dumps(payload))

        if "grade" in str(content).lower():
            payload = {"grade": "pass", "follow_up_queries": []}
            return SimpleNamespace(content=json.dumps(payload))

        # Default markdown body for a section
        return SimpleNamespace(
            content=f"### Fake {self.role or 'LLM'} output\n\nLorem ipsum …"
        )

    # ------------------------------------------------------------------
    # Helper methods used by call-sites
    # ------------------------------------------------------------------
    def with_structured_output(self, *args: Any, **kwargs: Any) -> "FakeChatModel":  # noqa: D401 – keep fluent API
        return self

    def with_retry(self, *args: Any, **kwargs: Any) -> "FakeChatModel":  # noqa: D401 – keep fluent API
        return self

    def bind_tools(self, *args: Any, **kwargs: Any) -> "FakeChatModel":  # noqa: D401 – supervisor multi-agent path
        return self

    # async variants (LangChain 0.2+)
    async def agenerate_prompt(self, *args: Any, **kwargs: Any) -> "FakeChatModel":
        return self

    async def agenerate(self, *args: Any, **kwargs: Any) -> "FakeChatModel":
        return self


async def fake_search(
    search_api: str, query_list: List[str], params_to_pass: Dict[str, Any]
) -> str:
    """Return deterministic fake search result string."""
    await asyncio.sleep(0)  # yield control, keeps async signature intact
    queries = " | ".join(query_list)
    return f"[Fake search:{search_api}] {queries}"
