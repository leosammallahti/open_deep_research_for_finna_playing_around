# New file tests/test_multi_agent_concurrency.py
"""Concurrency regression test for the multi-agent workflow.

This test simulates two parallel research sections and ensures the graph
completes without raising *InvalidUpdateError* or other concurrency-related
exceptions.

All external model calls/searches are monkey-patched with ultra-light stubs so
no network traffic occurs.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from open_deep_research.multi_agent import graph as multi_agent_graph
from open_deep_research.pydantic_state import DeepResearchState, Section


class _DummyLLM:  # noqa: D401 – minimal stub
    """Trivial chat model stub that *immediately* finishes the task."""

    def bind_tools(self, tools, **kwargs):  # noqa: D401
        return self

    def with_retry(self, **kwargs):  # noqa: D401 – no-op
        return self

    async def ainvoke(self, messages: list[BaseMessage]):  # noqa: D401
        # Always return an AIMessage signalling completion (no tool calls).
        return AIMessage(content="Done")


@pytest.mark.asyncio
async def test_multi_agent_concurrency(monkeypatch):  # noqa: D401
    """Graph should finish cleanly when two sections run in parallel."""

    # ------------------------------------------------------------------
    # Monkey-patch model initialisation so no real LLM is called
    # ------------------------------------------------------------------
    from open_deep_research import multi_agent as ma

    monkeypatch.setattr(ma, "init_chat_model", lambda *a, **k: _DummyLLM())

    # Patch search tools referenced in multi_agent to synchronous stubs that
    # just return a constant string.
    monkeypatch.setattr(ma, "tavily_search_tool", lambda *a, **k: "SOURCE")
    monkeypatch.setattr(ma, "duckduckgo_search", lambda *a, **k: "SOURCE")

    # ------------------------------------------------------------------
    # Seed state with two research sections
    # ------------------------------------------------------------------
    sections = [
        Section(name="A", description="Alpha", research=True),
        Section(name="B", description="Beta", research=True),
    ]
    seed_state = DeepResearchState(topic="Testing", sections=sections)

    # ------------------------------------------------------------------
    # Run the graph – *await* since multi_agent.graph is async
    # ------------------------------------------------------------------
    final_state = await multi_agent_graph.ainvoke(seed_state)

    # ------------------------------------------------------------------
    # Assertions – no exception was raised, sections completed
    # ------------------------------------------------------------------
    assert final_state.completed_sections, "No sections completed"  # type: ignore[attr-defined]
    assert isinstance(final_state.final_report, str)  # type: ignore[attr-defined]
