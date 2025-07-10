from __future__ import annotations

import pytest

from open_deep_research.node_adapter import NodeAdapter
from open_deep_research.pydantic_state import DeepResearchState, Section

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyReportState:  # noqa: D401 – simple stub object
    """Minimal state stub returned by the patched multi-agent graph."""

    def __init__(self, completed_sections):  # noqa: D401
        self.completed_sections = completed_sections
        self.source_str = "DUMMY_SOURCE"


# ---------------------------------------------------------------------------
# Fast-mode path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_agent_researcher_fast_mode(monkeypatch):  # noqa: D401
    """When *ODR_FAST_TEST* is enabled the adapter returns placeholder content."""

    monkeypatch.setenv("ODR_FAST_TEST", "1")

    state = DeepResearchState(
        topic="Testing Fast Mode",
        sections=[Section(name="Alpha", description="A", research=True)],
    )

    result = await NodeAdapter.multi_agent_researcher(state, {})  # type: ignore[arg-type]

    assert result.completed_sections, "Expected placeholder sections to be returned"
    assert result.completed_sections[0].content.startswith("# ")


# ---------------------------------------------------------------------------
# Real-mode path (graph invocation) – patched to avoid network/LLM usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_agent_researcher_real_mode(monkeypatch):  # noqa: D401
    """Adapter should forward to *multi_agent.graph* when not in fast mode."""

    # Ensure fast-mode flag is removed
    monkeypatch.delenv("ODR_FAST_TEST", raising=False)

    # Prepare dummy completed sections that the stub graph will return
    dummy_sections = [
        Section(name="Beta", description="B", research=True, content="Content"),
    ]

    # Build a stub graph object with an async *ainvoke* method
    class _GraphStub:  # noqa: D401
        async def ainvoke(self, _state, _config=None):  # noqa: D401
            return _DummyReportState(dummy_sections)

    # Patch the *graph* attribute before the adapter imports it
    from open_deep_research import multi_agent as ma

    monkeypatch.setattr(ma, "graph", _GraphStub())

    # Execute adapter
    state = DeepResearchState(topic="Testing", sections=[])
    result = await NodeAdapter.multi_agent_researcher(state, {})  # type: ignore[arg-type]

    # Validate that the sections from the stub graph were propagated
    assert result.completed_sections == dummy_sections
    assert result.source_str == "DUMMY_SOURCE"


# ---------------------------------------------------------------------------
# Planner adapter fast-mode test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_agent_planner_fast_mode(monkeypatch):  # noqa: D401
    """multi_agent_planner should return dummy sections when in fast mode."""

    monkeypatch.setenv("ODR_FAST_TEST", "1")

    from open_deep_research.pydantic_state import DeepResearchState

    state = DeepResearchState(topic="Planner Fast")
    result = await NodeAdapter.multi_agent_planner(state, {})  # type: ignore[arg-type]

    assert result.sections, "Expected dummy sections"
    assert result.sections[0].name == "Intro"
