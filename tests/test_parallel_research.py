import asyncio
import types
import time
from dataclasses import asdict

import pytest

from open_deep_research.node_adapter import NodeAdapter
from open_deep_research.pydantic_state import DeepResearchState, Section
from open_deep_research.configuration import WorkflowConfiguration, SearchAPI
from langchain_core.runnables import RunnableConfig


@pytest.mark.asyncio
async def test_parallel_research_execution(monkeypatch):
    """Ensure research sections are executed concurrently when the feature flag is enabled."""

    # ------------------------------------------------------------------
    # Prepare stub runnable that tracks concurrent invocations
    # ------------------------------------------------------------------
    active_calls = 0
    max_concurrency = 0
    sleep_time = 0.05  # seconds

    async def _stub(inputs, _config):  # noqa: D401 – test helper
        nonlocal active_calls, max_concurrency

        active_calls += 1
        max_concurrency = max(max_concurrency, active_calls)

        # Simulate IO-bound work
        await asyncio.sleep(sleep_time)

        active_calls -= 1

        sec: Section = inputs["section"]
        completed = sec.model_copy(update={"content": f"# {sec.name}\n\nStub content"})
        return {"completed_sections": [completed], "source_str": ""}

    stub_runnable = types.SimpleNamespace(ainvoke=_stub)

    # Patch the legacy graph node to our stub
    import open_deep_research.graph as _graph_mod

    monkeypatch.setattr(
        _graph_mod.graph,
        "get_node",
        lambda *args, **kwargs: stub_runnable,
        raising=False,
    )

    # ------------------------------------------------------------------
    # Ensure real-mode execution (disable FAST_TEST stub path)
    # ------------------------------------------------------------------
    monkeypatch.setenv("ODR_FAST_TEST", "0")

    # ------------------------------------------------------------------
    # Construct test state with multiple research sections
    # ------------------------------------------------------------------
    sections = [
        Section(name="Background", description="desc", research=True),
        Section(name="Methodology", description="desc", research=True),
        Section(name="Results", description="desc", research=True),
    ]
    state = DeepResearchState(topic="Test Topic", sections=sections)

    cfg = WorkflowConfiguration(
        search_api=SearchAPI.TAVILY,  # any non-NONE provider
        features={"parallel_research": True},
    )

    runnable_cfg_dict: dict[str, dict[str, object]] = {"configurable": asdict(cfg)}
    from typing import cast

    config_param = cast(RunnableConfig, runnable_cfg_dict)

    # ------------------------------------------------------------------
    # Execute and measure duration
    # ------------------------------------------------------------------
    start = time.perf_counter()
    result = await NodeAdapter.workflow_researcher(state, config_param)
    duration = time.perf_counter() - start

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    assert len(result.completed_sections) == len(sections)

    # Parallelism check: max_concurrency > 1 indicates overlapping tasks
    assert max_concurrency > 1, "Expected concurrent execution of research tasks"

    # Duration sanity check – should be less than sequential sleep_time * n
    assert duration < sleep_time * len(sections) * 0.9, (
        "Research calls did not execute in parallel"
    )
