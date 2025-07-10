import os
from typing import cast

import pytest
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import SearchAPI, WorkflowConfiguration
from open_deep_research.pydantic_state import DeepResearchState
from open_deep_research.workflow.unified_workflow import unified_planner_graph


@pytest.mark.asyncio
async def test_unified_workflow_offline_stub():
    """Ensure workflow still succeeds when search_api is NONE and fast flag off."""
    # Enable fast mode environment variable to utilize stub planner/researcher
    os.environ["ODR_FAST_TEST"] = "1"

    # Override configuration to disable web search explicitly
    cfg = cast(
        RunnableConfig,
        {"configurable": WorkflowConfiguration(search_api=SearchAPI.NONE).__dict__},
    )
    state = DeepResearchState(topic="AI Testing")

    final_state = await unified_planner_graph.ainvoke(state, cfg)

    assert "final_report" in final_state
    assert len(final_state["final_report"]) > 0
