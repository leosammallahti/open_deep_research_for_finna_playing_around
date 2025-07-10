import asyncio

import pytest

import os

os.environ["ODR_FAST_TEST"] = "1"

from open_deep_research.pydantic_state import DeepResearchState
from open_deep_research.node_adapter import NodeAdapter


@pytest.mark.asyncio
async def test_workflow_planner_adapter():
    state = DeepResearchState(topic="AI")
    # RunnableConfig can be empty for happy-path
    result = await NodeAdapter.workflow_planner(state, {})  # type: ignore[arg-type]
    assert result.sections is not None


@pytest.mark.asyncio
async def test_multi_agent_planner_adapter_smoke():
    state = DeepResearchState(topic="Medicine")
    result = await NodeAdapter.multi_agent_planner(state, {})  # type: ignore[arg-type]
    assert hasattr(result, "sections")
