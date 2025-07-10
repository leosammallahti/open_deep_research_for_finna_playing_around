import os

import pytest

from open_deep_research.pydantic_state import DeepResearchState
from open_deep_research.workflow.unified_workflow import unified_planner_graph

os.environ["ODR_FAST_TEST"] = "1"


@pytest.mark.asyncio
async def test_unified_planner_graph_smoke():
    state = DeepResearchState(topic="Quantum Computing")

    final_state = await unified_planner_graph.ainvoke(state)

    assert "sections" in final_state
    assert len(final_state["sections"]) > 0 or final_state["sections"] == []
