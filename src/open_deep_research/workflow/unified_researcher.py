from __future__ import annotations

"""Unified researcher node for the proof-of-concept workflow."""

from langchain_core.runnables import RunnableConfig

from open_deep_research.node_adapter import NodeAdapter
from open_deep_research.pydantic_state import DeepResearchState

__all__ = ["unified_researcher"]


async def unified_researcher(state: DeepResearchState, config: RunnableConfig):
    # Determine execution mode – follow the same precedence order used by
    # *unified_planner* for consistency.
    cfg_mapping = config.get("configurable", {}) if isinstance(config, dict) else {}
    features_map = (
        cfg_mapping.get("features", {})
        if isinstance(cfg_mapping.get("features", {}), dict)
        else {}
    )

    # Precedence: config override > feature flag > state default
    mode = (
        cfg_mapping.get("mode")
        or ("multi_agent" if features_map.get("mcp_support") else None)
        or state.execution_mode
    )

    if mode == "multi_agent":
        result = await NodeAdapter.multi_agent_researcher(state, config)
    else:
        result = await NodeAdapter.workflow_researcher(state, config)

    # Merge completed sections into state
    patch = {
        "completed_sections": result.completed_sections,
        "source_str": state.source_str + result.source_str,
    }
    return patch
