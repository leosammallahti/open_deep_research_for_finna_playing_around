from __future__ import annotations

"""Unified planner node that dynamically selects the correct planner.

This node is the first *real* piece of the upcoming unified workflow.
It relies on the previously-added ``NodeAdapter`` wrappers so we avoid
rewriting the legacy planner logic.
"""

from langchain_core.runnables import RunnableConfig

from open_deep_research.node_adapter import NodeAdapter
from open_deep_research.pydantic_state import DeepResearchState
from open_deep_research.unified_models import PlannerResult

__all__ = ["unified_planner"]


async def unified_planner(state: DeepResearchState, config: RunnableConfig) -> dict:  # noqa: D401
    """Route to workflow or multi-agent planner and return a state patch."""

    # --------------------------------------------------------------
    # Derive execution mode – precedence order:
    # 1. Explicit ``state.execution_mode`` (set by caller)
    # 2. ``config.configurable.mode`` override
    # 3. Feature flag *mcp_support* (treated as multi-agent)
    # 4. Fallback to "workflow"
    # --------------------------------------------------------------
    cfg_mapping = config.get("configurable", {}) if isinstance(config, dict) else {}
    features_map = (
        cfg_mapping.get("features", {})
        if isinstance(cfg_mapping.get("features", {}), dict)
        else {}
    )

    mode = (
        getattr(state, "execution_mode", None)
        or cfg_mapping.get("mode")
        or ("multi_agent" if features_map.get("mcp_support") else None)
        or "workflow"
    )

    # --------------------------------------------------------------
    # Dispatch
    # --------------------------------------------------------------
    if mode == "multi_agent":
        result: PlannerResult = await NodeAdapter.multi_agent_planner(state, config)
    else:  # default to workflow logic
        result = await NodeAdapter.workflow_planner(state, config)

    # Return as state-patch (LangGraph merge semantics)
    return {"sections": result.sections}
