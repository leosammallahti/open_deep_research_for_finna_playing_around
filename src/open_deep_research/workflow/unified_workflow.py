from __future__ import annotations

"""Proof-of-concept graph that uses the new *unified_planner* node.

For now the graph is extremely small: START → unified_planner → END.
It compiles and can be called from tests to validate plumbing while we
incrementally add more nodes.
"""

from langgraph.graph import END, START, StateGraph

from open_deep_research.pydantic_state import DeepResearchState
from open_deep_research.workflow.unified_compiler import compile_report_unified
from open_deep_research.workflow.unified_planner import unified_planner
from open_deep_research.workflow.unified_researcher import unified_researcher

__all__ = ["unified_planner_graph"]


# ---------------------------------------------------------------------------
# Build graph at import-time so tests can simply import & invoke.
# ---------------------------------------------------------------------------

_builder = StateGraph(DeepResearchState)
_builder.add_node("plan", unified_planner)
_builder.add_node("research", unified_researcher)
_builder.add_node("compile", compile_report_unified)

_builder.add_edge(START, "plan")
_builder.add_edge("plan", "research")
_builder.add_edge("research", "compile")
_builder.add_edge("compile", END)

unified_planner_graph = _builder.compile()
