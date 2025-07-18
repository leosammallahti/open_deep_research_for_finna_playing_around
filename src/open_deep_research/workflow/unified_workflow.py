from __future__ import annotations

"""Proof-of-concept graph that uses the new *unified_planner* node.

For now the graph is extremely small: START → unified_planner → END.
It compiles and can be called from tests to validate plumbing while we
incrementally add more nodes.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from open_deep_research.pydantic_state import DeepResearchState
from open_deep_research.workflow.unified_compiler import compile_report_unified
from open_deep_research.workflow.unified_planner import unified_planner
from open_deep_research.workflow.unified_researcher import unified_researcher

__all__ = ["unified_planner_graph"]


# ---------------------------------------------------------------------------
# Build graph at import-time so tests can simply import & invoke.
# ---------------------------------------------------------------------------

# Monkey-patch StateGraph with a tiny convenience helper so we can keep
# workflow definitions concise and readable.  This is deliberately
# implemented here (closest to first use) rather than in a shared utils
# module to avoid polluting the public surface – if more workflows start
# using it we can promote it.

def _add_edge_chain(self: StateGraph, *steps: Any) -> None:  # noqa: D401
    """Add edges for *steps* in sequential order (A→B→C...)."""

    if len(steps) < 2:
        raise ValueError("add_edge_chain requires at least two steps")

    for src, dst in zip(steps, steps[1:]):
        self.add_edge(src, dst)

# Monkey-patch once – mypy: ignore because StateGraph is external lib
StateGraph.add_edge_chain = _add_edge_chain  # type: ignore[attr-defined]

_builder = StateGraph(DeepResearchState)
_builder.add_node("plan", unified_planner)
_builder.add_node("research", unified_researcher)
_builder.add_node("compile", compile_report_unified)

_builder.add_edge_chain(START, "plan", "research", "compile", END)  # type: ignore[attr-defined]

unified_planner_graph = _builder.compile()

async def generate_report(self, topic: str) -> str:
    await self.planner.plan(topic)
    # Add await for other steps as needed
    return "Report generated asynchronously"
