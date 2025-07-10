"""Compile the final markdown report for the unified workflow.

This node receives the top-level ``DeepResearchState`` containing:
• ``sections`` – the planned skeleton in the original order.
• ``completed_sections`` – any sections already filled by earlier nodes.
• ``source_str`` – concatenated raw source dumps from the researcher step.

Responsibilities
-----------------
1. Stitch complete & placeholder sections into a single markdown body.
2. Optionally append a numbered **Sources** list and the verbose
   **Raw Sources** block depending on ``WorkflowConfiguration`` flags.

The implementation purposely avoids heavy formatting logic – anything
complex should live in ``open_deep_research.core.format_utils`` to keep
this node lightweight and easily testable.
"""

from __future__ import annotations

# Standard library
from typing import Dict, List

# Third-party
from langchain_core.runnables import RunnableConfig

# First-party
from open_deep_research.configuration import WorkflowConfiguration
from open_deep_research.core.format_utils import (
    extract_unique_urls,
    format_sources_section,
)
from open_deep_research.pydantic_state import DeepResearchState, Section

__all__ = ["compile_report_unified"]


def _section_lookup(completed: List[Section]) -> Dict[str, Section]:
    return {s.name: s for s in completed}


async def compile_report_unified(
    state: DeepResearchState, config: RunnableConfig
) -> Dict[str, str]:
    """Concatenate section content and append citations.

    Parameters
    ----------
    state
        The current immutable workflow state.
    config
        ``RunnableConfig`` whose ``configurable`` key may include an
        embedded ``WorkflowConfiguration`` dict.  When *config* is not a
        mapping (e.g. tests call the function directly) we fall back to
        default `WorkflowConfiguration()` values.

    Returns:
    --------
    dict
        A LangGraph-style state patch with the single key
        ``{"final_report": markdown_string}``.
    """
    lookup = _section_lookup(state.completed_sections)

    lines: List[str] = []
    for sec in state.sections:
        lines.append(f"# {sec.name}\n")
        if sec.name in lookup and lookup[sec.name].content:
            lines.append(lookup[sec.name].content)
        else:
            # For non-research or missing research section, include description
            lines.append(sec.description or "(no content)")
        lines.append("\n")

    report_body = "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Sources & citation handling
    # ------------------------------------------------------------------
    cfg = (
        WorkflowConfiguration.from_runnable_config(config)
        if isinstance(config, dict)
        else WorkflowConfiguration()
    )

    if cfg.include_source_str:
        unique_urls = extract_unique_urls(state.source_str)
        if unique_urls:
            report_body += format_sources_section(unique_urls)

            if cfg.include_raw_source_details:
                report_body += "\n\n---\n\n## Raw Sources\n\n" + state.source_str

    return {"final_report": report_body}
