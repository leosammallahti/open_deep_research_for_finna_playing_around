"""Pydantic state models for Open Deep Research application.

This module defines immutable state models using Pydantic for managing research workflow state,
including report sections, search queries, feedback, and multi-agent workflow states.
"""

import operator
from typing import Annotated, List, Literal, Optional

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, ConfigDict, Field


# Helper aggregator that always returns the new value (last write wins)
def _replace_fn(_old, new):
    """Aggregator that always prefers the *new* value.

    Used with Annotated[...] so multiple parallel updates to a single-value
    field do not raise a concurrency error. Effectively implements the
    *last write wins* semantics.
    """
    return new


ReplaceFn = _replace_fn


# Helper aggregator that returns the maximum value
def _max_fn(old, new):
    """Aggregator that returns the maximum value.

    Used with Annotated[...] for fields that track maximum values
    across parallel updates (e.g., iteration counts).
    """
    return max(old, new)


MaxFn = _max_fn


# Helper aggregator that returns the minimum value
def _min_fn(old, new):
    """Aggregator that returns the minimum value.

    Used with Annotated[...] for fields that track minimum values
    across parallel updates (e.g., remaining credits).
    """
    return min(old, new)


MinFn = _min_fn

# Import operator for proper reducer annotations


# Define a custom reducer for sections that accumulates them
def add_sections(current: List["Section"], new: List["Section"]) -> List["Section"]:
    """Custom reducer that accumulates sections without duplicates."""
    if not current:
        return new
    if not new:
        return current

    # Create a set of existing section names for deduplication
    existing_names = {s.name for s in current}

    # Add only new sections that don't already exist
    result = list(current)
    for section in new:
        if section.name not in existing_names:
            result.append(section)
            existing_names.add(section.name)

    return result


# Define a max function for search iterations
def max_value(a: int, b: int) -> int:
    """Return the maximum of two values."""
    return max(a, b)


# Data Models
class Section(BaseModel):
    """Model representing a section of the research report."""

    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(default="", description="The content of the section.")


class Sections(BaseModel):
    """Model representing a collection of report sections."""

    sections: List[Section] = Field(
        description="Sections of the report.",
    )


class SearchQuery(BaseModel):
    """Model representing a single search query."""

    search_query: str = Field(description="Query for web search.")


class Queries(BaseModel):
    """Model representing a collection of search queries."""

    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )


class Feedback(BaseModel):
    """Model representing feedback on report sections."""

    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )


class ClarifyWithUser(BaseModel):
    """Model representing a clarification request for the user."""

    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )


class SectionOutput(BaseModel):
    """Model representing the output of a section writing operation."""

    section_content: str = Field(
        description="The content of the section.",
    )


# Unified State
class DeepResearchState(BaseModel):
    """Main state for the deep research workflow using Pydantic models.

    This state uses proper reducers to handle concurrent updates from parallel nodes.
    """

    # Core fields
    topic: Annotated[str, ReplaceFn] = Field(description="Research topic")

    # Sections management with proper accumulation
    sections: Annotated[List[Section], ReplaceFn] = Field(
        default_factory=list, description="Report sections to write"
    )
    completed_sections: Annotated[List[Section], add_sections] = Field(
        default_factory=list, description="Completed report sections"
    )

    # Feedback and control
    feedback: Optional[List[str]] = Field(
        default=None, description="User feedback on the plan"
    )

    # Report output
    final_report: str = Field(default="", description="Final compiled report")

    # Search state with proper reducers
    search_iterations: Annotated[int, max_value] = Field(
        default=0, description="Number of search iterations completed"
    )
    credits_remaining: Annotated[Optional[int], MinFn] = Field(
        default=None, description="Remaining search credits"
    )

    # Source accumulation with operator.add reducer
    source_str: Annotated[str, operator.add] = Field(
        default="", description="Accumulated source strings"
    )

    class Config:
        # Allow extra fields for forward compatibility
        extra = "allow"


# Multi-Agent Specific States
class MultiAgentReportState(BaseModel):
    """State for multi-agent report generation workflow."""

    model_config = ConfigDict(frozen=True)

    messages: Annotated[List[AnyMessage], operator.add] = Field(
        default_factory=list, description="The message history."
    )
    sections: List[str] = Field(
        default_factory=list, description="List of report sections"
    )
    completed_sections: Annotated[List[Section], operator.add] = Field(
        default_factory=list, description="Send() API key for completed sections"
    )
    final_report: str = Field(default="", description="Final report")
    source_str: Annotated[str, operator.add] = Field(
        default="", description="String of formatted source content from web search"
    )


class MultiAgentSectionState(BaseModel):
    """State for multi-agent section research workflow."""

    model_config = ConfigDict(frozen=True)

    messages: Annotated[List[AnyMessage], operator.add] = Field(
        default_factory=list, description="The message history."
    )
    section: str = Field(default="", description="Report section")
    completed_sections: List[Section] = Field(
        default_factory=list,
        description="Final key we duplicate in outer state for Send() API",
    )
    source_str: str = Field(
        default="", description="String of formatted source content from web search"
    )


class SectionResearchState(BaseModel):
    """State for the section research sub-graph.

    This state is used within the parallel section research workflow.
    """

    # Inherited from parent
    topic: Annotated[str, ReplaceFn] = Field(description="Research topic")
    credits_remaining: Annotated[Optional[int], MinFn] = Field(
        default=None, description="Remaining search credits"
    )

    # Section being researched
    section: Section = Field(description="Current section being researched")

    # Search state
    search_queries: List[str] = Field(
        default_factory=list, description="Current search queries"
    )
    source_str: str = Field(
        default="", description="Accumulated sources for this section"
    )
    search_iterations: int = Field(default=0, description="Iterations for this section")

    # Control flags
    should_continue: bool = Field(
        default=False, description="Whether to continue research"
    )

    # Output field for completed sections (to be passed back to parent graph)
    completed_sections: Annotated[List[Section], add_sections] = Field(
        default_factory=list,
        description="Completed sections to pass back to parent graph",
    )

    class Config:
        extra = "allow"


class FinalSectionWritingState(BaseModel):
    """State for writing final sections that don't require research.

    Used in the parallel final section writing workflow.
    """

    topic: Annotated[str, ReplaceFn] = Field(description="Research topic")
    completed_sections: List[Section] = Field(
        default_factory=list, description="Already completed sections for context"
    )
    current_section: Section = Field(description="Section to write (non-research)")

    class Config:
        extra = "allow"
