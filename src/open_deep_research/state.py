"""Legacy state management for Open Deep Research application.

This module contains legacy TypedDict-based state definitions that were used
before the migration to Pydantic-based immutable state models.
"""
from typing import List, Literal

from pydantic import BaseModel, Field


class Section(BaseModel):
    """Legacy model representing a section of the research report."""
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(
        default="",
        description="The content of the section."
    )

class Sections(BaseModel):
    """Legacy model representing a collection of report sections."""
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

class SearchQuery(BaseModel):
    """Legacy model representing a single search query."""
    search_query: str | None = Field(None, description="Query for web search.")

class Queries(BaseModel):
    """Legacy model representing a collection of search queries."""
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

class Feedback(BaseModel):
    """Legacy model representing feedback on report sections."""
    grade: Literal["pass","fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )
