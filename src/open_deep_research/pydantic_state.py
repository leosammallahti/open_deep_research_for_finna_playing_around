"""Pydantic state models for Open Deep Research application.

This module defines immutable state models using Pydantic for managing research workflow state,
including report sections, search queries, feedback, and multi-agent workflow states.
"""
import operator
from typing import Annotated, List, Literal

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, ConfigDict, Field


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
    content: str = Field(
        default="",
        description="The content of the section."
    )

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
    grade: Literal["pass","fail"] = Field(
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
    """Immutable state model for the deep research workflow."""
    model_config = ConfigDict(frozen=True)

    messages: Annotated[List[AnyMessage], operator.add] = Field(default_factory=list, description="The message history.")
    
    topic: str = Field(default="", description="The topic of the research report.")
    
    already_clarified_topic: bool | None = Field(default=None, description="Whether the topic has been clarified with the user.")
    
    sections: List[Section] = Field(default_factory=list, description="The sections of the report to be generated.")
    
    initial_sections: List[Section] = Field(default_factory=list, description="The initial sections of the report for reference.")
    
    completed_sections: Annotated[List[Section], operator.add] = Field(default_factory=list, description="The sections of the report that have been completed.")
    
    final_report: str = Field(default="", description="The final generated report.")
    
    feedback: List[str] = Field(default_factory=list, description="Feedback on the generated report.")
    
    feedback_on_report_plan: List[str] = Field(default_factory=list, description="Feedback on the proposed report plan.")
    
    sources: Annotated[List[str], operator.add] = Field(default_factory=list, description="A list of sources used for the report.")
    
    report_sections_from_research: str = Field(default="", description="Content generated from research for sections.")
    
    source_str: Annotated[str, operator.add] = Field(default="", description="A string of formatted source content from web search for evaluation.")

    search_iterations: int = Field(default=0, description="The number of search iterations performed.")
    
    # For section-specific processing
    section: Section | None = Field(default=None, description="The current section being processed.")
    
    search_queries: List[str] = Field(default_factory=list, description="List of search queries for the current section.") 

# Multi-Agent Specific States
class MultiAgentReportState(BaseModel):
    """State for multi-agent report generation workflow."""
    model_config = ConfigDict(frozen=True)
    
    messages: Annotated[List[AnyMessage], operator.add] = Field(
        default_factory=list, 
        description="The message history."
    )
    sections: List[str] = Field(
        default_factory=list,
        description="List of report sections"
    )
    completed_sections: Annotated[List[Section], operator.add] = Field(
        default_factory=list,
        description="Send() API key for completed sections"
    )
    final_report: str = Field(
        default="",
        description="Final report"
    )
    source_str: Annotated[str, operator.add] = Field(
        default="",
        description="String of formatted source content from web search"
    )

class MultiAgentSectionState(BaseModel):
    """State for multi-agent section research workflow."""
    model_config = ConfigDict(frozen=True)
    
    messages: Annotated[List[AnyMessage], operator.add] = Field(
        default_factory=list,
        description="The message history."
    )
    section: str = Field(
        default="",
        description="Report section"
    )
    completed_sections: List[Section] = Field(
        default_factory=list,
        description="Final key we duplicate in outer state for Send() API"
    )
    source_str: str = Field(
        default="",
        description="String of formatted source content from web search"
    ) 