from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.messages import AnyMessage

# Data Models
class Section(BaseModel):
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
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

class SearchQuery(BaseModel):
    search_query: str = Field(description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

class Feedback(BaseModel):
    grade: Literal["pass","fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )

class ClarifyWithUser(BaseModel):
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )

class SectionOutput(BaseModel):
    section_content: str = Field(
        description="The content of the section.",
    )

# Unified State
class DeepResearchState(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: List[AnyMessage] = Field(default_factory=list, description="The message history.")
    
    topic: str = Field(default="", description="The topic of the research report.")
    
    already_clarified_topic: Optional[bool] = Field(default=None, description="Whether the topic has been clarified with the user.")
    
    sections: List[Section] = Field(default_factory=list, description="The sections of the report to be generated.")
    
    initial_sections: List[Section] = Field(default_factory=list, description="The initial sections of the report for reference.")
    
    completed_sections: List[Section] = Field(default_factory=list, description="The sections of the report that have been completed.")
    
    final_report: str = Field(default="", description="The final generated report.")
    
    feedback: List[str] = Field(default_factory=list, description="Feedback on the generated report.")
    
    feedback_on_report_plan: List[str] = Field(default_factory=list, description="Feedback on the proposed report plan.")
    
    sources: List[str] = Field(default_factory=list, description="A list of sources used for the report.")
    
    report_sections_from_research: str = Field(default="", description="Content generated from research for sections.")
    
    source_str: str = Field(default="", description="A string of formatted source content from web search for evaluation.")

    search_iterations: int = Field(default=0, description="The number of search iterations performed.")
    
    # For section-specific processing
    section: Optional[Section] = Field(default=None, description="The current section being processed.")
    
    search_queries: List[SearchQuery] = Field(default_factory=list, description="List of search queries for the current section.") 