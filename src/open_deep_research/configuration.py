import os
from enum import Enum
from dataclasses import dataclass, fields, field
from typing import Any, Optional, Dict, Literal

from langchain_core.runnables import RunnableConfig
from open_deep_research.model_registry import ModelRole

DEFAULT_REPORT_STRUCTURE = """Create a concise report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Key Findings (research needed)
   - Essential information and insights about the topic
   
3. Conclusion (no research needed)
   - Summary of main points in a clear, actionable format"""

COMPREHENSIVE_REPORT_STRUCTURE = """Create a comprehensive, multi-section report on the user-provided topic:

1. Introduction (no research needed)
   - Background, context, and rationale for the report
   - Clearly state the report's purpose and scope

2. Main Section 1: In-depth Analysis (research needed)
   - Detailed examination of a core aspect of the topic
   - Use data, examples, and evidence to support claims

3. Main Section 2: Comparative View (research needed)
   - Compare and contrast different facets of the topic
   - Highlight similarities, differences, and trade-offs

4. Main Section 3: Future Outlook (research needed)
   - Discuss future trends, challenges, and opportunities
   - Provide forward-looking insights

5. Conclusion (no research needed)
   - Summarize the key findings from all sections
   - Offer final recommendations or a concluding perspective"""

EXECUTIVE_SUMMARY_STRUCTURE = """Create a high-level executive summary on the user-provided topic:

1. Overview (no research needed)
   - State the main issue and the report's primary conclusion upfront

2. Key Findings (research needed)
   - A bulleted list of the most critical facts, findings, and data points
   - Keep each point brief and impactful

3. Recommendations (no research needed)
   - Actionable recommendations based on the key findings
   - Suggest clear next steps"""

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"
    DUCKDUCKGO = "duckduckgo"
    GOOGLESEARCH = "googlesearch"
    NONE = "none"

@dataclass(kw_only=True)
class BaseConfiguration:
    """Base configuration with common fields."""
    report_structure: str = DEFAULT_REPORT_STRUCTURE
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None
    summarization_model_provider: str | None = None
    summarization_model: str | None = None
    max_structured_output_retries: int = 3
    include_source_str: bool = True

@dataclass(kw_only=True)
class WorkflowConfiguration(BaseConfiguration):
    """Configuration for the workflow/graph-based implementation (graph.py)."""
    # Workflow-specific configuration
    number_of_queries: int = 1 # Number of search queries to generate per iteration
    max_search_depth: int = 2 # Maximum number of reflection + search iterations
    planner_provider: str = "together"
    planner_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    planner_model_kwargs: Optional[Dict[str, Any]] = None
    writer_provider: str = "together"
    writer_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    writer_model_kwargs: Optional[Dict[str, Any]] = None
    reflection_model_provider: str | None = None
    reflection_model: str | None = None

    def get_model_for_role(self, role: ModelRole) -> tuple[str, str, str | None]:
        """Returns (provider, model_name, full_model_id) for a given role."""
        if role == "planner":
            provider = self.planner_provider
            model = self.planner_model
        elif role == "writer":
            provider = self.writer_provider
            model = self.writer_model
        elif role == "summarizer":
            # Fall back to writer model if summarizer isn't set
            provider = self.summarization_model_provider or self.writer_provider
            model = self.summarization_model or self.writer_model
        elif role == "reflection":
            # Use writer model for reflection instead of falling back to planner
            # This keeps us within the user's selected model combination
            provider = self.reflection_model_provider or self.writer_provider
            model = self.reflection_model or self.writer_model
        else:
            raise ValueError(f"Unknown model role: {role}")
        
        return provider, model, f"{provider}:{model}"

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "WorkflowConfiguration":
        """Create a WorkflowConfiguration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {}
        
        for f in fields(cls):
            if not f.init:
                continue
            
            # Get value from environment variables or configurable
            env_value = os.environ.get(f.name.upper())
            config_value = configurable.get(f.name)
            
            # Use the first non-None value, or skip if both are None
            if env_value is not None:
                values[f.name] = env_value
            elif config_value is not None:
                values[f.name] = config_value
            # If both are None, don't include in values dict so default is used
        
        return cls(**values)

@dataclass(kw_only=True)
class MultiAgentConfiguration(BaseConfiguration):
    """Configuration for the multi-agent implementation (multi_agent.py)."""
    # Override some base defaults for multi-agent
    summarization_model_provider: str = "together"
    summarization_model: str = "Qwen/Qwen2.5-72B-Instruct-Turbo"
    
    # Multi-agent specific configuration
    number_of_queries: int = 1 # Number of search queries to generate per section
    supervisor_model: str = "together:meta-llama/Llama-3.3-70B-Instruct-Turbo"
    researcher_model: str = "together:meta-llama/Llama-3.3-70B-Instruct-Turbo"
    ask_for_clarification: bool = False # Whether to ask for clarification from the user
    # MCP server configuration
    mcp_server_config: Optional[Dict[str, Any]] = None
    mcp_prompt: Optional[str] = None
    mcp_tools_to_include: Optional[list[str]] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "MultiAgentConfiguration":
        """Create a MultiAgentConfiguration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {}
        
        for f in fields(cls):
            if not f.init:
                continue
            
            # Get value from environment variables or configurable
            env_value = os.environ.get(f.name.upper())
            config_value = configurable.get(f.name)
            
            # Use the first non-None value, or skip if both are None
            if env_value is not None:
                values[f.name] = env_value
            elif config_value is not None:
                values[f.name] = config_value
            # If both are None, don't include in values dict so default is used
        
        return cls(**values)

# Keep the old Configuration class for backward compatibility
Configuration = WorkflowConfiguration
