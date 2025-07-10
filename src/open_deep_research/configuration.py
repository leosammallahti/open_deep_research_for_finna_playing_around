"""Configuration management for Open Deep Research application.

This module provides configuration classes and enums for managing research workflows,
search providers, model providers, and report structure settings.
"""

# NOTE: We now rely on settings for env-backed defaults
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, Literal

from langchain_core.runnables import RunnableConfig

from open_deep_research.core.settings import settings
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
    """Enumeration of supported search API providers."""

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
    search_api_config: Dict[str, Any] | None = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None
    summarization_model_provider: str | None = None
    summarization_model: str | None = None
    max_structured_output_retries: int = 3
    include_source_str: bool = True
    # Whether to append the full unprocessed "Raw Sources" block (potentially
    # very large) at the end of the final report.  When set to *False* the
    # report will still include the numbered citations section – controlled by
    # ``include_source_str`` – but will omit the verbose raw dump.
    include_raw_source_details: bool = True


@dataclass(kw_only=True)
class WorkflowConfiguration(BaseConfiguration):
    """Configuration for the workflow/graph-based implementation (graph.py)."""

    # Workflow-specific configuration
    number_of_queries: int = 1  # Number of search queries to generate per iteration
    max_search_depth: int = 2  # Maximum number of reflection + search iterations
    clarify_with_user: bool = False  # Whether to ask for clarification from the user
    sections_user_approval: bool = (
        False  # Whether to require user approval of report plan
    )
    planner_provider: str = "together"
    planner_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    planner_model_kwargs: Dict[str, Any] | None = None
    writer_provider: str = "together"
    writer_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    writer_model_kwargs: Dict[str, Any] | None = None
    reflection_model_provider: str | None = None
    reflection_model: str | None = None
    search_budget: int = 100  # Total credits for web search during a single run

    def get_model_for_role(self, role: ModelRole) -> tuple[str, str, str | None]:
        """Get (provider, model_name, full_model_id) for a given role."""
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
        cls, config: RunnableConfig | None = None
    ) -> "WorkflowConfiguration":
        """Create a WorkflowConfiguration instance from a RunnableConfig."""
        # Safely extract the configurable mapping – runtime may pass in non-dict objects
        configurable: dict[str, Any] = {}
        if config and isinstance(config, dict):
            configurable = config.get("configurable", {})
        values: dict[str, Any] = {}

        for f in fields(cls):
            if not f.init:
                continue

            # Preference order: explicit override in RunnableConfig → settings default
            # Special case for 'features' which is itself a dict that should
            # *merge* with defaults instead of replacing entirely.
            if f.name == "features":
                # Pull user-supplied overrides (may be missing)
                user_feat = (
                    configurable.get("features", {})
                    if isinstance(configurable.get("features", {}), dict)
                    else {}
                )
                # Default will be injected in __post_init__, so we just pass
                # the user dict to let __post_init__ merge/validate.
                if user_feat:
                    values[f.name] = user_feat
                continue

            config_value = configurable.get(f.name)
            settings_default = getattr(settings, f.name, None)

            if config_value is not None:
                values[f.name] = config_value
            elif settings_default is not None:
                values[f.name] = settings_default

        return cls(**values)

    # ------------------------------------------------------------------
    # Optional feature toggles (validated against compatibility matrix)
    # ------------------------------------------------------------------

    # We place this field *after* from_runnable_config so the default is
    # visible to that factory.
    features: Dict[str, bool] | None = None

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:  # noqa: D401 – keep dataclass hook name
        """Run compatibility validation after init."""

        # Dataclasses call __post_init__ *after* fields are set but before
        # the instance is considered frozen, so we can safely mutate.

        default_map = {
            "clarification": False,
            "human_feedback": True,
            "parallel_research": False,
            "section_grading": True,
            "mcp_support": False,
        }
        if self.features is None:
            self.features = default_map
        else:
            # Merge defaults with user overrides – user keys win
            merged = {**default_map, **self.features}
            self.features = merged

        # ------------------------------------------------------------------
        # Validation (imported locally to avoid circular deps)
        # ------------------------------------------------------------------
        from open_deep_research.feature_compatibility import FeatureCompatibility

        errs = FeatureCompatibility.validate_features(self.features)
        errs += FeatureCompatibility.validate_mode("workflow", self.features)

        if errs:
            joined = "; ".join(errs)
            raise ValueError(f"Invalid feature configuration: {joined}")


@dataclass(kw_only=True)
class MultiAgentConfiguration(BaseConfiguration):
    """Configuration for the multi-agent implementation (multi_agent.py)."""

    # Override some base defaults for multi-agent
    summarization_model_provider: str = "together"
    summarization_model: str = "Qwen/Qwen2.5-72B-Instruct-Turbo"

    # Multi-agent specific configuration
    number_of_queries: int = 1  # Number of search queries to generate per section
    supervisor_model: str = "together:meta-llama/Llama-3.3-70B-Instruct-Turbo"
    researcher_model: str = "together:meta-llama/Llama-3.3-70B-Instruct-Turbo"
    ask_for_clarification: bool = (
        False  # Whether to ask for clarification from the user
    )
    # MCP server configuration
    mcp_server_config: Dict[str, Any] | None = None
    mcp_prompt: str | None = None
    mcp_tools_to_include: list[str] | None = None

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "MultiAgentConfiguration":
        """Create a MultiAgentConfiguration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {}

        for f in fields(cls):
            if not f.init:
                continue

            if f.name == "features":
                if isinstance(configurable.get("features"), dict):
                    values[f.name] = configurable["features"]
                continue

            config_value = configurable.get(f.name)
            settings_default = getattr(settings, f.name, None)

            if config_value is not None:
                values[f.name] = config_value
            elif settings_default is not None:
                values[f.name] = settings_default

        return cls(**values)

    # ------------------------------------------------------------------
    # Feature toggles – defaults tailored for multi-agent style
    # ------------------------------------------------------------------

    features: Dict[str, bool] | None = None

    def __post_init__(self) -> None:  # noqa: D401
        default_map = {
            "clarification": False,
            "human_feedback": False,
            "parallel_research": True,
            "section_grading": False,
            "mcp_support": True,
        }
        if self.features is None:
            self.features = default_map
        else:
            self.features = {**default_map, **self.features}

        from open_deep_research.feature_compatibility import FeatureCompatibility

        errs = FeatureCompatibility.validate_features(self.features)
        errs += FeatureCompatibility.validate_mode("multi_agent", self.features)

        if errs:
            raise ValueError("Invalid feature configuration: " + "; ".join(errs))


# Legacy alias removed - use WorkflowConfiguration or MultiAgentConfiguration explicitly
