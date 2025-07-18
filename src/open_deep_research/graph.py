"""LangGraph workflow implementation for Open Deep Research.

This module defines the workflow nodes and graph for the LangGraph-based research pipeline,
including report planning, section generation, web search, and content compilation.
"""

# NOTE: Keep the public typing imports grouped – we add ``Any`` utility below
from typing import Any, Dict, List, Mapping, TypeVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send

# Third-party/local imports --------------------------------------------------
from open_deep_research.configuration import WorkflowConfiguration
from open_deep_research.core import (
    extract_configuration,
    extract_unique_urls,
    format_sources_section,
    get_search_api_params,
    initialize_model,
)
from open_deep_research.core.config_utils import get_config_value
from open_deep_research.core.format_utils import safe_context
from open_deep_research.core.logging_utils import get_logger
from open_deep_research.model_registry import ModelRole
from open_deep_research.prompt_loader import load_prompt
from open_deep_research.prompts import (
    report_planner_instructions,
    section_writer_inputs,
)
from open_deep_research.pydantic_state import (
    DeepResearchState,
    Feedback,
    Queries,
    Section,
    SectionOutput,
    SectionResearchState,
    Sections,
)
from open_deep_research.search_dispatcher import select_and_execute_search
from open_deep_research.utils import (
    filter_think_tokens,
    format_sections_for_context,
    format_sections_for_final_report,
    get_structured_output_with_fallback,
    get_today_str,
)

logger = get_logger(__name__)

# --- Constants ---
SEARCH_API_NONE = "none"
GRADE_PASS = "pass"
GRADE_FAIL = "fail"
DEFAULT_TOPIC = "Research Topic"
DEFAULT_SECTION_NAME = "Main"
DEFAULT_SECTION_DESC = "Main content"

# --- helper utility for guarded state access ---

T = TypeVar("T")


def get_state_value(
    state: Any, key: str, default: T | None = None, *, required: bool = False
) -> T | None:
    """Safely retrieve a value from *state* whether it's a dict‐like Mapping or an object with attributes.

    Behaviour:
        • Mapping → ``state.get(key, default)``
        • Object  → ``getattr(state, key, default)``

    When *required* is ``True`` and the key is missing or the value is ``None``, a ``ValueError`` is raised.
    """
    # Determine existence and fetch value depending on state type
    if isinstance(state, Mapping):
        exists = key in state
        value = state.get(key, default)
    else:
        exists = hasattr(state, key)
        value = getattr(state, key, default) if exists else default

    # Enforce required constraint
    if required and (not exists or value is None):
        logger.error("Required state key '%s' is missing", key)
        raise ValueError(f"Required state key '{key}' is missing")

    # Log fallback for visibility (debug level to avoid log spam)
    if not exists:
        logger.debug("State key '%s' missing; using default", key)

    return cast("T | None", value)


# --- Common Model Helper ---

def _get_model_for_role(
    configurable: WorkflowConfiguration, role: ModelRole
) -> tuple[BaseChatModel, str | None]:
    """Get and initialize a model for a specific role.
    
    This consolidates the duplicate logic from _get_planner_model, _get_writer_model, etc.
    """
    provider, model_name, model_id = configurable.get_model_for_role(role)
    
    # Get the appropriate kwargs based on role
    model_kwargs = {
        "planner": configurable.planner_model_kwargs,
        "writer": configurable.writer_model_kwargs,
        "reflection": configurable.planner_model_kwargs,  # Re-uses planner kwargs
    }.get(role, {})
    
    model = initialize_model(provider, model_name, model_kwargs or {})
    return model, model_id


# --- Report Planning Helper Functions ---

async def _generate_report_sections(
    model: BaseChatModel, 
    model_id: str | None, 
    topic: str, 
    feedback: str, 
    configurable: WorkflowConfiguration
) -> Sections:
    """Generate report sections using the planner model."""
    system_instructions = report_planner_instructions.format(
        topic=topic,
        report_organization=configurable.report_structure,
        context="",  # Initial planning phase, no context needed
        feedback=feedback,
        today=get_today_str(),
    )
    
    output = await get_structured_output_with_fallback(
        model,
        Sections,
        [
            SystemMessage(content=system_instructions),
            HumanMessage(content="Generate the sections of the report. Your response must include a 'sections' field containing a list of sections."),
        ],
        model_id=model_id,
    )
    return cast(Sections, output)


def _create_fallback_sections(topic: str) -> List[Section]:
    """Create fallback sections when planner fails."""
    logger.warning("Planner model returned no sections – generating fallback plan.")
    return [
        Section(
            name="Introduction",
            description=f"Overview of {topic}",
            research=True,
        )
    ]


def _ensure_research_flags(sections: List[Section]) -> None:
    """Ensure sections have the required research flag set to True (mutates in place)."""
    for section in sections:
        # Pydantic will provide a default False if the field was omitted; we
        # flip it to *True* because the downstream graph expects at least the
        # introduction to trigger the research branch.
        if not getattr(section, "research", False):
            object.__setattr__(section, "research", True)


## Nodes --

async def generate_report_plan(
    state: DeepResearchState, config: RunnableConfig
) -> Dict[str, Any]:
    """Generate a report plan with sections and research queries."""
    from open_deep_research.core.logging_utils import bind_log_context

    bind_log_context(node="generate_report_plan")
    configurable = WorkflowConfiguration.from_runnable_config(config)
    topic = state.topic
    feedback = " ".join(state.feedback or [])

    # Get the planner model
    planner_model, planner_model_id = _get_model_for_role(configurable, "planner")

    # Generate the report sections
    report_sections = await _generate_report_sections(
        planner_model, planner_model_id, topic, feedback, configurable
    )

    # Handle fallback if no sections were generated
    if not report_sections.sections:
        report_sections.sections = _create_fallback_sections(topic)

    # Ensure research flags are set
    _ensure_research_flags(report_sections.sections)

    return {
        "sections": report_sections.sections,
        "credits_remaining": configurable.search_budget,
    }


async def human_feedback(state: DeepResearchState) -> Dict[str, Any]:
    """Wait for human feedback on the report plan."""
    # This is a placeholder for human-in-the-loop interaction
    # In a real app, this would pause and wait for UI input
    # For now, we automatically proceed to the research phase
    return {}


async def generate_queries(
    state: SectionResearchState, config: RunnableConfig
) -> Dict[str, Any]:
    """Generate search queries for researching a specific section."""
    from open_deep_research.core.logging_utils import bind_log_context

    bind_log_context(node="generate_queries")
    configurable = WorkflowConfiguration.from_runnable_config(config)
    # Guarded state access – fall back gracefully while logging
    topic = get_state_value(state, "topic", "Unknown Topic")
    section = cast("Section", get_state_value(state, "section", required=True))

    # 1. Get the writer model (or a dedicated query model if specified)
    writer_model, writer_model_id = _get_model_for_role(configurable, "writer")

    # 2. Generate queries
    system_instructions = load_prompt("query_writer").format(
        topic=topic,
        section_topic=section.description,
        number_of_queries=configurable.number_of_queries,
        today=get_today_str(),
    )

    output_q = await get_structured_output_with_fallback(
        writer_model,
        Queries,
        [
            SystemMessage(content=system_instructions),
            HumanMessage(content="Generate search queries."),
        ],
        model_id=writer_model_id,
    )
    queries = cast("Queries", output_q)

    return {
        "search_queries": [q.search_query for q in queries.queries],
        "section": section,
        "credits_remaining": state.credits_remaining or configurable.search_budget,
    }


# --- Search Web Helper Functions ---

def _calculate_search_cost(params: Dict[str, Any]) -> int:
    """Calculate the cost of a search operation based on parameters."""
    ADVANCED_DEPTH = "advanced"
    DEFAULT_RESULTS = 5
    ADVANCED_MULTIPLIER = 2
    BASIC_MULTIPLIER = 1
    
    depth_label = params.get("search_depth", ADVANCED_DEPTH)
    depth_multiplier = (
        ADVANCED_MULTIPLIER if str(depth_label).lower() == ADVANCED_DEPTH 
        else BASIC_MULTIPLIER
    )
    
    num_results = params.get("num_results") or params.get("max_results") or DEFAULT_RESULTS
    try:
        num_results_int = int(num_results)
    except (ValueError, TypeError):
        num_results_int = DEFAULT_RESULTS

    return depth_multiplier * num_results_int


async def _execute_search(
    search_api: str, query_list: List[str], params: Dict[str, Any]
) -> str:
    """Execute the search operation and handle both sync and async results."""
    search_result = select_and_execute_search(
        search_api, query_list, params
    )

    if hasattr(search_result, "__await__"):
        # The helper is an async function – await the coroutine
        return await search_result
    else:
        # Synchronous stub – use value directly
        return str(search_result)


async def search_web(
    state: SectionResearchState, config: RunnableConfig
) -> Dict[str, Any]:
    """Perform a web search for the generated queries."""
    from open_deep_research.core.logging_utils import bind_log_context

    bind_log_context(node="search_web")
    configurable = WorkflowConfiguration.from_runnable_config(config)
    params = get_search_api_params(configurable)
    search_api_val = get_config_value(configurable.search_api)
    search_api = str(search_api_val) if search_api_val is not None else SEARCH_API_NONE

    query_list = list(state.search_queries)

    # Budget accounting
    credits_remaining = (
        state.credits_remaining
        if state.credits_remaining is not None
        else configurable.search_budget
    )
    
    # Update credits if using a search API
    if search_api != SEARCH_API_NONE:
        cost = _calculate_search_cost(params)
        credits_remaining -= cost

    # Execute search
    source_str = await _execute_search(search_api, query_list, params)

    # Structured logging
    cost_used = (configurable.search_budget - credits_remaining) if search_api != SEARCH_API_NONE else 0
    logger.info(
        "search_web | iter=%s/%s | cost=%s | remaining=%s | section=%s",
        state.search_iterations + 1,
        configurable.max_search_depth,
        cost_used,
        credits_remaining,
        state.section.name if state.section else "-",
    )

    return {
        "source_str": source_str,
        "section": state.section,
        "search_queries": state.search_queries,
        "search_iterations": state.search_iterations + 1,
        "credits_remaining": credits_remaining,
    }


# --- Write Section Helper Functions ---

def _ensure_section_exists(section: Section | None) -> Section:
    """Ensure a section exists, creating a default if necessary."""
    if not section:
        logger.warning("No section found in state, creating default")
        return Section(
            name=DEFAULT_SECTION_NAME, 
            description=DEFAULT_SECTION_DESC, 
            research=True
        )
    return section


async def _write_section_content(
    model: BaseChatModel, 
    model_id: str | None, 
    topic: str, 
    section: Section, 
    source_str: str
) -> str:
    """Write the section content using the writer model."""
    writer_system_message = load_prompt("section_writer").format(
        topic=topic, section_name=section.name
    )
    writer_human_message = section_writer_inputs.format(
        topic=topic,
        section_name=section.name,
        section_topic=section.description,
        context=source_str,
        section_content=section.content,
    )

    section_content_result = cast(
        "SectionOutput",
        await get_structured_output_with_fallback(
            model,
            SectionOutput,
            [
                SystemMessage(content=writer_system_message),
                HumanMessage(content=writer_human_message),
            ],
            model_id=model_id,
        ),
    )
    return filter_think_tokens(section_content_result.section_content)


async def _reflect_on_section(
    model: BaseChatModel, 
    model_id: str | None, 
    topic: str, 
    section: Section, 
    configurable: WorkflowConfiguration
) -> Feedback:
    """Reflect on and grade the section content."""
    section_grader_instructions_formatted = load_prompt("section_grader").format(
        topic=topic,
        section_topic=section.description,
        section=section.content,
        number_of_follow_up_queries=configurable.number_of_queries,
    )
    section_grader_message = (
        "Grade the report section and suggest follow-up queries if needed."
    )

    feedback = cast(
        "Feedback",
        await get_structured_output_with_fallback(
            model,
            Feedback,
            [
                SystemMessage(content=section_grader_instructions_formatted),
                HumanMessage(content=section_grader_message),
            ],
            model_id=model_id,
        ),
    )
    return feedback


def _create_completion_update(
    section: Section, source_str: str, state: SectionResearchState, configurable: WorkflowConfiguration
) -> Dict[str, Any]:
    """Create update dictionary for completed section."""
    update = {
        "completed_sections": [section],
        "credits_remaining": state.credits_remaining or configurable.search_budget,
        "should_continue": False,  # Signal that we're done
    }
    if configurable.include_source_str:
        update["source_str"] = source_str
    return update


def _create_iteration_update(
    section: Section, feedback: Feedback, state: SectionResearchState, configurable: WorkflowConfiguration
) -> Dict[str, Any]:
    """Create update dictionary for another iteration."""
    return {
        "search_queries": [q.search_query for q in feedback.follow_up_queries],
        "section": section,
        "credits_remaining": state.credits_remaining or configurable.search_budget,
        "should_continue": True,  # Signal that we need more research
    }


async def draft_section_content(
    state: SectionResearchState, config: RunnableConfig
) -> Dict[str, Any]:
    """Generate the draft content for a single report section.

    This node performs a single LLM call (writer model) and **does not** mutate
    the incoming Section instance.  Instead it returns an immutable update with
    a *new* Section carrying the generated content so LangGraph can capture the
    state diff cleanly for Studio.
    """
    from open_deep_research.core.logging_utils import bind_log_context

    bind_log_context(node="draft_section_content")

    configurable = WorkflowConfiguration.from_runnable_config(config)

    # Extract basic state – never mutate the original objects
    topic: str = get_state_value(state, "topic", DEFAULT_TOPIC) or DEFAULT_TOPIC
    section: Section = _ensure_section_exists(get_state_value(state, "section"))
    source_str: str = cast("str", get_state_value(state, "source_str", ""))

    # Select the writer model and prepare the context
    writer_model, writer_model_id = _get_model_for_role(configurable, "writer")
    source_str = await safe_context(source_str, target_model=writer_model_id)

    # Call the model – single LLM interaction
    new_content: str = await _write_section_content(
        writer_model, writer_model_id, topic, section, source_str
    )

    # Return a *new* Section object – never mutate in-place
    new_section: Section = section.model_copy(update={"content": new_content})

    return {"section": new_section}


async def grade_section(
    state: SectionResearchState, config: RunnableConfig
) -> Dict[str, Any]:
    """Evaluate a drafted section and decide whether to iterate or complete.

    A single LLM call (reflection model) providing a clear Studio trace.  The
    node either returns a completion update or an iteration update containing
    follow-up queries.  No in-place mutations occur here.
    """
    from open_deep_research.core.logging_utils import bind_log_context

    bind_log_context(node="grade_section")

    configurable = WorkflowConfiguration.from_runnable_config(config)

    # Gather state
    topic: str = get_state_value(state, "topic", DEFAULT_TOPIC) or DEFAULT_TOPIC
    section: Section = _ensure_section_exists(get_state_value(state, "section"))
    source_str: str = cast("str", get_state_value(state, "source_str", ""))
    search_iterations: int = state.search_iterations

    # Reflection model call – single LLM interaction
    reflection_model, reflection_model_id = _get_model_for_role(configurable, "reflection")
    feedback: Feedback = await _reflect_on_section(
        reflection_model, reflection_model_id, topic, section, configurable
    )

    # Decide whether to finish or iterate
    if (
        feedback.grade == GRADE_PASS
        or search_iterations >= configurable.max_search_depth
    ):
        return _create_completion_update(section, source_str, state, configurable)
    else:
        return _create_iteration_update(section, feedback, state, configurable)


async def write_final_sections(state: Any, config: RunnableConfig) -> Dict[str, Any]:
    """Write sections that don't require research using completed sections as context.

    This node handles sections like conclusions or summaries that build on
    the researched sections rather than requiring direct research.

    Args:
        state: Current state with completed sections as context
        config: Configuration for the writing model

    Returns:
        Dict containing the newly written section
    """
    from open_deep_research.core.logging_utils import bind_log_context

    bind_log_context(node="write_final_sections")

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)

    # For backward compatibility, check both 'section' and 'current_section'
    # This allows gradual migration while avoiding the concurrent update issue
    topic = get_state_value(state, "topic", "Research Topic")

    # Try 'current_section' first (new field name to avoid conflicts)
    section = cast("Section | None", get_state_value(state, "current_section"))

    # Fall back to 'section' if 'current_section' is not found
    if not section:
        section = cast(
            "Section",
            get_state_value(
                state,
                "section",
                Section(name="Misc", description="", research=False),
            ),
        )

    completed_sections_raw: List[Section] | None = get_state_value(
        state, "completed_sections", []
    )
    completed_sections: List[Section] = completed_sections_raw or []

    completed_report_sections = format_sections_for_context(completed_sections)

    # Format system instructions
    system_instructions = load_prompt("final_section_writer").format(
        topic=topic,
        section_name=section.name,
        section_topic=section.description,
        context=completed_report_sections,
    )

    # Generate section
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = initialize_model(
        writer_provider, writer_model_name, writer_model_kwargs
    )

    section_content = await writer_model.with_retry(stop_after_attempt=3).ainvoke(
        [
            SystemMessage(content=system_instructions),
            HumanMessage(
                content="Generate a report section based on the provided sources."
            ),
        ]
    )

    # Write content to section
    section.content = cast("str", section_content.content)

    # Only return the fields that should be accumulated
    return {"completed_sections": [section]}


def gather_completed_sections(state: DeepResearchState) -> Dict[str, Any]:
    """Pass-through node kept for graph structure; returns no new state."""
    return {}


# --- Compile Final Report Helper Functions ---

def _organize_sections_by_order(
    sections: List[Section], completed_sections: List[Section]
) -> List[Section]:
    """Organize sections by their original planned order."""
    section_order = [s.name for s in sections]
    completed_sections_map = {s.name: s for s in completed_sections}
    
    # Ensure all sections are present, using original if completion is missing
    organized = []
    for name in section_order:
        section = completed_sections_map.get(name)
        if not section:
            # Find original section as fallback
            section = next((s for s in sections if s.name == name), None)
        if section:
            organized.append(section)
    
    return organized


def _log_compilation_debug_info(
    sections: List[Section], completed_sections: List[Section], final_sections: List[Section]
) -> None:
    """Log debug information about section compilation."""
    logger.info(
        "compile_final_report called with %d sections, %d completed",
        len(sections),
        len(completed_sections),
    )
    
    for i, section in enumerate(sections):
        logger.info(
            "Section %d: %s (research=%s)", i, section.name, section.research
        )
    
    for i, section in enumerate(completed_sections):
        logger.info(
            "Completed %d: %s (content length=%d)",
            i,
            section.name,
            len(section.content),
        )
    
    logger.info("Final sections count: %d", len(final_sections))


def _create_fallback_report_body(final_sections: List[Section], topic: str) -> str:
    """Create a fallback report body when formatting fails."""
    logger.warning("Report body is empty after formatting!")
    report_parts = []
    report_parts.append(f"# Research Report: {topic}\n")

    for section in final_sections:
        if section.content and section.content.strip():
            report_parts.append(f"## {section.name}\n")
            report_parts.append(f"{section.content}\n")

    if len(report_parts) > 1:
        report_body = "\n".join(report_parts)
        logger.info("Constructed fallback report with %d parts", len(report_parts))
        return report_body
    else:
        return "Report generation failed. No section content was available."


def _add_sources_to_report(
    report_body: str, state: DeepResearchState, configurable: WorkflowConfiguration
) -> str:
    """Add sources section to the report if requested."""
    if not configurable.include_source_str:
        logger.debug("Sources not included because include_source_str is False")
        return report_body

    all_source_str = get_state_value(state, "source_str", "")
    if not all_source_str:
        logger.debug("No source_str collected from any sections")
        return report_body

    # Use shared utility for URL extraction and formatting
    unique_urls = extract_unique_urls(all_source_str)
    sources_section = format_sources_section(unique_urls)
    if sources_section:
        report_body += sources_section
        logger.debug("Added sources section with %d URLs", len(unique_urls))
    else:
        logger.debug(
            "Sources section was empty even though we have %d URLs",
            len(unique_urls),
        )

    # Optionally include the raw sources string for richer context
    if configurable.include_raw_source_details:
        report_body += "\n\n---\n\n## Raw Sources\n\n" + all_source_str

    return report_body


def _create_error_report(state: DeepResearchState, error: Exception) -> str:
    """Create an error report when compilation fails."""
    return f"""# Research Report Generation Failed

**Topic:** {get_state_value(state, "topic", "Unknown")}

**Error:** {str(error)}

**Sections Planned:** {len(get_state_value(state, "sections", []) or [])}
**Sections Completed:** {len(get_state_value(state, "completed_sections", []) or [])}

Please check the logs for more details.
"""


async def compile_final_report(
    state: DeepResearchState, config: RunnableConfig
) -> Dict[str, str]:
    """Compile the final report from all completed sections."""
    from open_deep_research.core.logging_utils import bind_log_context

    bind_log_context(node="compile_final_report")
    configurable = cast(
        "WorkflowConfiguration", extract_configuration(config, WorkflowConfiguration)
    )

    try:
        # Organize sections by their original planned order
        sections: List[Section] = get_state_value(state, "sections", []) or []
        completed_sections: List[Section] = (
            get_state_value(state, "completed_sections", []) or []
        )
        final_sections = _organize_sections_by_order(sections, completed_sections)
        
        # Log debug information
        _log_compilation_debug_info(sections, completed_sections, final_sections)

        # Check if we have any content at all
        if not final_sections:
            logger.warning("No sections found for report compilation!")
            return {
                "final_report": "No report could be generated. No sections were completed."
            }

        # Format the sections into a single markdown string
        report_body = format_sections_for_final_report(final_sections)

        logger.info("Report body length: %d characters", len(report_body))
        logger.info(
            "Report body preview: %s",
            report_body[:200] + "..." if len(report_body) > 200 else report_body,
        )

        # If report body is empty, try to construct something from what we have
        if not report_body or report_body.strip() == "":
            topic = get_state_value(state, "topic", "Unknown Topic") or "Unknown Topic"
            report_body = _create_fallback_report_body(final_sections, topic)

        # Add sources if requested
        report_body = _add_sources_to_report(report_body, state, configurable)

        logger.info("Returning final report with %d characters", len(report_body))
        return {"final_report": report_body}

    except Exception as e:
        logger.error("Error in compile_final_report: %s", str(e), exc_info=True)
        error_report = _create_error_report(state, e)
        return {"final_report": error_report}


def initiate_final_section_writing(state: DeepResearchState) -> List[Send]:
    """Create parallel tasks for writing non-research sections.

    This edge function identifies sections that don't need research and
    creates parallel writing tasks for each one. It will only emit the
    `compile_final_report` task once *all* sections (research + non-research)
    have been completed to avoid returning an empty report prematurely.
    """
    sections: List[Section] = get_state_value(state, "sections", []) or []
    completed_sections: List[Section] = (
        get_state_value(state, "completed_sections", []) or []
    )

    # 1) Spawn parallel tasks for sections that don't require additional research
    tasks: List[Send] = [
        Send(
            "write_final_sections",
            {
                "current_section": s,  # use dedicated key to avoid conflicts
                "completed_sections": completed_sections,
            },
        )
        for s in sections
        if not s.research and s.name not in {c.name for c in completed_sections}
    ]

    if tasks:
        # There is still work to do – return the new tasks and *do not* compile the report yet.
        return tasks

    # 2) If there are no pending tasks, check whether all sections have been completed
    if sections and len(completed_sections) == len(sections):
        # All sections are done → compile final report exactly once
        # Pass the necessary state fields to compile_final_report
        return [
            Send(
                "compile_final_report",
                {
                    "sections": sections,
                    "completed_sections": completed_sections,
                    "topic": get_state_value(state, "topic", ""),
                    "source_str": get_state_value(state, "source_str", ""),
                },
            )
        ]

    # 3) Otherwise, not ready to compile and no new tasks to emit
    return []


def initiate_section_research(state: DeepResearchState) -> List[Send]:
    """Create parallel tasks for researching sections that require research."""
    sections: List[Section] = get_state_value(state, "sections", []) or []
    credits_remaining: int = cast(int, get_state_value(state, "credits_remaining", 100))
    topic: str = get_state_value(state, "topic", "Research Topic") or "Research Topic"

    return [
        Send(
            "build_section_with_web_research",
            {
                # Include topic as it's required by SectionResearchState
                "topic": topic,
                "section": s,
                # Remove search_iterations initialization to avoid concurrent updates
                # It will be properly initialized by the MaxFn reducer
                "credits_remaining": credits_remaining,
            },
        )
        for s in sections
        if s.research
    ]


# Report section sub-graph --

# Add nodes
section_builder = StateGraph(SectionResearchState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node(
    "search_web",
    search_web,
    retry=RetryPolicy(max_attempts=3),
)  # type: ignore[arg-type]
# Remove monolithic write_section node
# section_builder.add_node("write_section", write_section)
# Add new thin nodes for Studio alignment
section_builder.add_node("draft_section_content", draft_section_content)
section_builder.add_node("grade_section", grade_section)


# Add conditional edge function for the reflection loop
def should_continue_research(state: SectionResearchState) -> str:
    """Determine if we should continue research or end the section."""
    # Check if we should continue based on the flag set by write_section
    should_continue = getattr(state, "should_continue", False)
    if should_continue:
        return "search_web"
    else:
        return END


# Rewrite edges to incorporate new nodes
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
# section_builder.add_edge("search_web", "write_section")
section_builder.add_edge("search_web", "draft_section_content")
section_builder.add_edge("draft_section_content", "grade_section")
# section_builder.add_conditional_edges("write_section", should_continue_research)
section_builder.add_conditional_edges("grade_section", should_continue_research)  # type: ignore[arg-type]

# Outer graph for initial report plan compiling results from each section --

# Add nodes
builder = StateGraph(DeepResearchState, config_schema=WorkflowConfiguration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
# ``compile()`` returns a ``Runnable`` but the node API expects the same so the
# ignore is no longer necessary with recent LangGraph versions – remove it.
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_section_research)  # type: ignore[arg-type]
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing)  # type: ignore[arg-type]
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()
