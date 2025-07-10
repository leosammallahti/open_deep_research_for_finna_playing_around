"""LangGraph workflow implementation for Open Deep Research.

This module defines the workflow nodes and graph for the LangGraph-based research pipeline,
including report planning, section generation, web search, and content compilation.
"""

# NOTE: Keep the public typing imports grouped – we add ``Any`` utility below
from typing import Any, Dict, List, Mapping, TypeVar, Union, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy, Command, RetryPolicy, Send

# We purposely import the utils module itself so we can access functions that
# might be monkey-patched during tests (e.g. ``select_and_execute_search``).
import open_deep_research.utils as _odr_utils
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
from open_deep_research.exceptions import OutOfBudgetError
from open_deep_research.prompts import (
    final_section_writer_instructions,
    query_writer_instructions,
    report_planner_instructions,
    section_grader_instructions,
    section_writer_inputs,
    section_writer_instructions,
)
from open_deep_research.pydantic_state import (
    DeepResearchState,
    Feedback,
    Queries,
    Section,
    SectionOutput,
    Sections,
)
from open_deep_research.utils import (
    filter_think_tokens,
    format_sections_for_context,
    format_sections_for_final_report,
    get_structured_output_with_fallback,
    get_today_str,
)

logger = get_logger(__name__)

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


# Import the new Pydantic version of the state used by the section-level
# sub-graph.  This unifies state handling under Pydantic models while we
# incrementally remove legacy dict-style access.
from open_deep_research.pydantic_state import (
    SectionResearchState,
    FinalSectionWritingState,
)

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

    # 1. Get the planner model
    planner_provider, planner_model_name, planner_model_id = (
        configurable.get_model_for_role("planner")
    )
    planner_llm = initialize_model(
        planner_provider, planner_model_name, configurable.planner_model_kwargs
    )

    # 2. Generate the report sections
    system_instructions_sections = report_planner_instructions.format(
        topic=topic,
        report_organization=configurable.report_structure,
        context="",  # Initial planning phase, no context needed
        feedback=feedback,
        today=get_today_str(),
    )
    planner_message = "Generate the sections of the report. Your response must include a 'sections' field containing a list of sections."

    output = await get_structured_output_with_fallback(
        planner_llm,
        Sections,
        [
            SystemMessage(content=system_instructions_sections),
            HumanMessage(content=planner_message),
        ],
        model_id=planner_model_id,
    )
    report_sections = cast("Sections", output)

    # --- Fallback Handling -------------------------------------------------
    # If the planner fails to produce any sections (e.g. due to missing API
    # keys or parsing errors) we still want the workflow to progress instead
    # of silently completing with an empty final report.  In that case we
    # create a minimal single‐section plan that can be filled in by the
    # downstream nodes.  This keeps the user experience smooth and surfaces
    # a meaningful report even when advanced planning models are unavailable.
    if not report_sections.sections:
        logger.warning("Planner model returned no sections – generating fallback plan.")
        report_sections.sections = [
            Section(
                name="Introduction",
                description=f"Overview of {topic}",
                research=True,
            )
        ]

    # Ensure the 'research' flag is present and defaults to True when missing
    for s in report_sections.sections:
        # Pydantic will provide a default False if the field was omitted; we
        # flip it to *True* because the downstream graph expects at least the
        # introduction to trigger the research branch.
        if not hasattr(s, "research") or s.research is False:
            object.__setattr__(s, "research", True)

    # Initialise credits_remaining in state using the configured search_budget
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
    writer_provider, writer_model_name, writer_model_id = (
        configurable.get_model_for_role("writer")
    )
    writer_model = initialize_model(
        writer_provider, writer_model_name, configurable.writer_model_kwargs
    )

    # 2. Generate queries
    system_instructions = query_writer_instructions.format(
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


async def search_web(
    state: SectionResearchState, config: RunnableConfig
) -> Dict[str, Any]:
    """Perform a web search for the generated queries."""
    from open_deep_research.core.logging_utils import bind_log_context

    bind_log_context(node="search_web")
    configurable = WorkflowConfiguration.from_runnable_config(config)
    params_to_pass = get_search_api_params(configurable)
    search_api_val = get_config_value(configurable.search_api)
    search_api = str(search_api_val) if search_api_val is not None else "none"

    # Fetch current queries directly
    query_list = list(state.search_queries)

    # Budget accounting -----------------------------------------------------
    credits_remaining = (
        state.credits_remaining
        if state.credits_remaining is not None
        else configurable.search_budget
    )

    if search_api != "none":
        depth_label = params_to_pass.get("search_depth", "advanced")
        # Bug fix: A "basic" or "standard" search costs 1 credit, not 2.
        # Check explicitly for "advanced" to get the 2x multiplier.
        depth_multiplier = 2 if str(depth_label).lower() == "advanced" else 1
        num_results = (
            params_to_pass.get("num_results") or params_to_pass.get("max_results") or 5
        )
        try:
            num_results_int = int(num_results)
        except Exception:
            num_results_int = 5

        cost = depth_multiplier * num_results_int

        # TEMPORARILY DISABLED: Search budget enforcement
        # if cost > credits_remaining:
        #     logger.warning(
        #         "Search budget (%s) exhausted – cost %s > remaining %s",
        #         configurable.search_budget,
        #         cost,
        #         credits_remaining,
        #     )
        #     raise OutOfBudgetError(remaining=credits_remaining)

        credits_remaining -= cost

    # ----------------------------------------------------------------------

    # Run the web search helper – it might be patched to a synchronous lambda
    # in unit tests, so we detect coroutine results at runtime.
    # ``select_and_execute_search`` may return either a string or a coroutine
    # depending on whether the underlying implementation is synchronous or
    # asynchronous.  We type it as ``Any`` and resolve at runtime.
    search_result: Any = _odr_utils.select_and_execute_search(
        search_api,
        query_list,
        params_to_pass,
    )

    if hasattr(search_result, "__await__"):
        # The helper is an async function – await the coroutine.
        source_str = await search_result
    else:
        # Synchronous stub – use value directly.
        source_str = cast(str, search_result)

    # Structured logging ----------------------------------------------------
    logger.info(
        "search_web | iter=%s/%s | cost=%s | remaining=%s | section=%s",
        state.search_iterations + 1,
        configurable.max_search_depth,
        (configurable.search_budget - credits_remaining) if search_api != "none" else 0,
        credits_remaining,
        state.section.name if state.section else "-",
    )

    return {
        "source_str": source_str,
        "section": state.section,
        "search_queries": state.search_queries,
        # Increment the search depth counter to ensure the reflection loop can terminate
        "search_iterations": state.search_iterations + 1,
        "credits_remaining": credits_remaining,
    }


async def write_section(
    state: SectionResearchState, config: RunnableConfig
) -> Union[Dict[str, Any], Command[Any]]:
    """Write a section of the report based on search results and reflect on it."""
    from open_deep_research.core.logging_utils import bind_log_context

    bind_log_context(node="write_section")
    configurable = WorkflowConfiguration.from_runnable_config(config)
    # Guarded state access with sensible defaults
    topic = get_state_value(state, "topic", "Research Topic")
    section = get_state_value(state, "section")
    source_str = cast("str", get_state_value(state, "source_str", ""))
    search_iterations = state.search_iterations

    # Ensure section exists
    if not section:
        # Should rarely happen but keep behaviour consistent
        logger.warning("No section found in state, creating default")
        section = Section(name="Main", description="Main content", research=True)

    # 1. Get the writer model
    writer_provider, writer_model_name, writer_model_id = (
        configurable.get_model_for_role("writer")
    )
    writer_model = initialize_model(
        writer_provider, writer_model_name, configurable.writer_model_kwargs or {}
    )

    # 2. Truncate source_str if needed for models with limited context
    # Ensure context fits model window
    source_str = await safe_context(source_str, target_model=writer_model_id)

    # 3. Write the section content
    writer_system_message = section_writer_instructions.format(
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
            writer_model,
            SectionOutput,
            [
                SystemMessage(content=writer_system_message),
                HumanMessage(content=writer_human_message),
            ],
            model_id=writer_model_id,
        ),
    )
    section.content = filter_think_tokens(section_content_result.section_content)

    # 4. Get the reflection model
    reflection_provider, reflection_model_name, reflection_model_id = (
        configurable.get_model_for_role("reflection")
    )
    reflection_model = initialize_model(
        reflection_provider,
        reflection_model_name,
        configurable.planner_model_kwargs or {},  # Re-use planner kwargs for reflection
    )

    # 5. Reflect and grade the section
    section_grader_instructions_formatted = section_grader_instructions.format(
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
            reflection_model,
            Feedback,
            [
                SystemMessage(content=section_grader_instructions_formatted),
                HumanMessage(content=section_grader_message),
            ],
            model_id=reflection_model_id,
        ),
    )

    # 6. Decide whether to finish or iterate
    # If the section is passing or the max search depth is reached, publish the section to completed sections
    if feedback.grade == "pass" or search_iterations >= configurable.max_search_depth:
        # Create a dictionary with only the fields that need to be accumulated.
        update = {
            "completed_sections": [section],
            "credits_remaining": state.credits_remaining or configurable.search_budget,
            "should_continue": False,  # Signal that we're done
        }
        if configurable.include_source_str:
            update["source_str"] = source_str

        # Return the update dictionary. The conditional edge will route to END.
        return update

    # If more research is needed, prepare for another iteration
    else:
        return {
            "search_queries": [q.search_query for q in feedback.follow_up_queries],
            "section": section,
            "credits_remaining": state.credits_remaining or configurable.search_budget,
            "should_continue": True,  # Signal that we need more research
        }


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
    system_instructions = final_section_writer_instructions.format(
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
        # Sort sections by their original planned order
        # Use get_state_value to handle both dict and Pydantic model inputs
        sections: List[Section] = get_state_value(state, "sections", []) or []
        section_order = [s.name for s in sections]

        # Use a dictionary for quick lookup and update
        completed_sections: List[Section] = (
            get_state_value(state, "completed_sections", []) or []
        )
        completed_sections_map = {s.name: s for s in completed_sections}

        # Debug logging
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

        # Ensure all sections are present, using original if completion is missing.
        # We build an intermediate list that may contain ``None`` then immediately
        # filter it so the public ``final_sections`` variable is strictly
        # ``List[Section]`` for downstream type‐safety.
        _maybe_sections = [
            completed_sections_map.get(
                name, next((s for s in sections if s.name == name), None)
            )
            for name in section_order
        ]

        final_sections: List[Section] = [s for s in _maybe_sections if s is not None]

        logger.info("Final sections count: %d", len(final_sections))

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
            logger.warning("Report body is empty after formatting!")
            # Try to construct a basic report from completed sections
            report_parts = []
            report_parts.append(
                f"# Research Report: {get_state_value(state, 'topic', 'Unknown Topic')}\n"
            )

            for section in final_sections:
                if section.content and section.content.strip():
                    report_parts.append(f"## {section.name}\n")
                    report_parts.append(f"{section.content}\n")

            if len(report_parts) > 1:
                report_body = "\n".join(report_parts)
                logger.info(
                    "Constructed fallback report with %d parts", len(report_parts)
                )
            else:
                report_body = (
                    "Report generation failed. No section content was available."
                )

        # Add sources if requested
        if configurable.include_source_str:
            # The source_str field accumulates all sources from parallel sub-graphs
            # due to the Annotated[str, operator.add] in the state definition
            all_source_str = get_state_value(state, "source_str", "")
            if all_source_str:
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

                # Optionally include the raw sources string for richer context.
                # This block can be **very** large, so we expose a dedicated
                # configuration flag (``include_raw_source_details``) to let
                # users disable it while still keeping the numbered citations
                # above.
                if configurable.include_raw_source_details:
                    report_body += "\n\n---\n\n## Raw Sources\n\n" + all_source_str
            else:
                logger.debug("No source_str collected from any sections")
        else:
            logger.debug("Sources not included because include_source_str is False")

        logger.info("Returning final report with %d characters", len(report_body))
        return {"final_report": report_body}

    except Exception as e:
        logger.error("Error in compile_final_report: %s", str(e), exc_info=True)
        # Return an error report rather than failing silently
        error_report = f"""# Research Report Generation Failed

**Topic:** {get_state_value(state, "topic", "Unknown")}

**Error:** {str(e)}

**Sections Planned:** {len(get_state_value(state, "sections", []) or [])}
**Sections Completed:** {len(get_state_value(state, "completed_sections", []) or [])}

Please check the logs for more details.
"""
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
    # Retry on transient errors (network issues, API hiccups, etc.)
    retry=RetryPolicy(max_attempts=3),
)
section_builder.add_node("write_section", write_section)


# Add conditional edge function for the reflection loop
def should_continue_research(state: SectionResearchState) -> str:
    """Determine if we should continue research or end the section."""
    # Check if we should continue based on the flag set by write_section
    should_continue = getattr(state, "should_continue", False)
    if should_continue:
        return "search_web"
    else:
        return END


# Add edges with proper conditional logic
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")
section_builder.add_conditional_edges("write_section", should_continue_research)

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
builder.add_conditional_edges(
    "gather_completed_sections", initiate_final_section_writing
)  # type: ignore[arg-type]
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()
