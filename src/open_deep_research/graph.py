"""LangGraph workflow implementation for Open Deep Research.

This module defines the workflow nodes and graph for the LangGraph-based research pipeline,
including report planning, section generation, web search, and content compilation.
"""
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from open_deep_research.configuration import WorkflowConfiguration
from open_deep_research.core import (
    extract_configuration,
    extract_unique_urls,
    format_sources_section,
    get_search_api_params,
    initialize_model,
)
from open_deep_research.core.logging_utils import get_logger
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
from open_deep_research.core.config_utils import get_config_value
from open_deep_research.utils import (
    filter_think_tokens,
    format_sections_for_context,
    format_sections_for_final_report,
    get_structured_output_with_fallback,
    get_today_str,
    select_and_execute_search,
    summarize_search_results,
)

logger = get_logger(__name__)

# --- helper utility for guarded state access ---
from typing import Any, Mapping, TypeVar

T = TypeVar("T")

def get_state_value(state: Mapping[str, Any], key: str, default: T | None = None, *, required: bool = False) -> T | None:
    """Retrieve a value from the section-state dict with optional fallback and requirement check.

    Args:
        state: Mapping containing state values.
        key: Key to look for.
        default: Value to return if key is missing (and *required* is False).
        required: If True, raise ValueError when the key is absent or value is None.

    Returns:
        The requested value or *default*.

    Raises:
        ValueError: If *required* is True and value is missing.
    """
    value = state.get(key, default)
    if required and value is None:
        logger.error("Required state key '%s' is missing", key)
        raise ValueError(f"Required state key '{key}' is missing")
    if value is default and key not in state:
        # log once for visibility, but don't flood logs on every access
        logger.warning("State key '%s' missing; using default", key)
    return value


class SectionResearchState(TypedDict):
    """State for the section research sub-graph."""
    topic: str
    section: Section
    search_queries: list[str]
    source_str: str
    search_iterations: int


## Nodes -- 

async def generate_report_plan(state: DeepResearchState, config: RunnableConfig):
    """Generate a report plan with sections and research queries."""
    configurable = WorkflowConfiguration.from_runnable_config(config)
    topic = state.topic
    feedback = " ".join(state.feedback or [])
    
    # 1. Get the planner model
    planner_provider, planner_model_name, planner_model_id = configurable.get_model_for_role("planner")
    planner_llm = initialize_model(
        planner_provider,
        planner_model_name,
        configurable.planner_model_kwargs
    )

    # 2. Generate the report sections
    system_instructions_sections = report_planner_instructions.format(
        topic=topic,
        report_organization=configurable.report_structure,
        context="",  # Initial planning phase, no context needed
        feedback=feedback,
        today=get_today_str()
    )
    planner_message = "Generate the sections of the report. Your response must include a 'sections' field containing a list of sections."
    
    report_sections: Sections = await get_structured_output_with_fallback(
        planner_llm, 
        Sections, 
        [SystemMessage(content=system_instructions_sections), HumanMessage(content=planner_message)],
        model_id=planner_model_id
    )
    
    return {"sections": report_sections.sections}

async def human_feedback(state: DeepResearchState):
    """Wait for human feedback on the report plan."""
    # This is a placeholder for human-in-the-loop interaction
    # In a real app, this would pause and wait for UI input
    # For now, we automatically proceed to the research phase
    return {}

async def generate_queries(state: SectionResearchState, config: RunnableConfig):
    """Generate search queries for researching a specific section."""
    configurable = WorkflowConfiguration.from_runnable_config(config)
    # Guarded state access – fall back gracefully while logging
    topic = get_state_value(state, "topic", "Unknown Topic")
    section = get_state_value(state, "section", required=True)

    # 1. Get the writer model (or a dedicated query model if specified)
    writer_provider, writer_model_name, writer_model_id = configurable.get_model_for_role("writer")
    writer_model = initialize_model(
        writer_provider,
        writer_model_name,
        configurable.writer_model_kwargs
    )

    # 2. Generate queries
    system_instructions = query_writer_instructions.format(
        topic=topic, 
        section_topic=section.description, 
        number_of_queries=configurable.number_of_queries,
        today=get_today_str()
    )
    
    queries: Queries = await get_structured_output_with_fallback(
        writer_model, 
        Queries, 
        [SystemMessage(content=system_instructions), HumanMessage(content="Generate search queries.")],
        model_id=writer_model_id
    )

    return {
        "search_queries": [q.search_query for q in queries.queries],
        "section": section
    }

async def search_web(state: SectionResearchState, config: RunnableConfig):
    """Perform a web search for the generated queries."""
    configurable = WorkflowConfiguration.from_runnable_config(config)
    params_to_pass = get_search_api_params(configurable)
    search_api_val = get_config_value(configurable.search_api)
    search_api = str(search_api_val) if search_api_val is not None else "none"
    
    # Use dictionary key access for all state variables
    query_list = [q for q in state.get("search_queries", [])]
    
    source_str = await select_and_execute_search(
        search_api, 
        query_list, 
        params_to_pass
    )
    
    return {
        "source_str": source_str,
        "section": state["section"],
        "search_queries": state.get("search_queries", [])
    }

async def write_section(state: SectionResearchState, config: RunnableConfig):
    """Write a section of the report based on search results and reflect on it."""
    configurable = WorkflowConfiguration.from_runnable_config(config)
    # Guarded state access with sensible defaults
    topic = get_state_value(state, "topic", "Research Topic")
    section = get_state_value(state, "section")
    source_str = get_state_value(state, "source_str", "")
    search_iterations = get_state_value(state, "search_iterations", 0)

    # Ensure section exists
    if not section:
        # Should rarely happen but keep behaviour consistent
        logger.warning("No section found in state, creating default")
        section = Section(name="Main", description="Main content", research=True)

    # 1. Get the writer model
    writer_provider, writer_model_name, writer_model_id = configurable.get_model_for_role("writer")
    writer_model = initialize_model(
        writer_provider,
        writer_model_name,
        configurable.writer_model_kwargs or {}
    )

    # 2. Truncate source_str if needed for models with limited context
    if writer_model_id and "gpt-3.5-turbo" in writer_model_id.lower():
        # Limit source content for GPT-3.5-turbo to prevent context errors
        source_str = await summarize_search_results(source_str, max_tokens=6000, model=writer_model_id)

    # 3. Write the section content
    writer_system_message = section_writer_instructions.format(topic=topic, section_name=section.name)
    writer_human_message = section_writer_inputs.format(
        topic=topic,
        section_name=section.name,
        section_topic=section.description, 
        context=source_str, 
        section_content=section.content
    )
    
    section_content_result: SectionOutput = await get_structured_output_with_fallback(
        writer_model,
        SectionOutput,
        [SystemMessage(content=writer_system_message), HumanMessage(content=writer_human_message)],
        model_id=writer_model_id
    )
    section.content = filter_think_tokens(section_content_result.section_content)

    # 4. Get the reflection model
    reflection_provider, reflection_model_name, reflection_model_id = configurable.get_model_for_role("reflection")
    reflection_model = initialize_model(
        reflection_provider,
        reflection_model_name, 
        configurable.planner_model_kwargs or {} # Re-use planner kwargs for reflection
    )

    # 5. Reflect and grade the section
    section_grader_instructions_formatted = section_grader_instructions.format(
        topic=topic,
        section_topic=section.description,
        section=section.content, 
        number_of_follow_up_queries=configurable.number_of_queries
    )
    section_grader_message = "Grade the report section and suggest follow-up queries if needed."
    
    feedback: Feedback = await get_structured_output_with_fallback(
        reflection_model,
        Feedback,
        [SystemMessage(content=section_grader_instructions_formatted), HumanMessage(content=section_grader_message)],
        model_id=reflection_model_id
    )

    # 6. Decide whether to finish or iterate
    # If the section is passing or the max search depth is reached, publish the section to completed sections 
    if feedback.grade == "pass" or search_iterations >= configurable.max_search_depth:
        # Create a dictionary with only the fields that need to be accumulated.
        update = {"completed_sections": [section]}
        if configurable.include_source_str:
            update["source_str"] = source_str

        # Return JUST the update dictionary. The sub-graph will end automatically.
        return update

    # If more research is needed, return a Command to continue the loop.
    else:
        return Command(
            update={"search_queries": [q.search_query for q in feedback.follow_up_queries], "section": section},
            goto="search_web"
        )
    
async def write_final_sections(state: dict, config: RunnableConfig):
    """Write sections that don't require research using completed sections as context.
    
    This node handles sections like conclusions or summaries that build on
    the researched sections rather than requiring direct research.
    
    Args:
        state: Current state with completed sections as context
        config: Configuration for the writing model
        
    Returns:
        Dict containing the newly written section
    """
    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)

    # Use dictionary key access for all state variables
    topic = state["topic"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    
    # Format system instructions
    system_instructions = final_section_writer_instructions.format(topic=topic, section_name=section.name, section_topic=section.description, context=completed_report_sections)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = initialize_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    
    section_content = await writer_model.ainvoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="Generate a report section based on the provided sources.")])
    
    # Write content to section 
    section.content = section_content.content

    # Only return the fields that should be accumulated
    return {"completed_sections": [section]}

def gather_completed_sections(state: DeepResearchState):
    """Gather all completed sections and format them for the final report."""
    completed_sections = state.completed_sections
    
    # Format the sections into a string for context
    formatted_str = format_sections_for_context(completed_sections)
    
    return {"report_sections_from_research": formatted_str}

async def compile_final_report(state: DeepResearchState, config: RunnableConfig):
    """Compile the final report from all completed sections."""
    configurable = extract_configuration(config, WorkflowConfiguration)
    
    # Sort sections by their original planned order
    section_order = [s.name for s in state.sections]
    
    # Use a dictionary for quick lookup and update
    completed_sections_map = {s.name: s for s in state.completed_sections}

    # Ensure all sections are present, using original if completion is missing
    final_sections = [completed_sections_map.get(name, next((s for s in state.sections if s.name == name), None)) for name in section_order]
    
    # Filter out any None values in case a section is missing entirely
    final_sections = [s for s in final_sections if s is not None]
    
    # Format the sections into a single markdown string
    report_body = format_sections_for_final_report(final_sections)
    
    # Add sources if requested
    if configurable.include_source_str:
        # The source_str field accumulates all sources from parallel sub-graphs
        # due to the Annotated[str, operator.add] in the state definition
        all_source_str = state.source_str
        if all_source_str:
            # Use shared utility for URL extraction and formatting
            unique_urls = extract_unique_urls(all_source_str)
            sources_section = format_sources_section(unique_urls)
            if sources_section:
                report_body += sources_section
                logger.debug("Added sources section with %d URLs", len(unique_urls))
            else:
                logger.debug("Sources section was empty even though we have %d URLs", len(unique_urls))
        else:
            logger.debug("No source_str collected from any sections")
    else:
        logger.debug("Sources not included because include_source_str is False")
    
    return {"final_report": report_body}

def initiate_final_section_writing(state: DeepResearchState):
    """Create parallel tasks for writing non-research sections.
    
    This edge function identifies sections that don't need research and
    creates parallel writing tasks for each one.
    
    Args:
        state: Current state with all sections and research context
        
    Returns:
        List of Send commands for parallel section writing
    """
    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections", {
            "section": s, 
            "report_sections_from_research": state.report_sections_from_research
        })
        for s in state.sections 
        if not s.research
    ]

def initiate_section_research(state: DeepResearchState):
    """Create parallel tasks for researching sections that require research."""
    return [
        Send("build_section_with_web_research", {
            "section": s, 
            "search_iterations": 0, 
            "initial_sections": state.sections
        })
        for s in state.sections
        if s.research
    ]

# Report section sub-graph -- 

# Add nodes 
section_builder = StateGraph(SectionResearchState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

section_builder.add_edge("write_section", END)

# Outer graph for initial report plan compiling results from each section -- 

# Add nodes
builder = StateGraph(DeepResearchState, config_schema=WorkflowConfiguration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile()) # type: ignore
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_section_research)
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing)
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()
