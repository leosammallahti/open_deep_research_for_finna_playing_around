from typing import Literal, Union
import json
import asyncio
import re

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import PydanticOutputParser

from langgraph.graph import START, END, StateGraph
from langgraph.types import Send, interrupt, Command
from langgraph.store.memory import InMemoryStore

from open_deep_research.pydantic_state import (
    DeepResearchState,
    Sections,
    Queries,
    Feedback,
    SectionOutput
)

from open_deep_research.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions, 
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs
)

from open_deep_research.configuration import WorkflowConfiguration
from open_deep_research.utils import (
    deduplicate_and_format_sources, 
    get_config_value, 
    get_search_params, 
    select_and_execute_search,
    get_today_str,
    get_structured_output_with_fallback,
    filter_think_tokens,
    format_sections,
    format_sections_for_final_report,
    format_sections_for_context,
    summarize_search_results
)

## Nodes -- 

async def generate_report_plan(state: DeepResearchState, config: RunnableConfig):
    """Generate a report plan with sections and research queries."""
    configurable = WorkflowConfiguration.from_runnable_config(config)
    topic = state.topic
    feedback = " ".join(state.feedback or [])
    
    # 1. Get the planner model
    planner_provider, planner_model_name, planner_model_id = configurable.get_model_for_role("planner")
    planner_llm = init_chat_model(
        model=planner_model_name, 
        model_provider=planner_provider,
        model_kwargs=configurable.planner_model_kwargs or {}
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
    
    report_sections = await get_structured_output_with_fallback(
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

async def generate_queries(state: DeepResearchState, config: RunnableConfig):
    """Generate search queries for researching a specific section."""
    configurable = WorkflowConfiguration.from_runnable_config(config)
    topic = state.topic
    section = state.section

    # 1. Get the writer model (or a dedicated query model if specified)
    writer_provider, writer_model_name, writer_model_id = configurable.get_model_for_role("writer")
    writer_model = init_chat_model(
        model=writer_model_name, 
        model_provider=writer_provider, 
        model_kwargs=configurable.writer_model_kwargs or {}
    )

    # 2. Generate queries
    system_instructions = query_writer_instructions.format(
        topic=topic, 
        section_topic=section.description, 
        number_of_queries=configurable.number_of_queries,
        today=get_today_str()
    )
    
    queries = await get_structured_output_with_fallback(
        writer_model, 
        Queries, 
        [SystemMessage(content=system_instructions), HumanMessage(content="Generate search queries.")],
        model_id=writer_model_id
    )

    return {"search_queries": queries.queries}

async def search_web(state: DeepResearchState, config: RunnableConfig):
    """Perform a web search for the generated queries."""
    configurable = WorkflowConfiguration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}
    params_to_pass = get_search_params(search_api, search_api_config)
    
    query_list = [q.search_query for q in state.search_queries]
    
    source_str = await select_and_execute_search(
        search_api, 
        query_list, 
        params_to_pass
    )
    
    return {"source_str": source_str}

async def write_section(state: DeepResearchState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """Write a section of the report based on search results and reflect on it."""
    configurable = WorkflowConfiguration.from_runnable_config(config)
    topic = state.topic
    section = state.section
    source_str = state.source_str

    # 1. Get the writer model
    writer_provider, writer_model_name, writer_model_id = configurable.get_model_for_role("writer")
    writer_model = init_chat_model(
        model=writer_model_name,
        model_provider=writer_provider,
        model_kwargs=configurable.writer_model_kwargs or {}
    )

    # 2. Truncate source_str if needed for models with limited context
    if "gpt-3.5-turbo" in writer_model_id.lower():
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
    
    section_content_result = await get_structured_output_with_fallback(
        writer_model,
        SectionOutput,
        [SystemMessage(content=writer_system_message), HumanMessage(content=writer_human_message)],
        model_id=writer_model_id
    )
    section.content = filter_think_tokens(section_content_result.section_content)

    # 4. Get the reflection model
    reflection_provider, reflection_model_name, reflection_model_id = configurable.get_model_for_role("reflection")
    reflection_model = init_chat_model(
        model=reflection_model_name, 
        model_provider=reflection_provider, 
        model_kwargs=configurable.planner_model_kwargs or {} # Re-use planner kwargs for reflection
    )

    # 5. Reflect and grade the section
    section_grader_instructions_formatted = section_grader_instructions.format(
        topic=topic,
        section_topic=section.description,
        section=section.content, 
        number_of_follow_up_queries=configurable.number_of_queries
    )
    section_grader_message = "Grade the report section and suggest follow-up queries if needed."
    
    feedback = await get_structured_output_with_fallback(
        reflection_model,
        Feedback,
        [SystemMessage(content=section_grader_instructions_formatted), HumanMessage(content=section_grader_message)],
        model_id=reflection_model_id
    )

    # 6. Decide whether to finish or iterate
    if feedback.grade == "pass" or state.search_iterations >= configurable.max_search_depth:
        # Before finishing, find the original section to ensure all data is preserved
        original_section = next((s for s in state.initial_sections if s.name == section.name), None)
        if original_section:
            original_section.content = section.content
            update = {"completed_sections": [original_section]}
        else:
            # Fallback if the section can't be found (should not happen in normal flow)
            update = {"completed_sections": [section]}

        if configurable.include_source_str:
            update["sources"] = [source_str]
        return Command(update=update, goto=END)
    else:
        # Ensure the full section object is passed on, not just the name
        original_section = next((s for s in state.initial_sections if s.name == section.name), None)
        if original_section:
            section_to_pass = original_section
            section_to_pass.content = section.content # carry over the latest content
        else:
            section_to_pass = section # fallback

        return Command(
            update={
                "search_queries": feedback.follow_up_queries, 
                "section": section_to_pass,
                "initial_sections": state.initial_sections
            },
            goto="search_web"
        )
    
async def write_final_sections(state: DeepResearchState, config: RunnableConfig):
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

    # Get state 
    topic = state.topic
    section = state.section
    completed_report_sections = state.report_sections_from_research
    
    # Format system instructions
    system_instructions = final_section_writer_instructions.format(topic=topic, section_name=section.name, section_topic=section.description, context=completed_report_sections)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    
    section_content = await writer_model.ainvoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="Generate a report section based on the provided sources.")])
    
    # Write content to section 
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}

def gather_completed_sections(state: DeepResearchState):
    """Gather all completed sections and format them for the final report."""
    completed_sections = state.get("completed_sections", [])
    
    # Format the sections into a string for context
    formatted_str = format_sections_for_context(completed_sections)
    
    return {"report_sections_from_research": formatted_str}

async def compile_final_report(state: DeepResearchState, config: RunnableConfig):
    """Compile the final report from all completed sections."""
    configurable = WorkflowConfiguration.from_runnable_config(config)
    
    # Sort sections by their original planned order
    section_order = [s.name for s in state['sections']]
    
    # Use a dictionary for quick lookup and update
    completed_sections_map = {s.name: s for s in state['completed_sections']}

    # Ensure all sections are present, using original if completion is missing
    final_sections = [completed_sections_map.get(name, next((s for s in state['sections'] if s.name == name), None)) for name in section_order]
    
    # Filter out any None values in case a section is missing entirely
    final_sections = [s for s in final_sections if s is not None]
    
    # Format the sections into a single markdown string
    report_body = format_sections_for_final_report(final_sections)
    
    # Add sources if requested
    if configurable.include_source_str and state.get("sources"):
        # Extract unique URLs from all source strings
        all_urls = []
        url_pattern = r'URL:\s*(https?://[^\s\n]+)'
        
        for source_str in state["sources"]:
            urls = re.findall(url_pattern, source_str)
            all_urls.extend(urls)
        
        # Deduplicate while preserving order
        seen = set()
        unique_urls = []
        for url in all_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        # Format sources with numbers
        if unique_urls:
            sources_section = "\n\n---\n\n## Sources\n\n"
            for i, url in enumerate(unique_urls, 1):
                sources_section += f"[{i}] {url}\n"
            report_body += sources_section
        
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
        Send("write_final_sections", {"topic": state.topic, "section": s, "report_sections_from_research": state.report_sections_from_research}) 
        for s in state.sections 
        if not s.research
    ]

def initiate_section_research(state: DeepResearchState):
    """Create parallel tasks for researching sections that require research."""
    return [
        Send("build_section_with_web_research", {"topic": state.topic, "section": s, "search_iterations": 0, "initial_sections": state.sections})
        for s in state.sections
        if s.research
    ]

# Report section sub-graph -- 

# Add nodes 
section_builder = StateGraph(DeepResearchState, output=SectionOutput)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# Outer graph for initial report plan compiling results from each section -- 

# Add nodes
builder = StateGraph(DeepResearchState, input=DeepResearchState, output=DeepResearchState, config_schema=WorkflowConfiguration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_section_research, ["build_section_with_web_research"])
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()
