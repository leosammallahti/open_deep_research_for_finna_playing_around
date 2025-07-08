"""Workflow implementation for Open Deep Research.

This module implements the core workflow logic for the research pipeline,
including node functions and workflow orchestration.
"""
import re
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from open_deep_research.configuration import WorkflowConfiguration
from open_deep_research.core import (
    extract_configuration,
    get_model_with_thinking_budget,
    get_search_api_params,
    initialize_model,
    initialize_model_with_structured_output,
)
from open_deep_research.pydantic_state import (
    ClarifyWithUser,
    DeepResearchState,
    SectionOutput,
)
from open_deep_research.state import (
    Feedback,
    Queries,
    Sections,
)
from open_deep_research.utils import (
    format_sections,
    get_config_value,
    get_today_str,
    select_and_execute_search,
)
from open_deep_research.workflow.prompts import (
    clarify_with_user_instructions,
    final_section_writer_instructions,
    query_writer_instructions,
    report_planner_instructions,
    report_planner_query_writer_instructions,
    section_grader_instructions,
    section_writer_inputs,
    section_writer_instructions,
)


## Nodes
def initial_router(state: DeepResearchState, config: RunnableConfig):
    """Route to the appropriate starting node based on configuration.
    
    Args:
        state: Current state of the deep research workflow
        config: Configuration settings for the workflow
        
    Returns:
        Name of the next node to execute
    """
    configurable = extract_configuration(config, WorkflowConfiguration)
    if configurable.clarify_with_user and not state.already_clarified_topic:
        return "clarify_with_user"
    else:
        return "generate_report_plan"


async def clarify_with_user(state: DeepResearchState, config: RunnableConfig):
    """Ask the user for clarification on the research topic.
    
    Args:
        state: Current state of the deep research workflow
        config: Configuration settings for the workflow
        
    Returns:
        Updated state with clarification question and flag set
    """
    messages = state.messages
    configurable = extract_configuration(config, WorkflowConfiguration)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    structured_llm = writer_model.with_structured_output(ClarifyWithUser)
    system_instructions = clarify_with_user_instructions.format(messages=get_buffer_string(messages))
    results = await structured_llm.ainvoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])
    return {"messages": messages + [AIMessage(content=results.question)], "already_clarified_topic": True}


async def generate_report_plan(state: DeepResearchState, config: RunnableConfig) -> Command[Literal["human_feedback","build_section_with_web_research"]]:
    """Generate a structured report plan with sections based on the topic and research.
    
    Args:
        state: Current state of the deep research workflow
        config: Configuration settings for the workflow
        
    Returns:
        Command directing to next step (human feedback or section building)
    """
    messages = state.messages
    feedback_list = state.feedback_on_report_plan
    feedback = " /// ".join(feedback_list) if feedback_list else ""

    configurable = extract_configuration(config, WorkflowConfiguration)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    params_to_pass = get_search_api_params(configurable)
    sections_user_approval = configurable.sections_user_approval

    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    structured_llm = initialize_model_with_structured_output(
        writer_provider, writer_model_name, Queries, writer_model_kwargs
    )

    system_instructions_query = report_planner_query_writer_instructions.format(
        messages=get_buffer_string(messages),
        report_organization=report_structure,
        number_of_queries=number_of_queries,
        today=get_today_str()
    )
    results = await structured_llm.ainvoke([SystemMessage(content=system_instructions_query),
                                     HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])
    
    query_list = [query.search_query for query in results.queries]
    search_api = get_config_value(configurable.search_api)
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)
    system_instructions_sections = report_planner_instructions.format(messages=get_buffer_string(messages), report_organization=report_structure, context=source_str, feedback=feedback)

    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    get_config_value(configurable.planner_model_kwargs or {})

    planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: name, description, research, and content fields."""
    
    structured_llm = get_model_with_thinking_budget(
        planner_provider, planner_model
    ).with_structured_output(Sections)
    report_sections = await structured_llm.ainvoke([SystemMessage(content=system_instructions_sections),
                                             HumanMessage(content=planner_message)])
    sections = report_sections.sections

    if sections_user_approval:
        return Command(goto="human_feedback", update={"sections": sections})
    else:
        return Command(goto=[
            Send("build_section_with_web_research", {"messages": messages, "section": s, "search_iterations": 0}) 
            for s in sections 
            if s.research
        ], update={"sections": sections})


async def human_feedback(state: DeepResearchState, config: RunnableConfig) -> Command[Literal["generate_report_plan","build_section_with_web_research"]]:
    """Collect human feedback on the report plan and route accordingly.
    
    Args:
        state: Current state of the deep research workflow
        config: Configuration settings for the workflow
        
    Returns:
        Command directing to report plan regeneration or section building
    """
    messages = state.messages
    sections = state.sections
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )
    interrupt_message = f"""Please provide feedback on the following report plan. 
                        \n\n{sections_str}\n
                        \nDoes the report plan meet your needs?\nPass 'true' to approve the report plan.\nOr, provide feedback to regenerate the report plan:"""
    feedback = interrupt(interrupt_message)
    if (isinstance(feedback, bool) and feedback is True) or (isinstance(feedback, str) and feedback.lower() == "true"):
        return Command(goto=[
            Send("build_section_with_web_research", {"messages": messages, "section": s, "search_iterations": 0}) 
            for s in sections 
            if s.research
        ])
    elif isinstance(feedback, str):
        return Command(goto="generate_report_plan", 
                       update={"feedback_on_report_plan": state.feedback_on_report_plan + [feedback]})
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")


async def generate_queries(state: DeepResearchState, config: RunnableConfig):
    """Generate search queries for a specific report section.
    
    Args:
        state: Current state of the deep research workflow
        config: Configuration settings for the workflow
        
    Returns:
        Updated state with search queries for the section
    """
    messages = state.messages
    section = state.section
    if not section:
        raise ValueError("Section not found in state.")
        
    configurable = extract_configuration(config, WorkflowConfiguration)
    number_of_queries = configurable.number_of_queries
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    structured_llm = initialize_model_with_structured_output(
        writer_provider, writer_model_name, Queries, writer_model_kwargs
    )
    system_instructions = query_writer_instructions.format(messages=get_buffer_string(messages), 
                                                           section_topic=section.description, 
                                                           number_of_queries=number_of_queries,
                                                           today=get_today_str())

    queries = await structured_llm.ainvoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content="Generate search queries on the provided topic.")])
    return {"search_queries": queries.queries}


async def search_web(state: DeepResearchState, config: RunnableConfig):
    """Execute web search using generated queries and return results.
    
    Args:
        state: Current state of the deep research workflow
        config: Configuration settings for the workflow
        
    Returns:
        Updated state with search results and incremented search iterations
    """
    search_queries = state.search_queries
    configurable = extract_configuration(config, WorkflowConfiguration)
    params_to_pass = get_search_api_params(configurable)

    query_list = [query.search_query for query in search_queries]
    search_api = get_config_value(configurable.search_api)
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    return {"source_str": source_str, "search_iterations": state.search_iterations + 1}


async def write_section(state: DeepResearchState, config: RunnableConfig):
    """Write a report section based on search results and grade the output.
    
    Args:
        state: Current state of the deep research workflow
        config: Configuration settings for the workflow
        
    Returns:
        Command directing to next step (more research or completion)
    """
    messages = state.messages
    section = state.section
    source_str = state.source_str
    configurable = WorkflowConfiguration.from_runnable_config(config)
    section_writer_inputs_formatted = section_writer_inputs.format(messages=get_buffer_string(messages), 
                                                             section_name=section.name, 
                                                             section_topic=section.description, 
                                                             context=source_str, 
                                                             section_content=section.content)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = initialize_model_with_structured_output(
        writer_provider, writer_model_name, SectionOutput, writer_model_kwargs,
        max_retries=configurable.max_structured_output_retries
    )

    section_content = await writer_model.ainvoke([SystemMessage(content=section_writer_instructions),
                                           HumanMessage(content=section_writer_inputs_formatted)])
    
    # Create a new section object with the updated content to maintain immutability
    updated_section = section.model_copy(update={'content': section_content.section_content})

    section_grader_message = ("Grade the report and consider follow-up questions for missing information. "
                              "If the grade is 'pass', return empty strings for all follow-up queries. "
                              "If the grade is 'fail', provide specific search queries to gather missing information.")
    
    section_grader_instructions_formatted = section_grader_instructions.format(messages=get_buffer_string(messages), 
                                                                               section_topic=updated_section.description,
                                                                               section=updated_section.content, 
                                                                               number_of_follow_up_queries=configurable.number_of_queries)

    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    get_config_value(configurable.planner_model_kwargs or {})

    reflection_model = get_model_with_thinking_budget(
        planner_provider, planner_model
    ).with_structured_output(Feedback)

    feedback = await reflection_model.ainvoke([SystemMessage(content=section_grader_instructions_formatted),
                                        HumanMessage(content=section_grader_message)])

    if feedback.grade == "pass" or state.search_iterations >= configurable.max_search_depth:
        update = {"completed_sections": state.completed_sections + [updated_section]}
        if configurable.include_source_str:
            update["source_str"] = state.source_str + source_str
        return Command(update=update, goto=END)
    else:
        return Command(
            update={"search_queries": feedback.follow_up_queries, "section": updated_section},
            goto="search_web"
        )


async def write_final_sections(state: DeepResearchState, config: RunnableConfig):
    """Write final sections of the report without additional research.
    
    Args:
        state: Current state of the deep research workflow
        config: Configuration settings for the workflow
        
    Returns:
        Updated state with completed sections
    """
    configurable = WorkflowConfiguration.from_runnable_config(config)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = initialize_model(writer_provider, writer_model_name, writer_model_kwargs) 

    messages = state.messages
    section = state.section
    completed_report_sections = state.report_sections_from_research
    system_instructions = final_section_writer_instructions.format(messages=get_buffer_string(messages), 
                                                                   section_name=section.name, 
                                                                   section_topic=section.description, 
                                                                   context=completed_report_sections)
    section_content = await writer_model.ainvoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="Generate a report section based on the provided sources.")])   
    updated_section = section.model_copy(update={'content': section_content.content})
    # Return the single section - LangGraph will automatically merge with existing completed_sections
    return {"completed_sections": [updated_section]}


async def gather_completed_sections(state: DeepResearchState):
    """Gather all completed sections and format them for the final report.
    
    Args:
        state: Current state of the deep research workflow
        
    Returns:
        Updated state with formatted report sections
    """
    completed_sections = state.completed_sections
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}


async def compile_final_report(state: DeepResearchState, config: RunnableConfig):
    """Compile the final report from all completed sections.
    
    Args:
        state: Current state of the deep research workflow
        config: Configuration settings for the workflow
        
    Returns:
        Updated state with the final compiled report
    """
    configurable = WorkflowConfiguration.from_runnable_config(config)
    
    completed_sections_map = {s.name: s.content for s in state.completed_sections}
    
    updated_sections = []
    for section in state.sections:
        # Pydantic models are immutable, so we create a new one with updated content
        updated_section = section.model_copy(update={'content': completed_sections_map.get(section.name, section.content)})
        updated_sections.append(updated_section)

    all_sections = "\n\n".join([s.content for s in updated_sections])

    if configurable.include_source_str and state.source_str:
        # Extract unique URLs from all source strings
        all_source_str = state.source_str
        if isinstance(all_source_str, str):
            # If it's a single string, convert to list
            source_strings = [all_source_str]
        else:
            # If it's already a list
            source_strings = all_source_str
            
        all_urls = []
        url_pattern = r'URL:\s*(https?://[^\s\n]+)'
        
        for source_str_item in source_strings:
            urls = re.findall(url_pattern, source_str_item)
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
            all_sections += sources_section
            
        return {
            "final_report": all_sections, 
            "source_str": state.source_str, 
            "messages": state.messages + [AIMessage(content=all_sections)],
            "sections": updated_sections
        }
    else:
        return {
            "final_report": all_sections, 
            "messages": state.messages + [AIMessage(content=all_sections)],
            "sections": updated_sections
        }


async def gather_all_sections(state: DeepResearchState):
    """Wait for all parallel write_final_sections to complete.
    
    This node serves as a proper fan-in point for the parallel final section writing,
    ensuring all sections are completed before proceeding to final report compilation.
    """
    # LangGraph automatically waits for all parallel nodes to complete before executing this
    return {}


async def initiate_final_section_writing(state: DeepResearchState):
    """Initiate the writing of final sections that don't need research.
    
    Args:
        state: Current state of the deep research workflow
        
    Returns:
        List of Send commands for final section writing
    """
    return [
        Send("write_final_sections", DeepResearchState(
            messages=state.messages, 
            section=s, 
            report_sections_from_research=state.report_sections_from_research,
            completed_sections=state.completed_sections,
            source_str=state.source_str
        ))
        for s in state.sections 
        if not s.research
    ]


## Graph
section_builder = StateGraph(DeepResearchState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

builder = StateGraph(DeepResearchState, config_schema=WorkflowConfiguration)
builder.add_node("clarify_with_user", clarify_with_user)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)
builder.add_node("gather_all_sections", gather_all_sections)
builder.add_conditional_edges(START, initial_router, ["clarify_with_user", "generate_report_plan"])
builder.add_edge("clarify_with_user", END)
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "gather_all_sections")
builder.add_edge("gather_all_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)
workflow = builder.compile()