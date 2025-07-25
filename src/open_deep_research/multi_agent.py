"""Multi-agent system for research report generation.

This module implements a multi-agent architecture for generating research reports
using LangGraph. It includes supervisor and research agents that work together
to create comprehensive reports by orchestrating search tools and writing content.
"""

import re
import warnings
from typing import Any, Dict, List, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from open_deep_research.configuration import MultiAgentConfiguration

# (The previous inline ReportState and SectionState definitions have been removed in favor of the
# Pydantic-based models that already exist in ``open_deep_research.pydantic_state``.)
# --------------------------------------------------------------------------- #
# Model configuration and helper utilities that were accidentally removed
# --------------------------------------------------------------------------- #
# Model configuration for handling different model capabilities
# DELETE the entire MODEL_CONFIGS dict and get_model_config function
from open_deep_research.core.model_utils import (
    bind_tools_with_capability as bind_tools_with_model_support,
)
from open_deep_research.prompt_loader import load_prompt

## State
from open_deep_research.pydantic_state import (
    MultiAgentReportState as ReportState,
)
from open_deep_research.pydantic_state import (
    MultiAgentSectionState as SectionState,
)

# New Tavily tool from dedicated module
from open_deep_research.tavily_tools import tavily_search_tool
from open_deep_research.utils import (
    duckduckgo_search,
    get_today_str,
    summarize_search_results,
    truncate_messages_for_context,
)

# --------------------------------------------------------------------------- #
# Search tool factory
# --------------------------------------------------------------------------- #


def get_search_tool(config: RunnableConfig) -> BaseTool | None:
    """Return the configured search tool (or ``None``)."""
    configurable = MultiAgentConfiguration.from_runnable_config(config)

    # Avoid a circular import – local import keeps startup fast as well
    from open_deep_research.core.config_utils import get_config_value

    search_api = str(get_config_value(configurable.search_api)).lower()

    if search_api == "none":
        return None

    if search_api == "tavily":
        search_tool = tavily_search_tool
    elif search_api == "duckduckgo":
        search_tool = duckduckgo_search
    else:
        raise NotImplementedError(
            f"The search API '{search_api}' is not yet supported in the multi-agent implementation. "
            "Currently, only Tavily/DuckDuckGo/None is supported. "
            "Please use the graph-based implementation in src/open_deep_research/graph.py "
            "for other search APIs, or set search_api to 'tavily', 'duckduckgo', or 'none'."
        )

    # Mark as a search tool so we can identify it later
    search_tool.metadata = {**(search_tool.metadata or {}), "type": "search"}
    return search_tool


# --------------------------------------------------------------------------- #
# Data models exposed as tools (unchanged from original code)
# --------------------------------------------------------------------------- #


class Section(BaseModel):
    """Section of the report."""

    name: str = Field(description="Name for this section of the report.")
    description: str = Field(
        description="Research scope for this section of the report."
    )
    content: str = Field(description="The content of the section.")


class Sections(BaseModel):
    """List of section titles of the report."""

    sections: List[str] = Field(description="Sections of the report.")


class Introduction(BaseModel):
    """Introduction to the report."""

    name: str = Field(description="Name for the report.")
    content: str = Field(
        description="The content of the introduction, giving an overview of the report."
    )


class Conclusion(BaseModel):
    """Conclusion to the report."""

    name: str = Field(description="Name for the conclusion of the report.")
    content: str = Field(
        description="The content of the conclusion, summarizing the report."
    )


class Question(BaseModel):
    """Ask a follow-up question to clarify the report scope."""

    question: str = Field(
        description="A specific question to ask the user to clarify the scope, focus, or requirements of the report."
    )


# No-op tool to indicate that the research is complete
class FinishResearch(BaseModel):
    """Finish the research."""


# No-op tool to indicate that the report writing is complete
class FinishReport(BaseModel):
    """Finish the report."""


@tool
def ask_question(query: str) -> str:
    """Ask user for clarification."""
    return input(f"Clarification: {query}")


async def _load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    if not configurable.mcp_server_config:
        return []

    mcp_server_config = configurable.mcp_server_config
    client = MultiServerMCPClient(mcp_server_config)
    mcp_tools = await client.get_tools()
    filtered_mcp_tools: list[BaseTool] = []
    for tool in mcp_tools:
        # TODO: this will likely be hard to manage
        # on a remote server that's not controlled by the developer
        # best solution here is allowing tool name prefixes in MultiServerMCPClient
        if tool.name in existing_tool_names:
            warnings.warn(
                f"Trying to add MCP tool with a name {tool.name} that is already in use - this tool will be ignored."
            )
            continue

        if (
            configurable.mcp_tools_to_include
            and tool.name not in configurable.mcp_tools_to_include
        ):
            continue

        filtered_mcp_tools.append(tool)

    return filtered_mcp_tools


# Tool lists will be built dynamically based on configuration
async def get_supervisor_tools(config: RunnableConfig) -> list[BaseTool]:
    """Get supervisor tools based on configuration"""
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    search_tool = get_search_tool(config)
    tools = [tool(Sections), tool(Introduction), tool(Conclusion), tool(FinishReport), tool(ask_question)]
    if configurable.ask_for_clarification:
        tools.append(ask_question)
    if search_tool is not None:
        tools.append(search_tool)  # Add search tool, if available
    existing_tool_names = {cast("BaseTool", tool).name for tool in tools}
    mcp_tools = await _load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    return tools


async def get_research_tools(config: RunnableConfig) -> list[BaseTool]:
    """Get research tools based on configuration"""
    search_tool = get_search_tool(config)
    tools = [tool(Section), tool(FinishResearch)]
    if search_tool is not None:
        tools.append(search_tool)  # Add search tool, if available
    existing_tool_names = {cast("BaseTool", tool).name for tool in tools}
    mcp_tools = await _load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    return tools


async def supervisor(state: ReportState, config: RunnableConfig):
    """LLM decides whether to call a tool or not."""
    from open_deep_research.core.logging_utils import bind_log_context

    bind_log_context(node="supervisor")
    # Get configuration
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    supervisor_model = configurable.supervisor_model

    # Initialize the model
    llm = init_chat_model(model=supervisor_model)

    # Get current messages (AnyMessage covers BaseMessage and others)
    raw_messages = list(state.messages)  # no strict typing – avoids invariance issues

    # Inlining a cast so truncate_messages_for_context receives the exact union it expects
    messages = await truncate_messages_for_context(
        cast("List[dict[str, Any] | BaseMessage]", raw_messages),
        max_context_tokens=60000,
        model_name=supervisor_model,
    )

    # If sections have been completed, but we don't yet have the final report, then we need to initiate writing the introduction and conclusion
    if state.completed_sections and not state.final_report:
        research_complete_message = {
            "role": "user",
            "content": "Research is complete. Now write the introduction and conclusion for the report. Here are the completed main body sections: \n\n"
            + "\n\n".join([s.content for s in state.completed_sections]),
        }
        messages = messages + [cast("BaseMessage", research_complete_message)]

    # Get tools based on configuration
    supervisor_tool_list = await get_supervisor_tools(config)

    llm_with_tools = bind_tools_with_model_support(
        llm, supervisor_tool_list, supervisor_model, parallel_tool_calls=False
    )

    # Get system prompt
    system_prompt = load_prompt("supervisor").replace("{today}", get_today_str())
    if configurable.mcp_prompt:
        system_prompt += f"\n\n{configurable.mcp_prompt}"

    # Invoke
    response = await llm_with_tools.ainvoke(
        [{"role": "system", "content": system_prompt}] + messages
    )
    return {"messages": [response]}


async def supervisor_tools(state: ReportState, config: RunnableConfig):
    """Perform the tool call (search/section writing/etc.) and decide next step."""
    from open_deep_research.core.logging_utils import bind_log_context

    bind_log_context(node="supervisor_tools")
    configurable = MultiAgentConfiguration.from_runnable_config(config)

    result: List[BaseMessage] = []
    sections_list = []
    intro_content = None
    conclusion_content = None
    source_str = ""

    # Get tools based on configuration
    supervisor_tool_list = await get_supervisor_tools(config)
    supervisor_tools_by_name = {tool.name: tool for tool in supervisor_tool_list}
    search_tool_names = {
        tool.name
        for tool in supervisor_tool_list
        if tool.metadata is not None and tool.metadata.get("type") == "search"
    }

    # First process all tool calls to ensure we respond to each one (required for OpenAI)
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        # No tool calls, so nothing to do
        return {"messages": []}

    for tool_call in last_message.tool_calls:
        # Get the tool
        tool = supervisor_tools_by_name[tool_call["name"]]
        # Perform the tool call - use ainvoke for async tools
        try:
            observation = await tool.ainvoke(tool_call["args"], config)
        except NotImplementedError:
            observation = tool.invoke(tool_call["args"], config)

        # **NEW: Truncate search results if they're too long**
        if tool_call["name"] in search_tool_names and isinstance(observation, str):
            observation = await summarize_search_results(observation, max_tokens=8000)

        # Append to messages
        result.append(
            ToolMessage(
                content=str(observation),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

        # Store special tool results for processing after all tools have been called
        if tool_call["name"] == "Question":
            # Question tool was called - return to supervisor to ask the question
            question_obj = cast("Question", observation)
            result.append(
                cast(
                    "BaseMessage",
                    {"role": "assistant", "content": question_obj.question},
                )
            )
            return {"messages": result}
        elif tool_call["name"] == "Sections":
            sections_list = cast("Sections", observation).sections
        elif tool_call["name"] == "Introduction":
            # Format introduction with proper H1 heading if not already formatted
            observation = cast("Introduction", observation)
            if not observation.content.startswith("# "):
                intro_content = f"# {observation.name}\n\n{observation.content}"
            else:
                intro_content = observation.content
        elif tool_call["name"] == "Conclusion":
            # Format conclusion with proper H2 heading if not already formatted
            observation = cast("Conclusion", observation)
            if not observation.content.startswith("## "):
                conclusion_content = f"## {observation.name}\n\n{observation.content}"
            else:
                conclusion_content = observation.content
        elif tool_call["name"] in search_tool_names and configurable.include_source_str:
            source_str += cast("str", observation)

    state_update: Dict[str, Any] = {}
    # After processing all tool calls, decide what to do next
    if sections_list:
        # Send the sections to the research agents
        send_commands = []
        for s in sections_list:
            send_commands.append(Send("research_team", {"section": s}))
        return send_commands
    elif intro_content:
        # Store introduction while waiting for conclusion
        # Append to messages to guide the LLM to write conclusion next
        result.append(
            AIMessage(content="Introduction written. Now write a conclusion section.")
        )
        state_update = {
            "final_report": intro_content,
            "messages": result,
        }
    elif conclusion_content:
        # Get all sections and combine in proper order: Introduction, Body Sections, Conclusion
        intro = state.final_report or ""
        body_sections = "\n\n".join([s.content for s in state.completed_sections])

        # Assemble final report in correct order
        complete_report = f"{intro}\n\n{body_sections}\n\n{conclusion_content}"

        # Add sources if configured
        if configurable.include_source_str and state.source_str:
            all_source_str = state.source_str
            if isinstance(all_source_str, list):
                all_source_str = "\n\n".join(all_source_str)

            url_pattern = r"URL:\s*(https?://[^\s\n]+)"
            urls = re.findall(url_pattern, all_source_str)

            # Deduplicate while preserving order
            seen = set()
            unique_urls = []
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)

            # Format sources with numbers
            if unique_urls:
                sources_section = "\n\n---\n\n## Sources\n\n"
                for i, url in enumerate(unique_urls, 1):
                    sources_section += f"[{i}] {url}\n"
                complete_report += sources_section

        # Append to messages to indicate completion
        result.append(
            AIMessage(
                content="Report is now complete with introduction, body sections, and conclusion."
            )
        )

        state_update = {
            "final_report": complete_report,
            "messages": result,
        }
    else:
        # Default case (for search tools, etc.)
        state_update["messages"] = result

    # Include source string for evaluation
    if configurable.include_source_str and source_str:
        state_update["source_str"] = source_str

    return state_update


def supervisor_should_continue(state: ReportState) -> str:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call."""
    messages = state.messages
    last_message = messages[-1]
    # End because the supervisor asked a question or is finished
    if (
        not isinstance(last_message, AIMessage)
        or not last_message.tool_calls
        or (
            len(last_message.tool_calls) == 1
            and last_message.tool_calls[0]["name"] == "FinishReport"
        )
    ):
        # Exit the graph
        return END

    # If the LLM makes a tool call, then perform an action
    return "supervisor_tools"


async def research_agent(state: SectionState, config: RunnableConfig):
    """LLM decides whether to call a tool or not."""
    from open_deep_research.core.logging_utils import bind_log_context

    bind_log_context(node="research_agent")
    # Get configuration
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    researcher_model = configurable.researcher_model

    # Initialize the model
    llm = init_chat_model(model=researcher_model)

    # Get tools based on configuration
    research_tool_list = await get_research_tools(config)
    system_prompt = (
        load_prompt("researcher").replace("{section_description}", state.section)
        .replace("{number_of_queries}", str(configurable.number_of_queries))
        .replace("{today}", get_today_str())
    )
    if configurable.mcp_prompt:
        system_prompt += f"\n\n{configurable.mcp_prompt}"

    # Ensure we have at least one user message (required by Anthropic)
    raw_messages = list(state.messages)
    if not raw_messages:
        raw_messages = [
            HumanMessage(
                content=f"Please research and write the section: {state.section}"
            )
        ]

    # **NEW: Truncate messages to prevent context length errors**
    max_context_tokens = 60000  # Leave room for tools and response
    messages = await truncate_messages_for_context(
        cast("List[dict[str, Any] | BaseMessage]", raw_messages),
        max_context_tokens=max_context_tokens,
        model_name=researcher_model,
    )

    response = await bind_tools_with_model_support(
        llm, research_tool_list, researcher_model, parallel_tool_calls=False
    ).ainvoke([{"role": "system", "content": system_prompt}] + messages)
    return {
        "messages": [
            # Enforce tool calling to either perform more search or call the Section tool to write the section
            response
        ]
    }


async def research_agent_tools(state: SectionState, config: RunnableConfig):
    """Perform the tool call and route to supervisor or continue the research loop."""
    from open_deep_research.core.logging_utils import bind_log_context

    bind_log_context(node="research_agent_tools")
    configurable = MultiAgentConfiguration.from_runnable_config(config)

    result: List[BaseMessage] = []
    completed_section = None
    source_str = ""

    # Get tools based on configuration
    research_tool_list = await get_research_tools(config)
    research_tools_by_name = {tool.name: tool for tool in research_tool_list}
    search_tool_names = {
        tool.name
        for tool in research_tool_list
        if tool.metadata is not None and tool.metadata.get("type") == "search"
    }

    # Process all tool calls first (required for OpenAI)
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        # No tool calls, so nothing to do
        # We should continue the research loop to let the LLM generate a response
        return {"messages": []}

    for tool_call in last_message.tool_calls:
        # Get the tool
        tool = research_tools_by_name[tool_call["name"]]
        # Perform the tool call - use ainvoke for async tools
        try:
            observation = await tool.ainvoke(tool_call["args"], config)
        except NotImplementedError:
            observation = tool.invoke(tool_call["args"], config)

        # Append to messages
        result.append(
            ToolMessage(
                content=str(observation),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

        # Store the section observation if a Section tool was called
        if tool_call["name"] == "Section":
            completed_section = cast("Section", observation)

        # Store the source string if a search tool was called
        if tool_call["name"] in search_tool_names and configurable.include_source_str:
            source_str += cast("str", observation)

    # After processing all tools, decide what to do next
    state_update: Dict[str, Any] = {"messages": result}
    if completed_section:
        # Write the completed section to state and return to the supervisor
        state_update["completed_sections"] = [completed_section]  # type: ignore[assignment]
    if configurable.include_source_str and source_str:
        state_update["source_str"] = source_str

    return state_update


def research_agent_should_continue(state: SectionState) -> str:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call."""
    messages = state.messages
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        # This case should not be reached, as the research agent should always call a tool
        # but as a fallback, we continue the loop
        return "research_agent"

    if last_message.tool_calls[0]["name"] == "FinishResearch":
        # Research is done - return to supervisor
        return END
    else:
        return "research_agent_tools"


"""Build the multi-agent workflow"""

# Research agent workflow
research_builder = StateGraph(SectionState)
research_builder.add_node("research_agent", research_agent)
research_builder.add_node("research_agent_tools", research_agent_tools)
research_builder.add_edge(START, "research_agent")
research_builder.add_conditional_edges(
    "research_agent",
    research_agent_should_continue,
    {"research_agent_tools": "research_agent_tools", END: END},
)
research_builder.add_edge("research_agent_tools", "research_agent")

# Supervisor workflow
supervisor_builder = StateGraph(ReportState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("research_team", research_builder.compile())  # type: ignore[arg-type]

# Flow of the supervisor agent
supervisor_builder.add_edge(START, "supervisor")
supervisor_builder.add_conditional_edges(
    "supervisor",
    supervisor_should_continue,
    {"supervisor_tools": "supervisor_tools", END: END},
)
supervisor_builder.add_edge("research_team", "supervisor")

graph = supervisor_builder.compile()
