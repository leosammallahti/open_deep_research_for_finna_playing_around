"""Formatting utilities for URLs, sections, and reports.

This module contains shared utilities used across different implementations
to reduce code duplication and improve maintainability.

## Phase 2 Refactoring Context

These utilities were created during Phase 2 of the generate_report_plan refactoring to:

1. **Standardize Configuration Patterns**: Both graph.py and workflow.py now use
   the same configuration extraction and search API parameter utilities.

2. **Extract Common Prompt Formatting**: The prompt formatting functions handle
   the key difference between implementations:
   - graph.py uses `topic` parameter
   - workflow.py uses `messages` parameter (converted to string)

3. **Unify Common Patterns**: URL extraction, source formatting, and section
   formatting are now shared across implementations.

## Why generate_report_plan Functions Remain Separate

The generate_report_plan functions in graph.py and workflow.py remain separate because:

1. **Fundamentally Different Approaches**:
   - graph.py: Simple section generation (no search, no commands)
   - workflow.py: Full pipeline with search, routing, and complex state management

2. **Different Return Types**:
   - graph.py: Returns simple dictionary with sections
   - workflow.py: Returns Command objects with routing logic

3. **Different State Management**:
   - graph.py: Uses DeepResearchState with structured state transitions
   - workflow.py: Uses message-based state with command routing

4. **Different Features**:
   - workflow.py: Has clarification and user approval features
   - graph.py: Has structured output with fallback handling

The refactoring achieved the right balance: **shared utilities where possible,
separate implementations where necessary** for maintainability and clarity.
"""

import re
from typing import Any, List, Union

from langchain_core.messages import get_buffer_string


def extract_unique_urls(source_strings: Union[str, List[str]]) -> List[str]:
    """Extract unique URLs from source strings.
    
    This pattern appears in compile_final_report functions in both
    graph.py and workflow.py for formatting source citations.
    
    Args:
        source_strings: Single string or list of strings containing URLs
        
    Returns:
        List of unique URLs in order of first appearance
    """
    # Normalize to list
    if isinstance(source_strings, str):
        source_strings = [source_strings]
    
    all_urls = []
    url_pattern = r'URL:\s*(https?://[^\s\n]+)'
    
    for source_str in source_strings:
        urls = re.findall(url_pattern, source_str)
        all_urls.extend(urls)
    
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for url in all_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls


def format_sources_section(unique_urls: List[str]) -> str:
    """Format unique URLs into a sources section.
    
    This pattern appears in compile_final_report functions.
    
    Args:
        unique_urls: List of unique URLs
        
    Returns:
        Formatted sources section string
    """
    if not unique_urls:
        return ""
    
    sources_section = "\n\n---\n\n## Sources\n\n"
    for i, url in enumerate(unique_urls, 1):
        sources_section += f"[{i}] {url}\n"
    
    return sources_section


def format_sections_to_string(sections: List[Any]) -> str:
    """Format sections into a string representation.
    
    This pattern appears in multiple places for formatting sections.
    
    Args:
        sections: List of section objects
        
    Returns:
        Formatted string representation
    """
    return "\n\n".join([s.content for s in sections])


def format_prompt_with_context(
    prompt_template: str,
    topic: str | None = None,
    messages: List[Any] | None = None,
    **kwargs
) -> str:
    """Format prompt templates with standardized context patterns.
    
    This utility handles the common pattern where prompts need either
    a topic string or message history context.
    
    Args:
        prompt_template: The prompt template string
        topic: Topic string (for graph.py style)
        messages: Message history (for workflow.py style)
        **kwargs: Additional format parameters
        
    Returns:
        Formatted prompt string
    """
    format_args = kwargs.copy()
    
    # Handle the topic vs messages pattern
    if topic is not None:
        format_args['topic'] = topic
    elif messages is not None:
        format_args['messages'] = get_buffer_string(messages)
    
    return prompt_template.format(**format_args)


def format_section_writer_inputs(
    section_writer_inputs_template: str,
    section: Any,
    source_str: str,
    topic: str | None = None,
    messages: List[Any] | None = None
) -> str:
    """Format section writer inputs with standardized patterns.
    
    This handles the common pattern of formatting section writer inputs
    with either topic or message context.
    
    Args:
        section_writer_inputs_template: Template string
        section: Section object
        source_str: Source string for context
        topic: Topic string (for graph.py style)
        messages: Message history (for workflow.py style)
        
    Returns:
        Formatted section writer inputs
    """
    format_args = {
        'section_name': section.name,
        'section_topic': section.description,
        'section_content': section.content,
        'context': source_str
    }
    
    # Handle the topic vs messages pattern
    if topic is not None:
        format_args['topic'] = topic
    elif messages is not None:
        format_args['messages'] = get_buffer_string(messages)
    
    return section_writer_inputs_template.format(**format_args)


def format_grader_instructions(
    grader_instructions_template: str,
    section: Any,
    number_of_queries: int,
    topic: str | None = None,
    messages: List[Any] | None = None
) -> str:
    """Format section grader instructions with standardized patterns.
    
    Args:
        grader_instructions_template: Template string
        section: Section object
        number_of_queries: Number of follow-up queries
        topic: Topic string (for graph.py style)
        messages: Message history (for workflow.py style)
        
    Returns:
        Formatted grader instructions
    """
    format_args = {
        'section_topic': section.description,
        'section': section.content,
        'number_of_follow_up_queries': number_of_queries
    }
    
    # Handle the topic vs messages pattern
    if topic is not None:
        format_args['topic'] = topic
    elif messages is not None:
        format_args['messages'] = get_buffer_string(messages)
    
    return grader_instructions_template.format(**format_args) 