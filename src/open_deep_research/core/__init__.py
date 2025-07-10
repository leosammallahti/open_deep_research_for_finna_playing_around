"""Core utilities for Open Deep Research.

This module contains shared utilities used across different implementations
to reduce code duplication and improve maintainability.
"""

from .config_utils import (
    extract_configuration,
    get_search_api_params,
)
from .format_utils import (
    extract_unique_urls,
    format_grader_instructions,
    format_prompt_with_context,
    format_section_writer_inputs,
    format_sections_to_string,
    format_sources_section,
)
from .model_utils import (
    get_model_with_thinking_budget,
    initialize_model,
    initialize_model_with_structured_output,
)

__all__ = [
    # Model utilities
    "initialize_model",
    "initialize_model_with_structured_output",
    "get_model_with_thinking_budget",
    # Config utilities
    "extract_configuration",
    "get_search_api_params",
    # Format utilities
    "extract_unique_urls",
    "format_sources_section",
    "format_sections_to_string",
    "format_prompt_with_context",
    "format_section_writer_inputs",
    "format_grader_instructions",
]
