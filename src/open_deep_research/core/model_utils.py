"""Model utilities for common initialization patterns."""

from typing import Any, Dict, Type

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
try:
    from langsmith import Trace  # type: ignore
except ImportError:  # Fallback if older langsmith version
    class _NoOpTrace:  # noqa: D401 – Simple stub
        """Fallback Trace callback that does nothing if LangSmith is unavailable."""

        def __init__(self, *args, **kwargs):
            pass

        # The real Trace callback implements LangChain callbacks; we stub no-ops
        def __getattr__(self, _name):  # noqa: D401 – generic proxy
            def _noop(*_a, **_kw):
                return None

            return _noop

    Trace = _NoOpTrace  # type: ignore


def initialize_model(
    provider: str,
    model_name: str,
    model_kwargs: Dict[str, Any] | None = None,
    max_retries: int | None = None,
) -> BaseChatModel:
    """Initialize a chat model with common parameters.
    
    This function reduces duplication of model initialization code
    that appears throughout graph.py, workflow.py, and multi_agent.py.
    
    Args:
        provider: Model provider (e.g., "anthropic", "openai")
        model_name: Model name (e.g., "claude-3-5-sonnet")
        model_kwargs: Additional model-specific kwargs
        max_retries: Maximum retry attempts for the model
        
    Returns:
        Initialized chat model
    """
    kwargs = {
        "model": model_name,
        "model_provider": provider,
    }
    
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
        
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
        
    return init_chat_model(**kwargs)


def initialize_model_with_structured_output(
    provider: str,
    model_name: str,
    output_schema: Type[BaseModel],
    model_kwargs: Dict[str, Any] | None = None,
    max_retries: int | None = None,
) -> BaseChatModel:
    """Initialize a model and bind it to a structured output schema.
    
    Combines model initialization with structured output binding,
    a pattern that appears frequently in the codebase.
    
    Args:
        provider: Model provider
        model_name: Model name
        output_schema: Pydantic model for structured output
        model_kwargs: Additional model-specific kwargs
        max_retries: Maximum retry attempts
        
    Returns:
        Model configured for structured output
    """
    model = initialize_model(provider, model_name, model_kwargs, max_retries)
    return model.with_structured_output(output_schema)


def get_model_with_thinking_budget(
    provider: str,
    model_name: str,
    max_tokens: int = 20_000,
    thinking_budget: int = 16_000,
    model_kwargs: Dict[str, Any] | None = None,
    max_retries: int | None = None,
) -> BaseChatModel:
    """Initialize a model with thinking token budget (for supported models).
    
    This pattern appears in workflow.py for models that support
    thinking/reasoning tokens.
    
    Args:
        provider: Model provider
        model_name: Model name
        max_tokens: Maximum total tokens
        thinking_budget: Thinking token budget
        model_kwargs: Additional model-specific kwargs
        max_retries: Maximum retry attempts
        
    Returns:
        Model with thinking configuration if supported
    """
    # Models that support thinking tokens
    thinking_models = ["claude-3-7-sonnet-latest"]
    
    if model_name in thinking_models:
        kwargs = {
            "model": model_name,
            "model_provider": provider,
            "max_tokens": max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": thinking_budget}
        }
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        return init_chat_model(**kwargs)
    else:
        # Regular initialization for models without thinking support
        return initialize_model(provider, model_name, model_kwargs, max_retries) 


def trace_config(session: str = "dev") -> dict:
    """Return a LangSmith trace configuration dictionary.

    Usage:
        await graph.ainvoke(inputs, config=trace_config("quick-test"))
    """
    return {"callbacks": [Trace(session=session)]} 