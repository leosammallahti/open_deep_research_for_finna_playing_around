"""Model utilities for common initialization patterns."""

from typing import Any, Dict, Type

# ---------------------------------------------------------------------------
# Optional: credit / token usage logging via callbacks
# ---------------------------------------------------------------------------
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import init_chat_model
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from open_deep_research.credit_tracker import record_model


class _TokenLogger(BaseCallbackHandler):
    """Callback that logs token usage (best effort) to ``credit_tracker``.

    The callback first looks for provider-supplied usage metadata (``llm_output``)
    which is present for OpenAI-compatible backends like TogetherAI.  If that
    fails, it falls back to a crude approximation based on character counts
    (≈ 4 chars ≈ 1 token).
    """

    def __init__(self, provider: str, model_name: str) -> None:  # noqa: D401 – simple init
        super().__init__()
        self._provider = provider
        self._model_name = model_name

    # NOTE: Accept positional ``response`` arg to align with LangChain >=0.1 API.
    # The callback was previously keyword-only which led to a TypeError when
    # LangChain passed the LLMResult as a positional argument.  We now accept
    # it positionally while still referring to it internally as ``outputs`` to
    # avoid changing the original logic.
    def on_llm_end(self, response, **_kwargs):  # type: ignore[override]
        outputs = response  # alias used in the original implementation

        # Try to extract token counts from llm_output (preferred).
        input_tokens = 0
        output_tokens = 0

        try:
            usage_meta = getattr(outputs, "llm_output", None)
            if isinstance(usage_meta, dict):
                # TogetherAI (and most OpenAI-compatible endpoints) return
                # either a 'usage' or 'token_usage' dict.
                token_usage = (
                    usage_meta.get("usage") or usage_meta.get("token_usage") or {}
                )
                if isinstance(token_usage, dict):
                    input_tokens = int(
                        token_usage.get(
                            "prompt_tokens", token_usage.get("input_tokens", 0)
                        )
                    )
                    completion_tokens = int(
                        token_usage.get(
                            "completion_tokens", token_usage.get("output_tokens", 0)
                        )
                    )
                    total = int(
                        token_usage.get(
                            "total_tokens", input_tokens + completion_tokens
                        )
                    )
                    # Some providers report only total; try to infer split.
                    if completion_tokens == 0 and total and input_tokens:
                        completion_tokens = max(total - input_tokens, 0)
                    output_tokens = completion_tokens or max(total - input_tokens, 0)
        except Exception:  # pragma: no cover – best effort fallback
            usage_meta = None  # silence linter

        # Fallback: approximate from generated text length.
        if input_tokens == 0 and output_tokens == 0:
            try:
                gen_texts = [
                    g.text for gen_list in outputs.generations for g in gen_list
                ]
                output_tokens = sum(len(t) for t in gen_texts) // 4  # rough heuristic
            except Exception:  # pragma: no cover – still nothing, give up
                output_tokens = 0

        record_model(
            self._provider,
            self._model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


try:
    from langsmith import Trace
except ImportError:  # Fallback if older langsmith version

    class _NoOpTrace:  # noqa: D401 – Simple stub
        """Fallback Trace callback that does nothing if LangSmith is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        # The real Trace callback implements LangChain callbacks; we stub no-ops
        def __getattr__(self, _name: str) -> Any:  # noqa: D401 – generic proxy
            def _noop(*_a: Any, **_kw: Any) -> None:
                return None

            return _noop

    Trace = _NoOpTrace


if "__all__" not in globals():
    __all__: list[str] = []


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
    # Strip accidental whitespace for common provider API keys to avoid
    # HTTP header errors like ``Illegal header value b'Bearer … '`` which occurs
    # when the key contains trailing spaces/newlines (often introduced when
    # copying from a terminal or secrets manager).
    import os as _os  # local import to avoid polluting module namespace

    _keys_to_normalise = [
        "TOGETHER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY",
    ]

    for _k in _keys_to_normalise:
        _val = _os.getenv(_k)
        if _val is not None and _val.strip() != _val:
            _os.environ[_k] = _val.strip()

    # LangChain expects a mapping for model_kwargs. Passing None can lead to
    #   internal TypeError when the library iterates over the mapping.
    if model_kwargs is None:
        model_kwargs = {}

    # Set default max_retries if None to avoid validation errors
    if max_retries is None:
        max_retries = 3

    return init_chat_model(
        model=model_name,
        model_provider=provider,
        model_kwargs=model_kwargs,
        max_retries=max_retries,
        callbacks=[_TokenLogger(provider, model_name)],
    )


def initialize_model_with_structured_output(
    provider: str,
    model_name: str,
    output_schema: Type[BaseModel],
    model_kwargs: Dict[str, Any] | None = None,
    max_retries: int | None = None,
) -> Runnable:
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
        # Build kwargs for thinking model
        thinking_kwargs = {"type": "enabled", "budget_tokens": thinking_budget}

        # Initialize with thinking support
        return init_chat_model(
            model=model_name,
            model_provider=provider,
            max_tokens=max_tokens,
            thinking=thinking_kwargs,
            model_kwargs=model_kwargs,
            max_retries=max_retries,
        )
    else:
        # Regular initialization for models without thinking support
        return initialize_model(provider, model_name, model_kwargs, max_retries)


def trace_config(session: str = "dev") -> Dict[str, Any]:
    """Return a LangSmith trace configuration dictionary.

    Usage:
        await graph.ainvoke(inputs, config=trace_config("quick-test"))
    """
    import os

    # Ensure LangSmith environment is properly configured
    if not os.getenv("LANGCHAIN_TRACING_V2"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    # Set default project if not set
    if not os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = f"open-deep-research-{session}"

    # Ensure endpoint is set
    if not os.getenv("LANGCHAIN_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    # Log tracing configuration for debugging
    from open_deep_research.core.logging_utils import get_logger

    _logger = get_logger(__name__)
    _logger.debug(
        "LangSmith tracing configured - Project: %s, Session: %s",
        os.getenv("LANGSMITH_PROJECT"),
        session,
    )

    # Return both callbacks and tags for better tracing
    return {
        "callbacks": [Trace(session=session)],
        "tags": [f"session:{session}", "open-deep-research"],
        "metadata": {
            "session": session,
            "project": os.getenv("LANGSMITH_PROJECT"),
        },
    }


def bind_tools_with_capability(
    llm: Any,
    tools: list[BaseTool],
    model_id: str,
    *,
    parallel_tool_calls: bool = False,
    tool_choice: str | None = "any",
):
    """Bind *tools* to *llm* respecting the model's tool-calling capability.

    Most OpenAI-compatible models support function-calling via ``bind_tools`` but
    a few (e.g. DeepSeek-Reasoner) do **not**.  This helper centralises the
    try/except logic so callers don’t repeat the same pattern.
    """

    from open_deep_research.model_registry import supports_tool_choice

    if supports_tool_choice(model_id):
        return llm.bind_tools(
            tools,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )
    # Fallback – older models silently ignore *tool_choice* argument
    return llm.bind_tools(tools, parallel_tool_calls=parallel_tool_calls)
