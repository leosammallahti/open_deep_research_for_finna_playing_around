from __future__ import annotations

"""Map-driven search dispatcher.

This module exposes two public helpers – ``validate_search_params`` and
``select_and_execute_search`` – plus the registries ``SEARCH_PARAMS`` and
``SEARCH_IMPL``.  The design mirrors the upstream LangChain "open_deep_research"
implementation but is adapted to our fork:
    • Handlers are async callables returning *str* (formatted result).
    • The public surface is **pure** and contains **no LangGraph / tool**
      dependencies so it can be unit-tested in isolation (see
      ``tests/test_search_dispatcher.py``).

Existing provider-specific implementations already live in
``open_deep_research.utils``.  We reuse them instead of duplicating logic.

NOTE: Only a representative subset of providers is wired initially.  Adding a
new provider is a <10-line change:
    1.  Create / import an async handler ``async def my_search(queries, **p): ...``
    2.  Append allowed param names to ``SEARCH_PARAMS``
    3.  Add an entry to ``SEARCH_IMPL`` with ``SearchProviderConfig``.

This keeps discoverability high and reduces branching logic elsewhere.
"""

from typing import Any, Awaitable, Callable, Dict, List

from pydantic import BaseModel, Field
from open_deep_research.utils import TokenBudgetManager

# ---------------------------------------------------------------------------
# Provider config dataclass
# ---------------------------------------------------------------------------


class SearchProviderConfig(BaseModel):
    """Pydantic model describing a search provider."""

    handler: Callable[..., Awaitable[str]]
    accepted_params: List[str] = Field(default_factory=list)
    display_name: str
    description: str

    class Config:  # noqa: D106 – pydantic config namespace
        arbitrary_types_allowed = True
        frozen = True


# ---------------------------------------------------------------------------
# Parameter whitelist per provider
# ---------------------------------------------------------------------------


SEARCH_PARAMS: Dict[str, List[str]] = {
    "tavily": [
        "max_results",
        "search_depth",
        "include_domains",
        "exclude_domains",
    ],
    "duckduckgo": ["max_results"],
    "exa": [
        "num_results",
        "include_domains",
        "exclude_domains",
        "max_characters",
        "subpages",
    ],
    "perplexity": [],  # no additional params currently
    "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
    "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],
    "linkup": ["depth"],
    "googlesearch": ["max_results", "include_raw_content"],
    "azureaisearch": ["max_results", "topic", "include_raw_content"],
}

# ---------------------------------------------------------------------------
# Import concrete async handlers from utils
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper to wrap list-returning search fns into formatted str
# ---------------------------------------------------------------------------
# Re-exported thin wrappers live in open_deep_research.search
from open_deep_research.search import (  # noqa: E402
    arxiv_search_async,
    azureaisearch_search_async,
    deduplicate_and_format_sources,  # noqa: E402
    duckduckgo_search,
    exa_search,
    google_search_async,
    linkup_search,
    perplexity_search_async,
    pubmed_search_async,
)
from open_deep_research.utils import (
    # Re-use legacy monolithic dispatcher for Tavily to avoid duplication.
    select_and_execute_search as _legacy_select_and_execute_search,
)


def _wrap_list_provider(fn: Callable[..., Awaitable[List[Dict[str, Any]]]]) -> Callable[..., Awaitable[str]]:
    """Wrap a provider returning list[dict] so dispatcher still returns *str*."""

    async def _inner(queries: List[str], **params: Any) -> str:  # noqa: D401 – simple wrapper
        raw_results = await fn(queries, **params)  # type: ignore[arg-type]
        return deduplicate_and_format_sources(
            raw_results,
            max_tokens_per_source=4000,
            deduplication_strategy="keep_first",
        )

    return _inner


# ---------------------------------------------------------------------------
# Registry – will be filled just below.
# ---------------------------------------------------------------------------


SEARCH_IMPL: Dict[str, SearchProviderConfig] = {}


def _register(name: str, cfg: SearchProviderConfig) -> None:  # helper
    """Internal helper to populate registry with validation."""

    if name in SEARCH_IMPL:
        raise ValueError(f"Provider '{name}' already registered.")
    SEARCH_IMPL[name] = cfg


# ---- concrete registrations ------------------------------------------------

_register(
    "tavily",
    SearchProviderConfig(
        handler=lambda queries, **p: _legacy_select_and_execute_search(
            "tavily", queries, p
        ),
        accepted_params=SEARCH_PARAMS["tavily"],
        display_name="Tavily",
        description="Web search via Tavily API.",
    ),
)

_register(
    "duckduckgo",
    SearchProviderConfig(
        handler=lambda queries, **p: duckduckgo_search.ainvoke(
            {"search_queries": queries}
        ),
        accepted_params=SEARCH_PARAMS["duckduckgo"],
        display_name="DuckDuckGo",
        description="DuckDuckGo web search (HTML scraping).",
    ),
)

_register(
    "exa",
    SearchProviderConfig(
        handler=_wrap_list_provider(exa_search),
        accepted_params=SEARCH_PARAMS["exa"],
        display_name="Exa",
        description="Exa semantic web search.",
    ),
)

_register(
    "perplexity",
    SearchProviderConfig(
        handler=_wrap_list_provider(perplexity_search_async),
        accepted_params=SEARCH_PARAMS["perplexity"],
        display_name="Perplexity Web",
        description="Perplexity API web search.",
    ),
)

_register(
    "arxiv",
    SearchProviderConfig(
        handler=_wrap_list_provider(arxiv_search_async),
        accepted_params=SEARCH_PARAMS["arxiv"],
        display_name="ArXiv",
        description="ArXiv scholarly paper search.",
    ),
)

_register(
    "pubmed",
    SearchProviderConfig(
        handler=_wrap_list_provider(pubmed_search_async),
        accepted_params=SEARCH_PARAMS["pubmed"],
        display_name="PubMed",
        description="PubMed biomedical literature.",
    ),
)

_register(
    "linkup",
    SearchProviderConfig(
        handler=_wrap_list_provider(linkup_search),
        accepted_params=SEARCH_PARAMS["linkup"],
        display_name="LinkUp",
        description="Linkup graph search.",
    ),
)

_register(
    "googlesearch",
    SearchProviderConfig(
        handler=_wrap_list_provider(google_search_async),
        accepted_params=SEARCH_PARAMS["googlesearch"],
        display_name="Google Web Search",
        description="Unofficial Google Search wrapper.",
    ),
)

_register(
    "azureaisearch",
    SearchProviderConfig(
        handler=_wrap_list_provider(azureaisearch_search_async),
        accepted_params=SEARCH_PARAMS["azureaisearch"],
        display_name="Azure AI Search",
        description="Azure Cognitive Search index query.",
    ),
)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def validate_search_params(api: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter *params* keeping only whitelisted keys for *api*.

    Raises ``ValueError`` if *api* is unknown.
    """

    if api not in SEARCH_PARAMS:
        raise ValueError(f"Unsupported search API: {api}")

    allowed = set(SEARCH_PARAMS[api])
    return {k: v for k, v in params.items() if k in allowed and v is not None}


async def select_and_execute_search(
    query: str,
    search_api: str,
    max_results: int = 5,
    budget_mgr: TokenBudgetManager | None = None,
    **kwargs
) -> str:
    """Execute search with optional token budget enforcement."""
    
    # If budget is exhausted, return early
    if budget_mgr and budget_mgr.exhausted():
        return "Budget exhausted - no search performed."
    
    # Perform the search (existing logic)
    results = await dispatch_search(search_api, [query], **kwargs)
    
    # If no budget manager, return results as-is (existing behavior)
    if not budget_mgr:
        return deduplicate_and_format_sources(results)
    
    # Estimate token cost and enforce budget
    filtered_results = []
    for result in results:
        content = str(result.get('content', ''))
        estimated_tokens = len(content) // 4  # rough heuristic: 4 chars per token
        
        allocated = budget_mgr.allocate(estimated_tokens)
        if allocated > 0:
            # Truncate content if we only got partial allocation
            if allocated < estimated_tokens:
                truncate_at = allocated * 4
                content = content[:truncate_at] + "... [truncated due to budget]"
            filtered_results.append({**result, 'content': content})
        else:
            # Budget exhausted, stop processing more results
            break
    
    return deduplicate_and_format_sources(filtered_results)


async def dispatch_search(search_api: str, queries: List[str], **kwargs) -> str:
    """Dispatch search to appropriate handler and return formatted results."""
    handler = SEARCH_IMPL.get(search_api)
    if not handler:
        raise ValueError(f"Unsupported search API: {search_api}")
    
    # Execute handler and format results
    raw_results = await handler.handler(queries, **kwargs)
    return deduplicate_and_format_sources(raw_results)


__all__ = [
    "SEARCH_PARAMS",
    "SEARCH_IMPL",
    "SearchProviderConfig",
    "validate_search_params",
    "select_and_execute_search",
] 