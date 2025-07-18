from __future__ import annotations

"""Common helpers for search provider modules.

These utilities were migrated out of the monolithic *utils.py* so that
provider-specific modules can depend on them without reaching back into that
large file.  Over time, additional shared helpers can be placed here.
"""

from typing import Any, Dict, Literal

__all__ = [
    "safe_get",
    "deduplicate_and_format_sources",
]

# ---------------------------------------------------------------------------
# Generic safe_get helper (copied verbatim from utils.py)
# ---------------------------------------------------------------------------

def safe_get(obj: Any, key: str, default: Any | None = None) -> Any:
    """Return *obj[key]* regardless of whether *obj* is a mapping or an object.

    Falls back to *default* when the attribute/key is missing.  This mirrors the
    behaviour of the original implementation in *utils.py*.
    """

    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key, default)
    return default


# ---------------------------------------------------------------------------
# Basic de-duplication + formatting helper
# ---------------------------------------------------------------------------

def deduplicate_and_format_sources(
    search_response: Any,
    *,
    max_tokens_per_source: int = 5000,
    include_raw_content: bool = True,
    deduplication_strategy: Literal["keep_first", "keep_last"] = "keep_first",
) -> str:
    """Format a list of search responses into a readable markdown string.

    This logic is identical to the historical helper from *utils.py*.  It is
    kept here so we can eventually remove that dependency entirely.
    """

    formatted_output = ""
    unique_results: dict[str, Dict[str, Any]] = {}

    if not search_response:
        return "No search results found."

    sources_list = search_response if isinstance(search_response, list) else [search_response]

    for response in sources_list:
        if not isinstance(response, dict):
            continue
        results = safe_get(response, "results", [])
        if not isinstance(results, list):
            continue

        for result in results:
            if not isinstance(result, dict):
                continue
            url = safe_get(result, "url")
            if not url:
                continue

            if deduplication_strategy == "keep_first" and url in unique_results:
                # If we're keeping the first occurrence, skip duplicates.
                continue

            unique_results[url] = result  # keep last by default or overwrite

    for url, result in unique_results.items():
        formatted_output += f"\n\nSource: {safe_get(result, 'title', 'Untitled')}\n"
        formatted_output += f"URL: {url}\n===\n"
        formatted_output += (
            f"Most relevant content from source: {safe_get(result, 'content', '')}\n===\n"
        )

        if include_raw_content:
            raw_content = safe_get(result, "raw_content", "")
            if raw_content:
                formatted_output += f"Full content:\n{raw_content[:max_tokens_per_source]}\n"

    return formatted_output or "No valid search results found." 