from __future__ import annotations

"""Exa search provider implementation (extracted from utils).
"""

import asyncio
import os
from typing import Any, Dict, List

from exa_py import Exa
from langsmith import traceable

from open_deep_research.core.network_utils import async_retry

from .base import safe_get

__all__ = ["exa_search"]


@traceable
@async_retry()
async def exa_search(
    search_queries: List[str],
    max_characters: int | None = None,
    num_results: int = 5,
    include_domains: List[str] | None = None,
    exclude_domains: List[str] | None = None,
    subpages: int | None = None,
) -> List[Dict[str, Any]]:
    """Search the web using the Exa API.

    Parameters mirror the legacy implementation; the docstring is shortened for brevity.
    """

    if include_domains and exclude_domains:
        raise ValueError("Cannot specify both include_domains and exclude_domains")

    exa = Exa(api_key=f"{os.getenv('EXA_API_KEY')}")

    async def process_query(query: str) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()

        def exa_search_fn():
            kwargs: Dict[str, Any] = {
                "text": True if max_characters is None else {"max_characters": max_characters},
                "summary": True,
                "num_results": num_results,
            }
            if subpages is not None:
                kwargs["subpages"] = subpages
            if include_domains:
                kwargs["include_domains"] = include_domains
            elif exclude_domains:
                kwargs["exclude_domains"] = exclude_domains
            return exa.search_and_contents(query, **kwargs)

        response = await loop.run_in_executor(None, exa_search_fn)
        formatted_results: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()

        results_list = safe_get(response, "results", [])

        for result in results_list:
            title = safe_get(result, "title", "")
            url = safe_get(result, "url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            content = ""
            text = safe_get(result, "text", "")
            summary = safe_get(result, "summary", "")
            if summary:
                content = f"{summary}\n\n{text}" if text else summary
            else:
                content = text
            formatted_results.append(
                {
                    "title": title,
                    "url": url,
                    "content": content,
                    "score": safe_get(result, "score", 0.0),
                    "raw_content": text,
                }
            )

        # Handle subpages when requested
        if subpages is not None:
            for result in results_list:
                for sub in safe_get(result, "subpages", []):
                    sub_url = safe_get(sub, "url", "")
                    if sub_url in seen_urls:
                        continue
                    seen_urls.add(sub_url)
                    sub_text = safe_get(sub, "text", "")
                    sub_summary = safe_get(sub, "summary", "")
                    sub_content = (
                        f"{sub_summary}\n\n{sub_text}" if sub_summary else sub_text
                    )
                    formatted_results.append(
                        {
                            "title": safe_get(sub, "title", ""),
                            "url": sub_url,
                            "content": sub_content,
                            "score": safe_get(sub, "score", 0.0),
                            "raw_content": sub_text,
                        }
                    )

        return {"query": query, "results": formatted_results}

    tasks = [process_query(q) for q in search_queries]
    return await asyncio.gather(*tasks) 