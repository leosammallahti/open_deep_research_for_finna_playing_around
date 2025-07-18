from __future__ import annotations

"""Perplexity API search provider (extracted from utils).
"""

import asyncio
import os
from typing import Any, Dict, List

import httpx
from langsmith import traceable

from open_deep_research.core.network_utils import async_retry

__all__ = ["perplexity_search_async"]


@traceable
@async_retry()
async def perplexity_search_async(search_queries: List[str]) -> List[Dict[str, Any]]:
    """Perform concurrent web searches using the Perplexity API asynchronously.

    Mirrors the behaviour of the legacy implementation in *utils.py* while
    living in its own dedicated module.
    """

    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable is not set.")

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }

    async def perform_search(query: str, client: httpx.AsyncClient) -> Dict[str, Any]:
        payload = {
            "model": "pplx-70b-online",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI assistant that provides concise and accurate answers.",
                },
                {"role": "user", "content": query},
            ],
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 1,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1,
        }
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]

            results: List[Dict[str, Any]] = [
                {
                    "title": f"Perplexity AI Answer for '{query}'",
                    "url": "https://www.perplexity.ai/",
                    "content": content,
                    "score": 1.0,
                    "raw_content": content,
                }
            ]

            return {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": results,
            }
        except httpx.HTTPStatusError:
            return {"query": query, "results": []}
        except Exception:
            return {"query": query, "results": []}

    async with httpx.AsyncClient() as client:
        tasks = [perform_search(q, client) for q in search_queries]
        return await asyncio.gather(*tasks) 