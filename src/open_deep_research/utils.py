"""Utility functions for Open Deep Research application.

This module provides core utility functions for web search, content processing,
text formatting, API integration, and data manipulation used throughout the research workflow.
"""

import asyncio
import concurrent.futures
import datetime
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Union,
    cast,
)
from urllib.parse import unquote

import aiohttp
import httpx
import requests
from aiohttp import ClientTimeout
from exa_py import Exa

# Handle conditional imports with proper typing
if TYPE_CHECKING:
    from linkup import LinkupClient

    LINKUP_AVAILABLE = True  # For type checking, assume it's available
else:
    try:
        from linkup import LinkupClient

        LINKUP_AVAILABLE = True
    except ImportError:
        LINKUP_AVAILABLE = False

        # Create a proper type-safe placeholder
        class LinkupClient:  # type: ignore[no-redef]
            """Placeholder class when linkup is not available."""

            def __init__(self):
                raise ImportError(
                    "LinkupClient is not available. Please install the linkup package."
                )


from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient as AsyncAzureAISearchClient
from bs4 import BeautifulSoup, Tag
from duckduckgo_search import DDGS
from langchain_anthropic import ChatAnthropic
from langchain_community.retrievers import ArxivRetriever
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from markdownify import markdownify
from pydantic import BaseModel

from open_deep_research.core.logging_utils import get_logger
from open_deep_research.credit_tracker import record_tavily
from open_deep_research.message_utils import (
    count_message_tokens,
    truncate_message_content,
)
from open_deep_research.prompts import SUMMARIZATION_PROMPT
from open_deep_research.pydantic_state import Section
from open_deep_research.tavily_tools import (
    acquire_tavily_semaphore,
    tavily_extract_tool,
)

logger = get_logger(__name__)

# --- safe_get helper (added) ---


def safe_get(obj: Any, key: str, default: Any | None = None) -> Any:
    """Safe getter that works with both dicts and objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    elif hasattr(obj, key):
        return getattr(obj, key, default)
    return default


def deduplicate_and_format_sources(
    search_response,
    max_tokens_per_source: int = 5000,
    include_raw_content: bool = True,
    deduplication_strategy: Literal["keep_first", "keep_last"] = "keep_first",
) -> str:
    """Format a list of search responses into a readable string with basic de-duplication and robust null handling."""
    formatted_output = ""
    unique_results: dict[str, Dict[str, Any]] = {}

    # Early exit on empty input
    if not search_response:
        return "No search results found."

    # Normalise to list
    sources_list = (
        search_response if isinstance(search_response, list) else [search_response]
    )

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
                continue  # skip duplicates when keeping first

            unique_results[url] = result  # keep last by default or overwrite

    # Build output roughly following the original style
    for i, (url, result) in enumerate(unique_results.items()):
        formatted_output += f"\n\nSource: {safe_get(result, 'title', 'Untitled')}\n"
        formatted_output += f"URL: {url}\n===\n"
        formatted_output += f"Most relevant content from source: {safe_get(result, 'content', '')}\n===\n"

        if include_raw_content:
            raw_content = safe_get(result, "raw_content", "")
            if raw_content:
                formatted_output += (
                    f"Full content:\n{raw_content[:max_tokens_per_source]}\n"
                )

    return formatted_output or "No valid search results found."


def format_sections(sections: list[Section]) -> str:
    """Format a list of sections into a string."""
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{"=" * 60}
Section {idx}: {section.name}
{"=" * 60}
Description:
{section.description}
Requires Research: 
{section.research}

Content:
{section.content if section.content else "[Not yet written]"}

"""
    return formatted_str


@traceable
async def azureaisearch_search_async(
    search_queries: list[str],
    max_results: int = 5,
    topic: str = "general",
    include_raw_content: bool = True,
) -> list[dict]:
    """Perform concurrent web searches using the Azure AI Search API.

    Args:
        search_queries (List[str]): list of search queries to process
        max_results (int): maximum number of results to return for each query
        topic (str): semantic topic filter for the search.
        include_raw_content (bool)

    Returns:
        List[dict]: list of search responses from Azure AI Search API, one per query.
    """
    # configure and create the Azure Search client
    # ensure all environment variables are set
    if not all(
        var in os.environ
        for var in [
            "AZURE_AI_SEARCH_ENDPOINT",
            "AZURE_AI_SEARCH_INDEX_NAME",
            "AZURE_AI_SEARCH_API_KEY",
        ]
    ):
        raise ValueError(
            "Missing required environment variables for Azure Search API which are: AZURE_AI_SEARCH_ENDPOINT, AZURE_AI_SEARCH_INDEX_NAME, AZURE_AI_SEARCH_API_KEY"
        )

    # Since we've verified the environment variables exist, they won't be None
    endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
    api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")

    assert endpoint is not None, (
        "AZURE_AI_SEARCH_ENDPOINT should not be None after validation"
    )
    assert index_name is not None, (
        "AZURE_AI_SEARCH_INDEX_NAME should not be None after validation"
    )
    assert api_key is not None, (
        "AZURE_AI_SEARCH_API_KEY should not be None after validation"
    )

    credential = AzureKeyCredential(api_key)

    reranker_key = "@search.reranker_score"

    async with AsyncAzureAISearchClient(endpoint, index_name, credential) as client:

        async def do_search(query: str) -> dict:
            # search query
            # Note: Azure AI Search accepts vector queries as dictionaries which are converted internally to VectorQuery objects
            paged = await client.search(
                search_text=query,
                vector_queries=cast(
                    "Any",
                    [
                        {
                            "fields": "vector",
                            "kind": "text",
                            "text": query,
                            "exhaustive": True,
                        }
                    ],
                ),
                semantic_configuration_name="fraunhofer-rag-semantic-config",
                query_type="semantic",
                select=["url", "title", "chunk", "creationTime", "lastModifiedTime"],
                top=max_results,
            )
            # async iterator to get all results
            items = [doc async for doc in paged]
            # Umwandlung in einfaches Dict-Format
            results = [
                {
                    "title": doc.get("title"),
                    "url": doc.get("url"),
                    "content": doc.get("chunk"),
                    "score": doc.get(reranker_key),
                    "raw_content": doc.get("chunk") if include_raw_content else None,
                }
                for doc in items
            ]
            return {"query": query, "results": results}

        # parallelize the search queries
        tasks = [do_search(q) for q in search_queries]
        return await asyncio.gather(*tasks)


@traceable
async def perplexity_search_async(search_queries: List[str]) -> List[Dict[str, Any]]:
    """Perform concurrent web searches using the Perplexity API asynchronously.

    Args:
        search_queries (List[str]): A list of queries to search for.

    Returns:
        list[dict]: A list of search results.
    """
    if not os.environ.get("PERPLEXITY_API_KEY"):
        raise ValueError("PERPLEXITY_API_KEY environment variable is not set.")

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {os.environ['PERPLEXITY_API_KEY']}",
    }

    async def perform_search(query: str, client: httpx.AsyncClient):
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
            response = await client.post(
                url, json=payload, headers=headers, timeout=30.0
            )
            response.raise_for_status()
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]

            # Create a simplified results structure
            results = [
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
        tasks = [perform_search(query, client) for query in search_queries]
        results = await asyncio.gather(*tasks)

    return results


@traceable
async def exa_search(
    search_queries: List[str],
    max_characters: int | None = None,
    num_results: int = 5,
    include_domains: List[str] | None = None,
    exclude_domains: List[str] | None = None,
    subpages: int | None = None,
) -> List[Dict[str, Any]]:
    """Search the web using the Exa API.

    Args:
        search_queries (List[str]): List of search queries to process
        max_characters (int, optional): Maximum number of characters to retrieve for each result's raw content.
                                       If None, the text parameter will be set to True instead of an object.
        num_results (int): Number of search results per query. Defaults to 5.
        include_domains (List[str], optional): List of domains to include in search results.
            When specified, only results from these domains will be returned.
        exclude_domains (List[str], optional): List of domains to exclude from search results.
            Cannot be used together with include_domains.
        subpages (int, optional): Number of subpages to retrieve per result. If None, subpages are not retrieved.

    Returns:
        List[dict]: List of search responses from Exa API, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,
                'answer': None,
                'images': list,
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the result
                        'content': str,          # Summary/snippet of content
                        'score': float,          # Relevance score
                        'raw_content': str|None  # Full content or None for secondary citations
                    },
                    ...
                ]
            }
    """
    # Check that include_domains and exclude_domains are not both specified
    if include_domains and exclude_domains:
        raise ValueError("Cannot specify both include_domains and exclude_domains")

    # Initialize Exa client (API key should be configured in your .env file)
    exa = Exa(api_key=f"{os.getenv('EXA_API_KEY')}")

    # Define the function to process a single query
    async def process_query(query):
        # Use run_in_executor to make the synchronous exa call in a non-blocking way
        loop = asyncio.get_event_loop()

        # Define the function for the executor with all parameters
        def exa_search_fn():
            # Build parameters dictionary
            kwargs = {
                # Set text to True if max_characters is None, otherwise use an object with max_characters
                "text": True
                if max_characters is None
                else {"max_characters": max_characters},
                "summary": True,  # This is an amazing feature by EXA. It provides an AI generated summary of the content based on the query
                "num_results": num_results,
            }

            # Add optional parameters only if they are provided
            if subpages is not None:
                kwargs["subpages"] = subpages

            if include_domains:
                kwargs["include_domains"] = include_domains
            elif exclude_domains:
                kwargs["exclude_domains"] = exclude_domains

            return exa.search_and_contents(query, **kwargs)

        response = await loop.run_in_executor(None, exa_search_fn)

        # Format the response to match the expected output structure
        formatted_results = []
        seen_urls = set()  # Track URLs to avoid duplicates

        # Helper function to safely get value regardless of if item is dict or object
        def get_value(item, key, default=None):
            # Delegate to the module-level helper for consistent behaviour
            return safe_get(item, key, default)

        # Access the results from the SearchResponse object
        results_list = safe_get(response, "results", [])

        # First process all main results
        for result in results_list:
            # Get the score with a default of 0.0 if it's None or not present
            score = safe_get(result, "score", 0.0)

            # Combine summary and text for content if both are available
            text_content = safe_get(result, "text", "")
            summary_content = safe_get(result, "summary", "")

            content = text_content
            if summary_content:
                if content:
                    content = f"{summary_content}\n\n{content}"
                else:
                    content = summary_content

            title = safe_get(result, "title", "")
            url = safe_get(result, "url", "")

            # Skip if we've seen this URL before (removes duplicate entries)
            if url in seen_urls:
                continue

            seen_urls.add(url)

            # Main result entry
            result_entry = {
                "title": title,
                "url": url,
                "content": content,
                "score": score,
                "raw_content": text_content,
            }

            # Add the main result to the formatted results
            formatted_results.append(result_entry)

        # Now process subpages only if the subpages parameter was provided
        if subpages is not None:
            for result in results_list:
                subpages_list = safe_get(result, "subpages", [])
                for subpage in subpages_list:
                    # Get subpage score
                    subpage_score = safe_get(subpage, "score", 0.0)

                    # Combine summary and text for subpage content
                    subpage_text = safe_get(subpage, "text", "")
                    subpage_summary = safe_get(subpage, "summary", "")

                    subpage_content = subpage_text
                    if subpage_summary:
                        if subpage_content:
                            subpage_content = f"{subpage_summary}\n\n{subpage_content}"
                        else:
                            subpage_content = subpage_summary

                    subpage_url = safe_get(subpage, "url", "")

                    # Skip if we've seen this URL before
                    if subpage_url in seen_urls:
                        continue

                    seen_urls.add(subpage_url)

                    formatted_results.append(
                        {
                            "title": safe_get(subpage, "title", ""),
                            "url": subpage_url,
                            "content": subpage_content,
                            "score": subpage_score,
                            "raw_content": subpage_text,
                        }
                    )

        # Collect images if available (only from main results to avoid duplication)
        images = []
        for result in results_list:
            image = safe_get(result, "image")
            if image and image not in images:  # Avoid duplicate images
                images.append(image)

        return {
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": images,
            "results": formatted_results,
        }

    # Process all queries sequentially with delay to respect rate limit
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests (0.25s = 4 requests per second, well within the 5/s limit)
            if i > 0:  # Don't delay the first request
                await asyncio.sleep(0.25)

            result = await process_query(query)
            search_docs.append(result)
        except Exception as e:
            # Handle exceptions gracefully
            # Add a placeholder result for failed queries to maintain index alignment
            search_docs.append(
                {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": [],
                    "error": str(e),
                }
            )

            # Add additional delay if we hit a rate limit error
            if "429" in str(e):
                await asyncio.sleep(1.0)  # Add a longer delay if we hit a rate limit

    return search_docs


@traceable
async def arxiv_search_async(
    search_queries: List[str],
    load_max_docs: int = 5,
    get_full_documents: bool = True,
    load_all_available_meta: bool = True,
) -> List[Dict[str, Any]]:
    """Perform concurrent searches on arXiv using the ArxivRetriever.

    Args:
        search_queries (List[str]): List of search queries or article IDs
        load_max_docs (int, optional): Maximum number of documents to return per query. Default is 5.
        get_full_documents (bool, optional): Whether to fetch full text of documents. Default is True.
        load_all_available_meta (bool, optional): Whether to load all available metadata. Default is True.

    Returns:
        List[dict]: List of search responses from arXiv, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the paper
                        'url': str,              # URL (Entry ID) of the paper
                        'content': str,          # Formatted summary with metadata
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str|None  # Full paper content if available
                    },
                    ...
                ]
            }
    """

    async def process_single_query(query):
        try:
            # Create retriever for each query - using current LangChain API
            retriever = ArxivRetriever(  # type: ignore
                load_max_docs=load_max_docs,
                get_full_documents=get_full_documents,
                load_all_available_meta=load_all_available_meta,
            )

            # Run the synchronous retriever in a thread pool
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: retriever.invoke(query))

            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0

            for i, doc in enumerate(docs):
                # Extract metadata
                metadata = doc.metadata

                # Use entry_id as the URL (this is the actual arxiv link)
                url = metadata.get("entry_id", "")

                # Format content with all useful metadata
                content_parts = []

                # Primary information
                if "Summary" in metadata:
                    content_parts.append(f"Summary: {metadata['Summary']}")

                if "Authors" in metadata:
                    content_parts.append(f"Authors: {metadata['Authors']}")

                # Add publication information
                published = metadata.get("Published")
                published_str = ""
                if published:
                    if hasattr(published, "isoformat"):
                        published_str = published.isoformat()
                    else:
                        published_str = str(published)
                if published_str:
                    content_parts.append(f"Published: {published_str}")

                # Add additional metadata if available
                if "primary_category" in metadata:
                    content_parts.append(
                        f"Primary Category: {metadata['primary_category']}"
                    )

                if "categories" in metadata and metadata["categories"]:
                    content_parts.append(
                        f"Categories: {', '.join(metadata['categories'])}"
                    )

                if "comment" in metadata and metadata["comment"]:
                    content_parts.append(f"Comment: {metadata['comment']}")

                if "journal_ref" in metadata and metadata["journal_ref"]:
                    content_parts.append(
                        f"Journal Reference: {metadata['journal_ref']}"
                    )

                if "doi" in metadata and metadata["doi"]:
                    content_parts.append(f"DOI: {metadata['doi']}")

                # Get PDF link if available in the links
                pdf_link = ""
                if "links" in metadata and metadata["links"]:
                    for link in metadata["links"]:
                        if "pdf" in link:
                            pdf_link = link
                            content_parts.append(f"PDF: {pdf_link}")
                            break

                # Join all content parts with newlines
                content = "\n".join(content_parts)

                result = {
                    "title": metadata.get("Title", ""),
                    "url": url,  # Using entry_id as the URL
                    "content": content,
                    "score": base_score - (i * score_decrement),
                    "raw_content": doc.page_content if get_full_documents else None,
                }
                results.append(result)

            return {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": results,
            }
        except Exception as e:
            # Handle exceptions gracefully
            return {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "error": str(e),
            }

    # Process queries sequentially with delay to respect arXiv rate limit (1 request per 3 seconds)
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests (3 seconds per ArXiv's rate limit)
            if i > 0:  # Don't delay the first request
                await asyncio.sleep(3.0)

            result = await process_single_query(query)
            search_docs.append(result)
        except Exception as e:
            # Handle exceptions gracefully
            search_docs.append(
                {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": [],
                    "error": str(e),
                }
            )

            # Add additional delay if we hit a rate limit error
            if "429" in str(e) or "Too Many Requests" in str(e):
                await asyncio.sleep(5.0)  # Add a longer delay if we hit a rate limit

    return search_docs


@traceable
async def pubmed_search_async(
    search_queries: List[str],
    top_k_results: int = 5,
    email: str | None = None,
    api_key: str | None = None,
    doc_content_chars_max: int = 4000,
) -> List[Dict[str, Any]]:
    """Perform concurrent searches on PubMed using the PubMedAPIWrapper.

    Args:
        search_queries (List[str]): List of search queries
        top_k_results (int, optional): Maximum number of documents to return per query. Default is 5.
        email (str, optional): Email address for PubMed API. Required by NCBI.
        api_key (str, optional): API key for PubMed API for higher rate limits.
        doc_content_chars_max (int, optional): Maximum characters for document content. Default is 4000.

    Returns:
        List[dict]: List of search responses from PubMed, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the paper
                        'url': str,              # URL to the paper on PubMed
                        'content': str,          # Formatted summary with metadata
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str       # Full abstract content
                    },
                    ...
                ]
            }
    """

    async def process_single_query(query):
        try:
            # print(f"Processing PubMed query: '{query}'")

            # Create PubMed wrapper for the query
            wrapper = PubMedAPIWrapper(
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max,
                email=email if email else "your_email@example.com",
                api_key=api_key if api_key else "",
                parse=True,  # Default to parsing the results
            )

            # Run the synchronous wrapper in a thread pool
            loop = asyncio.get_event_loop()

            # Use wrapper.lazy_load instead of load to get better visibility
            docs = await loop.run_in_executor(
                None, lambda: list(wrapper.lazy_load(query))
            )

            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0

            for i, doc in enumerate(docs):
                # Format content with metadata
                content_parts = []

                if doc.get("Published"):
                    content_parts.append(f"Published: {doc['Published']}")

                if doc.get("Copyright Information"):
                    content_parts.append(
                        f"Copyright Information: {doc['Copyright Information']}"
                    )

                if doc.get("Summary"):
                    content_parts.append(f"Summary: {doc['Summary']}")

                # Generate PubMed URL from the article UID
                uid = doc.get("uid", "")
                url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/" if uid else ""

                # Join all content parts with newlines
                content = "\n".join(content_parts)

                result = {
                    "title": doc.get("Title", ""),
                    "url": url,
                    "content": content,
                    "score": base_score - (i * score_decrement),
                    "raw_content": doc.get("Summary", ""),
                }
                results.append(result)

            return {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": results,
            }
        except Exception as e:
            # Handle exceptions with more detailed information
            logger.error("Error processing PubMed query '%s': %s", query, str(e))

            return {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "error": str(e),
            }

    # Process all queries with a reasonable delay between them
    search_docs = []

    # Start with a small delay that increases if we encounter rate limiting
    delay = 1.0  # Start with a more conservative delay

    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests
            if i > 0:  # Don't delay the first request
                # print(f"Waiting {delay} seconds before next query...")
                await asyncio.sleep(delay)

            result = await process_single_query(query)
            search_docs.append(result)

            # If query was successful with results, we can slightly reduce delay (but not below minimum)
            if result.get("results") and len(result["results"]) > 0:
                delay = max(0.5, delay * 0.9)  # Don't go below 0.5 seconds

        except Exception as e:
            # Handle exceptions gracefully
            logger.error(
                "Error in main loop processing PubMed query '%s': %s", query, str(e)
            )

            search_docs.append(
                {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": [],
                    "error": str(e),
                }
            )

            # If we hit an exception, increase delay for next query
            delay = min(5.0, delay * 1.5)  # Don't exceed 5 seconds

    return search_docs


@traceable
async def linkup_search(
    search_queries: List[str], depth: str | None = "standard"
) -> List[Dict[str, Any]]:
    """Perform concurrent web searches using Linkup API.

    Args:
        search_queries (List[str]): List of search queries to process
        depth (str): Search depth, either "standard" or "deep"

    Returns:
        List[dict]: List of search responses from Linkup, one per query
    """
    # Validate depth parameter and cast to Literal type
    if depth not in ["standard", "deep"]:
        depth = "standard"
    depth_literal: Literal["standard", "deep"] = depth  # type: ignore

    # Import and check for Linkup client
    try:
        from linkup import LinkupClient
    except ImportError:
        logger.warning("Linkup client not available. Install with: pip install linkup")
        return []

    # Initialize client
    client = LinkupClient()

    # Create search tasks
    search_tasks = []
    for query in search_queries:
        search_tasks.append(
            client.async_search(
                query,
                depth_literal,  # Now correctly typed as Literal['standard', 'deep']
                output_type="searchResults",
            )
        )

    search_results = []
    for response in await asyncio.gather(*search_tasks):
        search_results.append(
            {
                "results": [
                    {"title": result.name, "url": result.url, "content": result.content}
                    for result in response.results
                ],
            }
        )

    return search_results


@traceable
async def google_search_async(
    search_queries: Union[str, List[str]],
    max_results: int = 5,
    include_raw_content: bool = True,
) -> List[Dict[str, Any]]:
    """Perform concurrent web searches using Google.

    Uses Google Custom Search API if environment variables are set, otherwise falls back to web scraping.

    Args:
        search_queries (List[str]): List of search queries to process
        max_results (int): Maximum number of results to return per query
        include_raw_content (bool): Whether to fetch full page content

    Returns:
        List[dict]: List of search responses from Google, one per query
    """
    # Check for API credentials from environment variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CX")
    use_api = bool(api_key and cx)

    # Handle case where search_queries is a single string
    if isinstance(search_queries, str):
        search_queries = [search_queries]

    # Define user agent generator
    def get_useragent():
        """Generate a random user agent string."""
        lynx_version = (
            f"Lynx/{random.randint(2, 3)}.{random.randint(8, 9)}.{random.randint(0, 2)}"
        )
        libwww_version = f"libwww-FM/{random.randint(2, 3)}.{random.randint(13, 15)}"
        ssl_mm_version = f"SSL-MM/{random.randint(1, 2)}.{random.randint(3, 5)}"
        openssl_version = f"OpenSSL/{random.randint(1, 3)}.{random.randint(0, 4)}.{random.randint(0, 9)}"
        return f"{lynx_version} {libwww_version} {ssl_mm_version} {openssl_version}"

    # Create executor for running synchronous operations
    executor = None if use_api else concurrent.futures.ThreadPoolExecutor(max_workers=5)

    # Use a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(5 if use_api else 2)

    async def search_single_query(query):
        async with semaphore:
            try:
                results = []

                # API-based search
                if use_api:
                    # The API returns up to 10 results per request
                    for start_index in range(1, max_results + 1, 10):
                        # Calculate how many results to request in this batch
                        num = min(10, max_results - (start_index - 1))

                        # Make request to Google Custom Search API
                        params = {
                            "q": query,
                            "key": api_key,
                            "cx": cx,
                            "start": start_index,
                            "num": num,
                        }

                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                "https://www.googleapis.com/customsearch/v1",
                                params=params,
                            ) as response:
                                if response.status != 200:
                                    await response.text()
                                    break

                                data = await response.json()

                                # Process search results
                                for item in data.get("items", []):
                                    result = {
                                        "title": item.get("title", ""),
                                        "url": item.get("link", ""),
                                        "content": item.get("snippet", ""),
                                        "score": None,
                                        "raw_content": item.get("snippet", ""),
                                    }
                                    results.append(result)

                        # Respect API quota with a small delay
                        await asyncio.sleep(0.2)

                        # If we didn't get a full page of results, no need to request more
                        if not data.get("items") or len(data.get("items", [])) < num:
                            break

                # Web scraping based search
                else:
                    # Add delay between requests
                    await asyncio.sleep(0.5 + random.random() * 1.5)

                    # Define scraping function
                    def google_search(query, max_results):
                        try:
                            lang = "en"
                            safe = "active"
                            start = 0
                            fetched_results = 0
                            fetched_links = set()
                            search_results = []

                            while fetched_results < max_results:
                                # Send request to Google
                                resp = requests.get(
                                    url="https://www.google.com/search",
                                    headers={
                                        "User-Agent": get_useragent(),
                                        "Accept": "*/*",
                                    },
                                    params={
                                        "q": query,
                                        "num": max_results + 2,
                                        "hl": lang,
                                        "start": start,
                                        "safe": safe,
                                    },
                                    cookies={
                                        "CONSENT": "PENDING+987",  # Bypasses the consent page
                                        "SOCS": "CAESHAgBEhIaAB",
                                    },
                                )
                                resp.raise_for_status()

                                # Parse results
                                soup = BeautifulSoup(resp.text, "html.parser")
                                result_block = soup.find_all("div", class_="ezO2md")
                                new_results = 0

                                for result in result_block:
                                    # Type checking for BeautifulSoup elements
                                    if not isinstance(result, Tag):
                                        continue

                                    link_tag = result.find("a", href=True)
                                    title_tag = (
                                        link_tag.find("span", class_="CVA68e")
                                        if isinstance(link_tag, Tag)
                                        else None
                                    )
                                    description_tag = result.find(
                                        "span", class_="FrIlee"
                                    )

                                    if (
                                        isinstance(link_tag, Tag)
                                        and isinstance(title_tag, Tag)
                                        and isinstance(description_tag, Tag)
                                    ):
                                        href_attr = link_tag.get("href", "")
                                        if isinstance(href_attr, str):
                                            link = unquote(
                                                href_attr.split("&")[0].replace(
                                                    "/url?q=", ""
                                                )
                                            )

                                        if link in fetched_links:
                                            continue

                                        fetched_links.add(link)
                                        title = title_tag.text
                                        description = description_tag.text

                                        # Store result in the same format as the API results
                                        search_results.append(
                                            {
                                                "title": title,
                                                "url": link,
                                                "content": description,
                                                "score": None,
                                                "raw_content": description,
                                            }
                                        )

                                        fetched_results += 1
                                        new_results += 1

                                        if fetched_results >= max_results:
                                            break

                                if new_results == 0:
                                    break

                                start += 10
                                time.sleep(1)  # Delay between pages

                            return search_results

                        except Exception:
                            return []

                    # Execute search in thread pool
                    loop = asyncio.get_running_loop()
                    search_results = await loop.run_in_executor(
                        executor, lambda: google_search(query, max_results)
                    )

                    # Process the results
                    results = search_results

                # If requested, fetch full page content asynchronously (for both API and web scraping)
                if include_raw_content and results:
                    content_semaphore = asyncio.Semaphore(3)

                    async with aiohttp.ClientSession() as session:
                        fetch_tasks = []

                        async def fetch_full_content(result):
                            async with content_semaphore:
                                url = result["url"]
                                headers = {
                                    "User-Agent": get_useragent(),
                                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                                }

                                try:
                                    await asyncio.sleep(0.2 + random.random() * 0.6)
                                    async with session.get(
                                        url,
                                        headers=headers,
                                        timeout=ClientTimeout(total=10),
                                    ) as response:
                                        if response.status == 200:
                                            # Check content type to handle binary files
                                            content_type = response.headers.get(
                                                "Content-Type", ""
                                            ).lower()

                                            # Handle PDFs and other binary files
                                            if (
                                                "application/pdf" in content_type
                                                or "application/octet-stream"
                                                in content_type
                                            ):
                                                # For PDFs, indicate that content is binary and not parsed
                                                result["raw_content"] = (
                                                    f"[Binary content: {content_type}. Content extraction not supported for this file type.]"
                                                )
                                            else:
                                                try:
                                                    # Try to decode as UTF-8 with replacements for non-UTF8 characters
                                                    html = await response.text(
                                                        errors="replace"
                                                    )
                                                    soup = BeautifulSoup(
                                                        html, "html.parser"
                                                    )
                                                    result["raw_content"] = (
                                                        soup.get_text()
                                                    )
                                                except UnicodeDecodeError as ude:
                                                    # Fallback if we still have decoding issues
                                                    result["raw_content"] = (
                                                        f"[Could not decode content: {str(ude)}]"
                                                    )
                                except Exception as e:
                                    result["raw_content"] = (
                                        f"[Error fetching content: {str(e)}]"
                                    )
                                return result

                        for result in results:
                            fetch_tasks.append(fetch_full_content(result))

                        updated_results = await asyncio.gather(*fetch_tasks)
                        results = updated_results

                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": results,
                }
            except Exception:
                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": [],
                }

    try:
        # Create tasks for all search queries
        search_tasks = [search_single_query(query) for query in search_queries]

        # Execute all searches concurrently
        search_results = await asyncio.gather(*search_tasks)

        return search_results
    finally:
        # Only shut down executor if it was created
        if executor:
            executor.shutdown(wait=False)


async def scrape_pages(titles: List[str], urls: List[str]) -> str:
    """Scrapes content from a list of URLs and formats it into a readable markdown document.

    This function:
    1. Takes a list of page titles and URLs
    2. Makes asynchronous HTTP requests to each URL
    3. Converts HTML content to markdown
    4. Formats all content with clear source attribution

    Args:
        titles (List[str]): A list of page titles corresponding to each URL
        urls (List[str]): A list of URLs to scrape content from

    Returns:
        str: A formatted string containing the full content of each page in markdown format,
             with clear section dividers and source attribution
    """
    # Create an async HTTP client
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        pages = []

        # Fetch each URL and convert to markdown
        for url in urls:
            try:
                # Fetch the content
                response = await client.get(url)
                response.raise_for_status()

                # Convert HTML to markdown if successful
                if response.status_code == 200:
                    # Handle different content types
                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" in content_type:
                        # Convert HTML to markdown
                        markdown_content = markdownify(response.text)
                        pages.append(markdown_content)
                    else:
                        # For non-HTML content, just mention the content type
                        pages.append(
                            f"Content type: {content_type} (not converted to markdown)"
                        )
                else:
                    pages.append(f"Error: Received status code {response.status_code}")

            except Exception as e:
                # Handle any exceptions during fetch
                pages.append(f"Error fetching URL: {str(e)}")

        # Create formatted output
        formatted_output = "Search results: \n\n"

        for i, (title, url, page) in enumerate(zip(titles, urls, pages)):
            formatted_output += f"\n\n--- SOURCE {i + 1}: {title} ---\n"
            formatted_output += f"URL: {url}\n\n"
            formatted_output += f"FULL CONTENT:\n {page}"
            formatted_output += "\n\n" + "-" * 80 + "\n"

    return formatted_output


@tool
async def duckduckgo_search(search_queries: List[str]):
    """Perform searches using DuckDuckGo with retry logic to handle rate limits.

    Args:
        search_queries (List[str]): List of search queries to process

    Returns:
        str: A formatted string of search results
    """

    async def process_single_query(query):
        # Execute synchronous search in the event loop's thread pool
        loop = asyncio.get_event_loop()

        def perform_search():
            max_retries = 3
            retry_count = 0
            backoff_factor = 2.0
            last_exception = None

            while retry_count <= max_retries:
                try:
                    results = []
                    with DDGS() as ddgs:
                        # Change query slightly and add delay between retries
                        if retry_count > 0:
                            # Random delay with exponential backoff
                            delay = backoff_factor**retry_count + random.random()
                            time.sleep(delay)

                            # Add a random element to the query to bypass caching/rate limits
                            modifiers = [
                                "about",
                                "info",
                                "guide",
                                "overview",
                                "details",
                                "explained",
                            ]
                            modified_query = f"{query} {random.choice(modifiers)}"
                        else:
                            modified_query = query

                        # Execute search
                        ddg_results = list(ddgs.text(modified_query, max_results=5))

                        # Format results
                        for i, result in enumerate(ddg_results):
                            results.append(
                                {
                                    "title": result.get("title", ""),
                                    "url": result.get("href", ""),
                                    "content": result.get("body", ""),
                                    "score": 1.0
                                    - (i * 0.1),  # Simple scoring mechanism
                                    "raw_content": result.get("body", ""),
                                }
                            )

                        # Return successful results
                        return {
                            "query": query,
                            "follow_up_questions": None,
                            "answer": None,
                            "images": [],
                            "results": results,
                        }
                except Exception as e:
                    # Store the exception and retry
                    last_exception = e
                    retry_count += 1

                    # If not a rate limit error, don't retry
                    if "Ratelimit" not in str(e) and retry_count >= 1:
                        break

            # If we reach here, all retries failed
            # Return empty results but with query info preserved
            return {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "error": str(last_exception),
            }

        return await loop.run_in_executor(None, perform_search)

    # Process queries with delay between them to reduce rate limiting
    search_docs = []
    urls = []
    titles = []
    for i, query in enumerate(search_queries):
        # Add delay between queries (except first one)
        if i > 0:
            delay = 2.0 + random.random() * 2.0  # Random delay 2-4 seconds
            await asyncio.sleep(delay)

        # Process the query
        result = await process_single_query(query)
        search_docs.append(result)

        # Safely extract URLs and titles from results, handling empty result cases
        results_list = safe_get(result, "results", [])
        if results_list:
            for res in results_list:
                url = safe_get(res, "url")
                title = safe_get(res, "title")
                if url and title:
                    urls.append(url)
                    titles.append(title)

    # If we got any valid URLs, scrape the pages
    if urls:
        return await scrape_pages(titles, urls)
    else:
        return "No valid search results found. Please try different search queries or use a different search API."


async def select_and_execute_search(
    search_api: str, query_list: list[str], params_to_pass: dict
) -> str:
    """Select and execute the appropriate search function based on the search_api string.

    Args:
        search_api (str): The search API to use (e.g., "tavily", "perplexity", "none").
        query_list (list[str]): The list of search queries.
        params_to_pass (dict): A dictionary of parameters for the search function.

    Returns:
        str: The formatted search results.
    """
    # Handle "none" search API - return empty results
    if search_api.lower() == "none":
        return "No search performed (search API set to 'none')."

    # Web search
    if search_api == "tavily":
        from langchain_tavily import TavilySearch

        from open_deep_research.tavily_tools import (
            acquire_tavily_semaphore,
        )

        async def _run_search(q: str):
            async with acquire_tavily_semaphore():
                invoke_args = {k: v for k, v in params_to_pass.items() if v is not None}
                invoke_args["query"] = q
                # Record estimated credit usage BEFORE making the call so we
                # capture errors/timeouts too.
                depth_val = invoke_args.get("search_depth", "advanced")
                record_tavily("search", depth_val, query=q)

                # Create a dynamic Tavily tool with the correct parameters
                # This ensures the search_depth from configuration is respected
                tavily_tool = TavilySearch(
                    search_depth=depth_val,
                    max_results=invoke_args.get("max_results", 5),
                    include_raw_content=False,
                )
                return await tavily_tool.ainvoke(invoke_args)

        search_results = await asyncio.gather(*[_run_search(q) for q in query_list])
        search_results = await _supplement_with_extract(search_results)
    elif search_api == "perplexity":
        search_results = await perplexity_search_async(query_list)
    elif search_api == "exa":
        search_results = await exa_search(query_list, **params_to_pass)
    elif search_api == "arxiv":
        search_results = await arxiv_search_async(query_list, **params_to_pass)
    elif search_api == "pubmed":
        search_results = await pubmed_search_async(query_list, **params_to_pass)
    elif search_api == "linkup":
        search_results = await linkup_search(query_list, **params_to_pass)
    elif search_api == "googlesearch":
        search_results = await google_search_async(query_list, **params_to_pass)
    elif search_api == "duckduckgo":
        # duckduckgo_search returns a formatted string directly, not a list of dicts
        return await duckduckgo_search.ainvoke({"search_queries": query_list})
    elif search_api == "azureaisearch":
        search_results = await azureaisearch_search_async(query_list, **params_to_pass)
    else:
        raise ValueError(f"Unsupported search API: {search_api}")

    return deduplicate_and_format_sources(
        search_results, max_tokens_per_source=4000, deduplication_strategy="keep_first"
    )


class Summary(BaseModel):
    """Summary model for webpage content."""

    summary: str
    key_excerpts: list[str]


async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """Summarize webpage content."""
    try:
        user_input_content: Union[str, List[Dict[str, Any]]] = (
            "Please summarize the article"
        )
        if isinstance(model, ChatAnthropic):
            user_input_content = [
                {
                    "type": "text",
                    "text": user_input_content,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ]

        summary = (
            await model.with_structured_output(Summary)
            .with_retry(stop_after_attempt=2)
            .ainvoke(
                [
                    {
                        "role": "system",
                        "content": SUMMARIZATION_PROMPT.format(
                            webpage_content=webpage_content
                        ),
                    },
                    {"role": "user", "content": user_input_content},
                ]
            )
        )
    except Exception:
        # fall back on the raw content
        return webpage_content

    def format_summary(summary: Summary):
        excerpts_str = "\n".join(f"- {e}" for e in summary.key_excerpts)
        return f"""<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{excerpts_str}\n</key_excerpts>"""

    # Ensure summary is the correct type
    if isinstance(summary, Summary):
        return format_summary(summary)
    else:
        # Fallback for unexpected types
        return webpage_content


def split_and_rerank_search_results(
    embeddings: Embeddings, query: str, search_results: list[dict], max_chunks: int = 5
):
    """Split search results into chunks and rerank them based on relevance to the query.

    Args:
        embeddings: The embeddings model to use for similarity search
        query: The search query to use for relevance ranking
        search_results: List of search result dictionaries
        max_chunks: Maximum number of chunks to return

    Returns:
        List of the most relevant document chunks
    """
    # split webpage content into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200, add_start_index=True
    )
    documents = [
        Document(
            page_content=safe_get(result, "raw_content")
            or safe_get(result, "content", ""),
            metadata={
                "url": safe_get(result, "url", ""),
                "title": safe_get(result, "title", "Untitled"),
            },
        )
        for result in search_results
    ]
    all_splits = text_splitter.split_documents(documents)

    # index chunks
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=all_splits)

    # retrieve relevant chunks
    retrieved_docs = vector_store.similarity_search(query, k=max_chunks)
    return retrieved_docs


def stitch_documents_by_url(documents: list[Document]) -> list[Document]:
    """Stitch together document chunks from the same URL while deduplicating content.

    Args:
        documents: List of Document objects to stitch together

    Returns:
        List of Document objects with content from the same URL combined
    """
    url_to_docs: defaultdict[str, list[Document]] = defaultdict(list)
    url_to_snippet_hashes: defaultdict[str, set[str]] = defaultdict(set)
    for doc in documents:
        snippet_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
        url = doc.metadata["url"]
        # deduplicate snippets by the content
        if snippet_hash in url_to_snippet_hashes[url]:
            continue

        url_to_docs[url].append(doc)
        url_to_snippet_hashes[url].add(snippet_hash)

    # stitch retrieved chunks into a single doc per URL
    stitched_docs = []
    for docs in url_to_docs.values():
        stitched_doc = Document(
            page_content="\n\n".join([f"...{doc.page_content}..." for doc in docs]),
            metadata=cast("Document", docs[0]).metadata,
        )
        stitched_docs.append(stitched_doc)

    return stitched_docs


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    now = datetime.datetime.now()
    # Use cross-platform approach to avoid leading zero in day
    return now.strftime("%a %b {}, %Y").format(now.day)


async def load_mcp_server_config(path: str) -> dict:
    """Load MCP server configuration from a file."""

    def _load():
        with open(path) as f:
            config = json.load(f)
        return config

    config = await asyncio.to_thread(_load)
    return config


def is_api_key_available(provider: str) -> bool:
    """Check if an API key is available for a given provider.

    Args:
        provider: The provider name (e.g., "openai", "anthropic", "deepseek")

    Returns:
        True if the API key is available, False otherwise
    """
    import os

    # Map provider names to their environment variable names
    provider_env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "together": "TOGETHER_API_KEY",
        "groq": "GROQ_API_KEY",
        "google": "GOOGLE_API_KEY",
        "cohere": "COHERE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
        "tavily": "TAVILY_API_KEY",
        "exa": "EXA_API_KEY",
        "azure": "AZURE_AI_SEARCH_API_KEY",
    }

    env_var = provider_env_map.get(provider.lower())
    if not env_var:
        return False

    api_key = os.environ.get(env_var)
    return api_key is not None and api_key.strip() != ""


def _create_fallback_instance(schema_class: type[BaseModel]) -> BaseModel:
    """Create a fallback instance of a schema class with appropriate defaults.

    Args:
        schema_class: The Pydantic schema class to create a fallback instance for

    Returns:
        A valid instance of the schema class with sensible defaults
    """
    # Import here to avoid circular imports
    from .pydantic_state import SectionOutput, Sections

    # Handle specific schema classes with appropriate defaults
    if schema_class == SectionOutput:
        return SectionOutput(
            section_content="Unable to generate section content. Please try again."
        )
    elif schema_class == Sections:
        return Sections(sections=[])
    elif schema_class.__name__ == "Feedback":
        # Handle Feedback class dynamically since it might not be imported
        try:
            return schema_class(grade="fail", follow_up_queries=[])
        except Exception:
            # If we can't create it with those defaults, try with no args
            return schema_class()
    elif schema_class.__name__ == "Queries":
        # Handle Queries class with empty list
        try:
            return schema_class(queries=[])
        except Exception:
            return schema_class()
    else:
        # For other schema classes, try to create with no args first
        try:
            return schema_class()
        except Exception:
            # Try to create a minimal instance with likely required fields
            try:
                # Get the schema fields and try to provide minimal defaults
                fields = getattr(schema_class, "__fields__", {})
                defaults: Dict[str, Any] = {}

                # Common fallback values for different field types
                for field_name, field_info in fields.items():
                    if hasattr(field_info, "annotation"):
                        field_type = field_info.annotation
                        # Handle common types
                        field_value: Any
                        if field_type is str:
                            field_value = "Fallback value"
                        elif field_type is int:
                            field_value = 0
                        elif field_type is bool:
                            field_value = False
                        elif field_type is list or str(field_type).startswith(
                            "typing.List"
                        ):
                            field_value = []
                        elif field_type is dict or str(field_type).startswith(
                            "typing.Dict"
                        ):
                            field_value = {}
                        else:
                            # For unknown types, try to provide None if it's Optional
                            if str(field_type).startswith("typing.Union") or str(
                                field_type
                            ).startswith("typing.Optional"):
                                field_value = None
                            else:
                                continue  # Skip this field if we can't determine a default
                        
                        defaults[field_name] = field_value

                return schema_class(**defaults)
            except Exception:
                # Last resort: return a basic object that might work
                return schema_class.__new__(schema_class)


async def get_structured_output_with_fallback(
    model: BaseChatModel,
    schema_class: type[BaseModel],
    messages: List[BaseMessage],
    model_id: str | None = None,
) -> BaseModel:
    """Get structured output from a model with fallback handling.

    Args:
        model: The language model to use
        schema_class: The Pydantic schema class to structure the output
        messages: List of messages to send to the model
        model_id: Optional model identifier for debugging

    Returns:
        Structured output matching the schema_class
    """
    import httpx

    from open_deep_research.exceptions import FatalModelError

    def _is_transient(exc: Exception) -> bool:  # noqa: D401 – small inline helper
        """Return *True* if *exc* is considered transient/network-related."""

        transient_types = (httpx.HTTPError,)
        if isinstance(exc, transient_types):
            return True

        # openai specific – treat explicit rate-limit or timeout as transient
        if "rate limit" in str(exc).lower() or "timeout" in str(exc).lower():
            return True

        return False

    try:
        # Try the standard structured output approach
        # Add automatic retries for transient LLM/tool failures (rate-limits,
        # 5xx, timeouts). Keeping attempts small since the higher-level graph
        # already retries – this guards the single call itself.
        structured_model = model.with_structured_output(schema_class).with_retry(
            stop_after_attempt=3
        )
        result = await structured_model.ainvoke(messages)
        return cast("BaseModel", result)
    except Exception as e:
        # If the error is fatal (auth, invalid request) immediately re-raise as
        # ``FatalModelError`` so upstream retry loops don’t hammer the API.
        if not _is_transient(e):
            raise FatalModelError(str(e), provider=model_id) from e

        # If structured output fails, fall back to prompt-based parsing

        # Add JSON format instructions to the last message
        parser: PydanticOutputParser[BaseModel] = PydanticOutputParser(
            pydantic_object=schema_class
        )
        format_instructions = parser.get_format_instructions()

        # Modify the last message to include format instructions
        modified_messages = messages.copy()
        if modified_messages and hasattr(modified_messages[-1], "content"):
            # Ensure content is a string before concatenating
            if isinstance(modified_messages[-1].content, str):
                modified_messages[-1].content += f"\n\n{format_instructions}"
            elif isinstance(modified_messages[-1].content, list):
                # If content is a list, append the format instructions as a new element
                modified_messages[-1].content.append(f"\n\n{format_instructions}")
            else:
                # Convert to string and concatenate
                modified_messages[-1].content = (
                    str(modified_messages[-1].content) + f"\n\n{format_instructions}"
                )

        # Get raw response
        raw_response = await model.ainvoke(modified_messages)

        # Parse the response
        try:
            if hasattr(raw_response, "content"):
                response_text = raw_response.content
            else:
                response_text = str(raw_response)

            # Try to parse JSON from the response
            parsed_result = parser.parse(str(response_text))
            return cast("BaseModel", parsed_result)
        except Exception:
            # ------------------------------------------------------------------
            # If JSON parsing fails COMPLETELY we still want to preserve the raw
            # assistant output.  We create a minimal valid instance and, when
            # possible, stuff the raw text into a suitable *str* field so it can
            # propagate through the pipeline instead of being lost.
            # ------------------------------------------------------------------
            fallback_instance = _create_fallback_instance(schema_class)

            # Try to identify a primary *content* field where we can store the
            # raw response.  We scan common names first ("section_content",
            # "content") but fall back to the first string field we find.
            target_field = None
            for name in ("section_content", "content"):
                if (
                    name in fallback_instance.model_fields
                    and isinstance(
                        fallback_instance.model_fields[name].annotation, type
                    )
                    and fallback_instance.model_fields[name].annotation is str
                ):
                    target_field = name
                    break

            if target_field is None:
                # Find the first str-typed field in the model
                for field_name, field_info in fallback_instance.model_fields.items():
                    if field_info.annotation is str:
                        target_field = field_name
                        break

            if target_field:
                try:
                    object.__setattr__(fallback_instance, target_field, response_text)
                except Exception:
                    # If the model is frozen or attribute setting fails we just
                    # leave the fallback instance as-is.
                    pass

            return fallback_instance


def filter_think_tokens(text: str) -> str:
    """Remove <think>...</think> tokens from text.

    Args:
        text: Input text that may contain think tokens

    Returns:
        Text with think tokens removed
    """
    if not isinstance(text, str):
        return text

    # Remove <think>...</think> blocks (handle nested tags by looping)
    import re

    pattern = r"<think>.*?</think>"
    while re.search(pattern, text, flags=re.DOTALL):
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    return text.strip()


def format_sections_for_final_report(sections: List[Section]) -> str:
    """Format sections for the final report."""
    formatted_text = ""
    for section in sections:
        # Use # for introduction/report title, ## for all other sections
        if section.name.lower() in ["introduction", "intro"]:
            formatted_text += f"\n\n# {section.name}\n\n{section.content}\n"
        else:
            formatted_text += f"\n\n## {section.name}\n\n{section.content}\n"
    return formatted_text


def format_sections_for_context(sections: List[Section]) -> str:
    """Format sections to provide context for other operations."""
    formatted_text = "Current sections:\n"
    for i, section in enumerate(sections, 1):
        formatted_text += f"\n{i}. {section.name}\n"
        formatted_text += f"   Description: {section.description}\n"
        formatted_text += (
            f"   Content: {section.content[:200]}...\n"
            if len(section.content) > 200
            else f"   Content: {section.content}\n"
        )
    return formatted_text


async def summarize_search_results(
    source_str: str, max_tokens: int = 6000, model: str | None = None
) -> str:
    """Summarize search results to fit within token limits.

    Args:
        source_str: The search results to summarize
        max_tokens: Maximum number of tokens to allow
        model: Model identifier for context-aware summarization

    Returns:
        Summarized search results
    """
    # Simple truncation for now - could be enhanced with actual summarization
    char_limit = max_tokens * 4  # Rough estimate of 4 chars per token
    if len(source_str) <= char_limit:
        return source_str

    return (
        source_str[:char_limit]
        + "\n\n... [Content truncated to fit model context limits]"
    )


def count_messages_tokens(
    messages: List[Union[Dict[str, Any], BaseMessage]],
    model_name: str = "gpt-3.5-turbo",
) -> int:
    """Count tokens in a list of messages."""
    total_tokens = 0
    for message in messages:
        total_tokens += count_message_tokens(message, model_name)
    return total_tokens


async def truncate_messages_for_context(
    messages: List[Union[Dict[str, Any], BaseMessage]],
    max_context_tokens: int,
    model_name: str = "gpt-3.5-turbo",
) -> List[Union[Dict[str, Any], BaseMessage]]:
    """Truncate messages to fit within context limits, preserving the most recent messages."""
    if not messages:
        return messages

    # Count tokens for all messages
    total_tokens = count_messages_tokens(messages, model_name)

    # If already within limits, return as-is
    if total_tokens <= max_context_tokens:
        return messages

    # Keep the most recent messages, truncating from the beginning
    truncated_messages: List[Union[Dict[str, Any], BaseMessage]] = []
    current_tokens = 0

    # Start from the most recent message and work backwards
    for message in reversed(messages):
        message_tokens = count_message_tokens(message, model_name)

        if current_tokens + message_tokens <= max_context_tokens:
            truncated_messages.insert(0, message)
            current_tokens += message_tokens
        else:
            # If this message would exceed the limit, try to truncate its content
            if isinstance(message, dict):
                content = message.get("content", "")
            else:
                content = message.content

            # Calculate how many tokens we can fit
            available_tokens = max_context_tokens - current_tokens

            if available_tokens > 50:  # Only truncate if we have meaningful space
                # Ensure content is a string for truncation
                content_str = str(content) if content is not None else ""
                truncated_content = truncate_message_content(
                    content_str, available_tokens, model_name
                )

                # Create truncated message
                if isinstance(message, dict):
                    truncated_message: Union[Dict[str, Any], BaseMessage] = {
                        **message,
                        "content": truncated_content,
                    }
                else:
                    truncated_message = type(message)(
                        content=truncated_content,
                        **{k: v for k, v in message.__dict__.items() if k != "content"},
                    )

                truncated_messages.insert(0, truncated_message)
            break

    return truncated_messages


# ---------------------------------------------------------------------------
# Tavily Extract helper
# ---------------------------------------------------------------------------


async def _supplement_with_extract(
    search_results: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """Fetch raw page content for Tavily search results via the Extract API.

    This helper can be a significant driver of Tavily credit usage because it
    issues *Extract* requests for every unique URL returned by the initial
    search.  To give users tighter control over their credit spend we now make
    the *top‐k* limit configurable:

    • Set the environment variable ``TAVILY_EXTRACT_TOP_K`` to override the
      default (e.g. ``export TAVILY_EXTRACT_TOP_K=5``).
    • Pass an explicit ``top_k`` parameter when calling this function to take
      full control programmatically.

    Providing ``0`` (or any negative number) disables extraction entirely.

    Args:
        search_results: List of Tavily search-response dictionaries (one per
            query).
        top_k: Maximum number of URLs to extract.  If *None*, the value will
            be taken from the ``TAVILY_EXTRACT_TOP_K`` environment variable (or
            default to *5*).

    Returns:
        The same structure but with a ``raw_content`` key populated for the
        URLs that were processed via the Extract API (if any).
    """
    import os  # Local import to avoid cycles during static analysis

    # Resolve the effective *top_k* value (env var → default 5)
    if top_k is None:
        top_k = int(os.getenv("TAVILY_EXTRACT_TOP_K", "5"))

    # Skip extraction completely if disabled
    if top_k <= 0:
        return search_results

    # Flatten results to unique URLs first (preserve reference)
    url_to_result: dict[str, dict] = {}
    for response in search_results:
        results_list = safe_get(response, "results", [])
        for item in results_list:
            url = safe_get(item, "url")
            if url:
                url_to_result.setdefault(url, item)
            if len(url_to_result) >= top_k:
                break
        if len(url_to_result) >= top_k:
            break

    if not url_to_result:
        return search_results

    urls = list(url_to_result.keys())

    async with acquire_tavily_semaphore():
        # Record credit usage (advanced depth assumed for now)
        record_tavily(
            "extract", "advanced", count=len(urls), urls=urls[:3]
        )  # log first few URLs for brevity
        extract_resp = await tavily_extract_tool.ainvoke(
            {"urls": urls, "extract_depth": "advanced"}
        )

    for result in safe_get(extract_resp, "results", []):
        url = safe_get(result, "url")
        if url and url in url_to_result:
            url_to_result[url]["raw_content"] = safe_get(result, "raw_content")

    return search_results
