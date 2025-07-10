"""Example implementation of a unified search interface for all providers.

This demonstrates how to create a consistent interface across different search providers.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from open_deep_research.tavily_tools import tavily_search_tool

from .dependency_manager import SearchProvider, validate_provider_or_raise


@dataclass
class SearchResult:
    """Standardized search result format."""

    title: str
    url: str
    content: str
    score: float
    raw_content: str | None = None
    provider: str | None = None


class SearchProviderBase(ABC):
    """Base class for all search providers."""

    def __init__(self, provider: SearchProvider):
        """Initialize SearchProviderBase with a specific provider.

        Args:
            provider: The search provider type to use
        """
        self.provider = provider
        validate_provider_or_raise(provider)

    @abstractmethod
    async def search(self, queries: List[str], **kwargs) -> List[SearchResult]:
        """Execute search queries and return standardized results."""
        pass

    @abstractmethod
    def get_required_params(self) -> List[str]:
        """Get list of required parameters for this provider."""
        pass

    @abstractmethod
    def get_optional_params(self) -> Dict[str, Any]:
        """Get optional parameters with their defaults."""
        pass

    def validate_params(self, **kwargs) -> Dict[str, Any]:
        """Validate and filter parameters for this provider."""
        required = set(self.get_required_params())
        optional = self.get_optional_params()

        # Check required params
        missing = required - set(kwargs.keys())
        if missing:
            raise ValueError(
                f"Missing required parameters for {self.provider.value}: {missing}"
            )

        # Filter to only valid params
        valid_params = {}
        all_valid = required | set(optional.keys())

        for key, value in kwargs.items():
            if key in all_valid:
                valid_params[key] = value

        # Add defaults for missing optional params
        for key, default in optional.items():
            if key not in valid_params:
                valid_params[key] = default

        return valid_params


class TavilySearchProvider(SearchProviderBase):
    """Tavily search provider implementation."""

    def __init__(self):
        """Initialize TavilySearchProvider."""
        super().__init__(SearchProvider.TAVILY)

    async def search(self, queries: List[str], **kwargs) -> List[SearchResult]:
        """Execute Tavily search."""
        params = self.validate_params(**kwargs)
        results = []

        # Execute search
        # Using the official TavilySearch tool
        response = await tavily_search_tool.ainvoke(
            {
                "query": "What is the capital of Finland?",
                "max_results": 5,
                "search_depth": "advanced",
                "include_raw_content": True,
            }
        )
        search_responses = [response]

        # Convert to standardized format
        for response in search_responses:
            for result in response.get("results", []):
                results.append(
                    SearchResult(
                        title=result["title"],
                        url=result["url"],
                        content=result["content"],
                        score=result.get("score", 0.0),
                        raw_content=result.get("raw_content"),
                        provider=self.provider.value,
                    )
                )

        return results

    def get_required_params(self) -> List[str]:
        """Get list of required parameters for Tavily search provider."""
        return []  # No required params beyond queries

    def get_optional_params(self) -> Dict[str, Any]:
        """Get optional parameters with their defaults for Tavily search provider."""
        return {"max_results": 5, "topic": "general", "include_raw_content": True}


class DuckDuckGoSearchProvider(SearchProviderBase):
    """DuckDuckGo search provider implementation."""

    def __init__(self):
        """Initialize DuckDuckGoSearchProvider."""
        super().__init__(SearchProvider.DUCKDUCKGO)

    async def search(self, queries: List[str], **kwargs) -> List[SearchResult]:
        """Execute DuckDuckGo search."""
        from open_deep_research.utils import duckduckgo_search

        # DuckDuckGo doesn't need parameter validation
        results = []

        # Execute search
        await duckduckgo_search.ainvoke({"search_queries": queries})

        # Parse the formatted string response
        # In a real implementation, we would modify duckduckgo_search to return structured data
        # For now, we'll create mock results
        for i, query in enumerate(queries):
            results.append(
                SearchResult(
                    title=f"DuckDuckGo result for: {query}",
                    url=f"https://example.com/{i}",
                    content=f"Search results for {query}",
                    score=1.0,
                    raw_content=None,
                    provider=self.provider.value,
                )
            )

        return results

    def get_required_params(self) -> List[str]:
        """Get list of required parameters for DuckDuckGo search provider."""
        return []

    def get_optional_params(self) -> Dict[str, Any]:
        """Get optional parameters with their defaults for DuckDuckGo search provider."""
        return {}


class UnifiedSearchInterface:
    """Unified interface for all search providers."""

    def __init__(self):
        """Initialize UnifiedSearchInterface with available providers."""
        self.providers = {
            SearchProvider.TAVILY: TavilySearchProvider,
            SearchProvider.DUCKDUCKGO: DuckDuckGoSearchProvider,
            # Add more providers as they're implemented
        }

    async def search(
        self, provider: SearchProvider, queries: List[str], **kwargs
    ) -> List[SearchResult]:
        """Execute search using the specified provider."""
        if provider not in self.providers:
            raise NotImplementedError(
                f"Provider {provider.value} not yet implemented in unified interface"
            )

        provider_class = self.providers[provider]
        provider_instance = provider_class()

        return await provider_instance.search(queries, **kwargs)

    def get_provider_info(self, provider: SearchProvider) -> Dict[str, Any]:
        """Get information about a provider's parameters."""
        if provider not in self.providers:
            raise NotImplementedError(f"Provider {provider.value} not yet implemented")

        provider_class = self.providers[provider]
        provider_instance = provider_class()

        return {
            "required_params": provider_instance.get_required_params(),
            "optional_params": provider_instance.get_optional_params(),
        }


# Example usage
async def example_unified_search():
    """Demonstrate usage of the unified search interface."""
    unified_search = UnifiedSearchInterface()

    # Search with Tavily
    try:
        tavily_results = await unified_search.search(
            SearchProvider.TAVILY,
            ["quantum computing applications", "machine learning trends"],
            max_results=3,
            topic="general",
        )

        for result in tavily_results:
            pass
    except Exception:
        pass

    # Search with DuckDuckGo
    try:
        ddg_results = await unified_search.search(
            SearchProvider.DUCKDUCKGO, ["artificial intelligence news"]
        )

        for result in ddg_results:
            pass
    except Exception:
        pass


if __name__ == "__main__":
    # Run example
    asyncio.run(example_unified_search())
