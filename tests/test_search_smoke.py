
import pytest

from open_deep_research import search as search_pkg
from open_deep_research.dependency_manager import SearchProvider, dependency_manager


def _provider_to_attr(provider: SearchProvider) -> str:
    mapping = {
        SearchProvider.DUCKDUCKGO: "duckduckgo_search",
        SearchProvider.EXA: "exa_search",
        SearchProvider.TAVILY: "tavily",  # handled by legacy dispatcher alias
        SearchProvider.ARXIV: "arxiv_search_async",
        SearchProvider.PUBMED: "pubmed_search_async",
        SearchProvider.AZURE: "azureaisearch_search_async",
        SearchProvider.GOOGLE: "google_search_async",
        SearchProvider.LINKUP: "linkup_search",
        SearchProvider.PERPLEXITY: "perplexity_search_async",
    }
    return mapping.get(provider, "")


@pytest.mark.parametrize("provider", list(SearchProvider))
def test_search_provider_import(provider: SearchProvider):
    """Ensure each provider function can be imported (or skipped when deps missing)."""

    attr_name = _provider_to_attr(provider)
    if not attr_name:
        pytest.skip(f"No attr mapping for {provider}")

    if not dependency_manager.is_provider_available(provider):
        pytest.skip(f"Dependencies for {provider} not available")

    func = getattr(search_pkg, attr_name, None)
    assert callable(func) 