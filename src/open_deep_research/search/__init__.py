from __future__ import annotations

"""Search provider wrapper package.

This thin layer re-exports the provider-specific async search helpers that
currently live in :pymod:`open_deep_research.utils`.  It allows gradual
migration away from the monolithic *utils.py* without breaking existing call
sites.  Downstream modules should import from this package instead of
``open_deep_research.utils``.
"""

from typing import Any, Dict, List  # noqa: F401 – kept for gradual migration

# Still re-export most providers from the legacy utils module until their
# dedicated files are extracted.  Newly migrated providers are imported from
# their own submodules.
from open_deep_research.utils import (  # type: ignore  # noqa: E402
    arxiv_search_async as arxiv_search_async,
)
from open_deep_research.utils import (
    azureaisearch_search_async as azureaisearch_search_async,
)
from open_deep_research.utils import (
    # placeholders for providers now wrapped below
    google_search_async as _legacy_google_search_async,
)
from open_deep_research.utils import (
    linkup_search as _legacy_linkup_search,
)
from open_deep_research.utils import (
    pubmed_search_async as _legacy_pubmed_search_async,
)

from .arxiv import arxiv_search_async  # noqa: E402 (overrides legacy)
from .azureaisearch import azureaisearch_search_async  # noqa: E402
from .base import deduplicate_and_format_sources  # noqa: E402
from .duckduckgo import duckduckgo_search  # noqa: E402

# Migrated provider helpers
from .exa import exa_search  # noqa: E402
from .google import google_search_async  # noqa: E402
from .linkup import linkup_search  # noqa: E402
from .perplexity import perplexity_search_async  # noqa: E402
from .pubmed import pubmed_search_async  # noqa: E402

__all__ = [
    "arxiv_search_async",
    "azureaisearch_search_async",
    "deduplicate_and_format_sources",
    "duckduckgo_search",
    "exa_search",
    "google_search_async",
    "linkup_search",
    "perplexity_search_async",
    "pubmed_search_async",
] 