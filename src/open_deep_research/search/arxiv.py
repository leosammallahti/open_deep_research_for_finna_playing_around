from __future__ import annotations

"""ArXiv search provider (wrapper).

The full implementation still lives in *open_deep_research.utils*.  This stub
re-exports it so that call sites can depend on the new
``open_deep_research.search`` namespace immediately.  We will migrate the code
properly in a follow-up.
"""

from open_deep_research.utils import arxiv_search_async as arxiv_search_async

__all__ = ["arxiv_search_async"] 