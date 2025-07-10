from __future__ import annotations

"""Utility for on-demand import of heavy workflow nodes.

The real LazyNodeLoader will be fleshed out during the unified-graph
migration.  For now we expose the public interface so downstream code
can depend on it without introducing hard runtime imports.
"""

from functools import lru_cache
import importlib
from typing import Callable, Dict

__all__ = ["LazyLoadError", "LazyNodeLoader"]


class LazyLoadError(ImportError):
    """Wrap import errors so we preserve the original traceback."""

    def __init__(self, original_error: Exception, loader_context: str):
        self.original_error = original_error
        self.loader_context = loader_context
        message = f"{loader_context}: {original_error}"
        super().__init__(message)


class LazyNodeLoader:
    """Minimal stub that will grow as part of the migration.

    For now only supports an empty mapping but keeps the public contract
    (``get_node``) intact so early adopters can experiment.
    """

    _node_modules: Dict[str, str] = {}

    @classmethod
    @lru_cache(maxsize=None)
    def get_node(cls, node_name: str) -> Callable:  # pragma: no cover
        """Return the callable for *node_name*.

        At this stage the function is a stub that always raises.  It will
        be populated with real mappings once the unified planner lands.
        """

        try:
            module_path = cls._node_modules[node_name]
            module = importlib.import_module(module_path)
            return getattr(module, node_name)
        except KeyError as e:
            raise LazyLoadError(e, f"Unknown node '{node_name}'") from e
        except Exception as e:
            raise LazyLoadError(e, f"Failed to load node '{node_name}'") from e
