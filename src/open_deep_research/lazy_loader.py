# -*- coding: utf-8 -*-
"""On-demand import helper for heavyweight workflow nodes.

This utility lets downstream code reference nodes *by name* without paying
import cost until the node is actually needed.  During the ongoing unified
workflow migration the mapping remains empty – it will be filled once nodes
are moved into standalone modules.
"""

from __future__ import annotations

import importlib
from functools import cache
from typing import Callable, Dict

__all__ = ["LazyLoadError", "LazyNodeLoader"]


class LazyLoadError(ImportError):
    """Wrap and preserve the original traceback when a lazy import fails."""

    def __init__(self, original_error: Exception, loader_context: str):
        """Capture *original_error* with contextual information."""
        self.original_error = original_error
        self.loader_context = loader_context
        super().__init__(f"{loader_context}: {original_error}")


class LazyNodeLoader:  # noqa: D101 – minimal stub
    """Minimal registry for lazily-imported workflow nodes.

    Public API mirrors what future code will expect (`get_node`).  Only an
    empty mapping is provided for now so unit tests can reference it without
    triggering heavy imports.
    """

    _node_modules: Dict[str, str] = {}

    @classmethod
    @cache
    def get_node(cls, node_name: str) -> Callable:  # pragma: no cover
        """Return the callable for *node_name* or raise *LazyLoadError*."""
        try:
            module_path = cls._node_modules[node_name]
            module = importlib.import_module(module_path)
            return getattr(module, node_name)
        except KeyError as exc:
            raise LazyLoadError(exc, f"Unknown node '{node_name}'") from exc
        except Exception as exc:  # noqa: BLE001 – propagate as LazyLoadError
            raise LazyLoadError(exc, f"Failed to load node '{node_name}'") from exc 