"""Centralised configuration via Pydantic *BaseSettings*.

This module loads environment variables (and a local ``.env`` file if present)
exactly once at import-time, then makes the validated settings available as
``settings``.  For legacy compatibility we also inject the loaded variables
into ``os.environ`` so that existing code using ``os.getenv`` continues to
work seamlessly.

Examples:
--------
>>> from open_deep_research.core.settings import settings
>>> settings.tavily_api_key
'sk-…'
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Project-wide environment configuration."""

    # API keys --------------------------------------------------------------
    tavily_api_key: str | None = None
    exa_api_key: str | None = None
    perplexity_api_key: str | None = None

    # Azure AI Search keys
    azure_ai_search_endpoint: str | None = None
    azure_ai_search_index_name: str | None = None
    azure_ai_search_api_key: str | None = None

    # Other providers
    linkup_api_key: str | None = None

    # General ----------------------------------------------------------------
    log_level: str | int = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Convenience helpers ----------------------------------------------------
    @property
    def ci(self) -> bool:
        """Whether we are running in a CI environment."""
        return bool(os.environ.get("CI"))


@lru_cache
def _load_settings() -> AppSettings:
    """Load settings once, cache for subsequent imports."""
    s = AppSettings()  # reads env + .env automatically

    # Bridge to legacy ``os.getenv`` usage across the codebase. We *only* set a
    # variable if it is currently missing to avoid overriding explicit envs.
    for key, value in s.model_dump(exclude_none=True).items():
        env_key = key.upper()
        if env_key not in os.environ and value is not None:
            os.environ[env_key] = str(value)

    return s


# Public, singleton-esque settings object
settings: AppSettings = _load_settings() 