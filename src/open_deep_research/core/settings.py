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

# Pydantic v2 exposes field/model validators in the main package, not via
# pydantic_settings.  We import them separately.
from pydantic import field_validator, model_validator
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

    # Workflow defaults ------------------------------------------------------
    # These replicate the default values found in WorkflowConfiguration so they can
    # be centrally overridden via environment variables or a .env file.

    # Search / credits
    search_api: str | None = "tavily"
    search_budget: int = 100

    # LLMS & models
    planner_provider: str | None = "together"
    planner_model: str | None = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    writer_provider: str | None = "together"
    writer_model: str | None = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    reflection_model_provider: str | None = None
    reflection_model: str | None = None

    # Search iteration parameters
    number_of_queries: int = 1
    max_search_depth: int = 2

    # Behaviour toggles
    include_source_str: bool = True
    clarify_with_user: bool = False

    # Multi-agent specifics ---------------------------------------------------
    supervisor_model: str | None = None
    researcher_model: str | None = None

    # MCP server integration
    mcp_server_config: str | None = None  # JSON / YAML string or path
    mcp_prompt: str | None = None
    mcp_tools_to_include: str | None = None  # CSV list

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

    # ------------------------------------------------------------------
    # Generic validators
    # ------------------------------------------------------------------

    # 1) Strip surrounding whitespace from *all* string fields so that accidental
    #    newlines or spaces in .env files do not corrupt HTTP headers.
    @classmethod
    @field_validator("*", mode="before")
    def _strip_whitespace(cls, v):  # noqa: D401 – Pydantic validator
        if isinstance(v, str):
            return v.strip()
        return v

    # 2) Extra sanity-check: API keys must not contain whitespace.  We perform
    #    this as a model-level validation so we can report *all* offending keys
    #    at once instead of failing fast on the first.
    @model_validator(mode="after")
    def _validate_api_keys(self):  # noqa: D401
        keys_with_whitespace: list[str] = []
        for field_name in (
            "tavily_api_key",
            "exa_api_key",
            "perplexity_api_key",
            "azure_ai_search_api_key",
            "linkup_api_key",
        ):
            key_val = getattr(self, field_name)
            if key_val and any(ch.isspace() for ch in key_val):
                keys_with_whitespace.append(field_name)

        if keys_with_whitespace:
            joined = ", ".join(keys_with_whitespace)
            raise ValueError(
                f"Whitespace detected in the following API keys: {joined}. "
                "Remove spaces/newlines or wrap the key without quotes in your .env."
            )

        return self

    # ------------------------------------------------------------------
    # Public helper
    # ------------------------------------------------------------------

    def validate_all(self) -> "AppSettings":
        """Return self after triggering full model validation.

        The settings object is validated at construction time already.  This
        helper exists so call-sites (e.g. Streamlit or CLI entry points) can
        call ``settings.validate_all()`` to make the intent explicit and to
        surface configuration errors before expensive network operations begin.
        """

        # ``model_validate`` will re-run validators; if it succeeds we simply
        # return ``self`` ref to allow fluent usage.
        type(self).model_validate(self.model_dump())
        return self


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
