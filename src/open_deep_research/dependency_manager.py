"""Dependency management for optional search providers.

This module handles optional imports and provides clear guidance when dependencies are missing.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Set

from open_deep_research.core.logging_utils import get_logger

logger = get_logger(__name__)


# Enhanced error classes for better error handling
class DependencyError(Exception):
    """Base exception for dependency-related errors."""

    pass


class ProviderNotInstalledError(DependencyError):
    """Raised when a search provider is not installed."""

    def __init__(self, provider: str, install_command: str, description: str):
        """Initialize ProviderNotInstalledError with installation details.

        Args:
            provider: Name of the search provider
            install_command: Command to install the provider
            description: Description of the provider's functionality
        """
        self.provider = provider
        self.install_command = install_command
        self.description = description
        super().__init__(
            f"❌ {provider} is not available. "
            f"To use this search provider, install it with:\n"
            f"  {install_command}\n"
            f"Description: {description}"
        )


class ProviderConfigurationError(DependencyError):
    """Raised when a provider is installed but not properly configured."""

    def __init__(self, provider: str, missing_config: List[str], help_text: str = ""):
        """Initialize ProviderConfigurationError with configuration details.

        Args:
            provider: Name of the search provider
            missing_config: List of missing configuration parameters
            help_text: Additional help text for configuration
        """
        self.provider = provider
        self.missing_config = missing_config
        super().__init__(
            f"⚠️ {provider} is installed but not configured properly.\n"
            f"Missing configuration: {', '.join(missing_config)}\n"
            f"{help_text}"
        )


class SearchProvider(str, Enum):
    """Enumeration of supported search providers."""

    DUCKDUCKGO = "duckduckgo"
    TAVILY = "tavily"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    EXA = "exa"
    LINKUP = "linkup"
    GOOGLE = "googlesearch"
    AZURE = "azureaisearch"
    PERPLEXITY = "perplexity"


@dataclass
class HealthStatus:
    """Health status of a provider."""

    healthy: bool
    error: str | None = None
    details: str | None = None
    fix_command: str | None = None


@dataclass
class DependencyInfo:
    """Information about a dependency."""

    name: str
    import_name: str
    pip_install: str
    description: str
    is_available: bool = False
    import_error: str | None = None
    required_env_vars: List[str] | None = None  # New field for API key requirements


class DependencyManager:
    """Manages optional dependencies for search providers."""

    def __init__(self) -> None:
        """Initialize DependencyManager and check provider availability."""
        self._available_providers: Set[SearchProvider] = set()
        self._dependency_info: Dict[SearchProvider, DependencyInfo] = {}
        self._imports_cache: Dict[str, Any] = {}
        self._initialize_dependencies()

    def _initialize_dependencies(self) -> None:
        """Initialize dependency information and check availability."""
        # Define dependency information
        dependencies = {
            SearchProvider.DUCKDUCKGO: DependencyInfo(
                name="DuckDuckGo Search",
                import_name="duckduckgo_search",
                pip_install="duckduckgo-search>=3.0.0",
                description="Privacy-focused web search",
                required_env_vars=[],  # No API key required
            ),
            SearchProvider.TAVILY: DependencyInfo(
                name="Tavily",
                import_name="tavily",
                pip_install="tavily-python>=0.5.0",
                description="Research-focused web search API",
                required_env_vars=["TAVILY_API_KEY"],
            ),
            SearchProvider.ARXIV: DependencyInfo(
                name="ArXiv",
                import_name="arxiv",
                pip_install="arxiv>=2.1.3",
                description="Academic paper search",
                required_env_vars=[],  # No API key required
            ),
            SearchProvider.PUBMED: DependencyInfo(
                name="PubMed",
                import_name="xmltodict",  # Main dependency for PubMed
                pip_install="xmltodict>=0.14.2",
                description="Medical literature search",
                required_env_vars=[],  # Optional email/API key
            ),
            SearchProvider.EXA: DependencyInfo(
                name="Exa",
                import_name="exa_py",
                pip_install="exa-py>=1.8.8",
                description="Semantic search engine",
                required_env_vars=["EXA_API_KEY"],
            ),
            SearchProvider.LINKUP: DependencyInfo(
                name="Linkup",
                import_name="linkup_sdk",
                pip_install="linkup-sdk>=0.2.3",
                description="Real-time web search",
                required_env_vars=["LINKUP_API_KEY"],
            ),
            SearchProvider.GOOGLE: DependencyInfo(
                name="Google Search",
                import_name="googlesearch",
                pip_install="googlesearch-python>=1.2.3",
                description="Google web search",
                required_env_vars=[],  # No API key for scraping version
            ),
            SearchProvider.AZURE: DependencyInfo(
                name="Azure AI Search",
                import_name="azure.search.documents",
                pip_install="azure-search-documents>=11.5.2 azure-identity>=1.21.0",
                description="Enterprise search service",
                required_env_vars=[
                    "AZURE_AI_SEARCH_ENDPOINT",
                    "AZURE_AI_SEARCH_INDEX_NAME",
                    "AZURE_AI_SEARCH_API_KEY",
                ],
            ),
            SearchProvider.PERPLEXITY: DependencyInfo(
                name="Perplexity",
                import_name="requests",  # Uses requests which is always available
                pip_install="requests>=2.32.3",
                description="AI-powered search (uses requests)",
                required_env_vars=["PERPLEXITY_API_KEY"],
            ),
        }

        # Check availability of each dependency
        for provider, dep_info in dependencies.items():
            try:
                self._try_import(dep_info.import_name)
                dep_info.is_available = True
                self._available_providers.add(provider)
                logger.debug(f"✓ {dep_info.name} is available")
            except ImportError as e:
                dep_info.is_available = False
                dep_info.import_error = str(e)
                logger.debug(f"✗ {dep_info.name} is not available: {e}")

            self._dependency_info[provider] = dep_info

    def _try_import(self, import_name: str) -> Any:
        """Try to import a module and cache the result."""
        if import_name in self._imports_cache:
            return self._imports_cache[import_name]

        try:
            if "." in import_name:
                # Handle nested imports like "azure.search.documents"
                module_parts = import_name.split(".")
                module = __import__(import_name)
                for part in module_parts[1:]:
                    module = getattr(module, part)
            else:
                module = __import__(import_name)

            self._imports_cache[import_name] = module
            return module
        except ImportError:
            self._imports_cache[import_name] = None
            raise

    def is_provider_available(self, provider: SearchProvider) -> bool:
        """Check if a search provider is available."""
        return provider in self._available_providers

    def get_available_providers(self) -> List[SearchProvider]:
        """Get list of available search providers."""
        return list(self._available_providers)

    def get_unavailable_providers(self) -> List[SearchProvider]:
        """Get list of unavailable search providers."""
        return [p for p in SearchProvider if p not in self._available_providers]

    def get_provider_info(self, provider: SearchProvider) -> DependencyInfo | None:
        """Get information about a specific provider."""
        return self._dependency_info.get(provider)

    def get_installation_command(self, provider: SearchProvider) -> str:
        """Get pip install command for a provider."""
        info = self.get_provider_info(provider)
        if info:
            return f"pip install {info.pip_install}"
        return f"pip install {provider.value}"

    def get_installation_commands_for_multiple(
        self, providers: List[SearchProvider]
    ) -> str:
        """Get pip install command for multiple providers."""
        packages = []
        for provider in providers:
            info = self.get_provider_info(provider)
            if info:
                packages.extend(info.pip_install.split())

        # Remove duplicates while preserving order
        unique_packages = []
        seen = set()
        for package in packages:
            if package not in seen:
                unique_packages.append(package)
                seen.add(package)

        return f"pip install {' '.join(unique_packages)}"

    def check_provider_health(self, provider: SearchProvider) -> HealthStatus:
        """Check if a provider is properly configured and ready to use."""
        info = self.get_provider_info(provider)
        if not info:
            return HealthStatus(False, "Provider not found")

        if not self.is_provider_available(provider):
            return HealthStatus(
                False,
                "Provider not installed",
                fix_command=self.get_installation_command(provider),
            )

        # Check required environment variables
        if info.required_env_vars:
            missing_vars = [var for var in info.required_env_vars if not os.getenv(var)]
            if missing_vars:
                return HealthStatus(
                    False, "Missing API keys", f"Missing: {', '.join(missing_vars)}"
                )

        return HealthStatus(True)

    def validate_provider_or_raise(self, provider: SearchProvider) -> None:
        """Validate that a provider is available and configured or raise a helpful error."""
        if not self.is_provider_available(provider):
            info = self.get_provider_info(provider)
            if info:
                raise ProviderNotInstalledError(
                    info.name, self.get_installation_command(provider), info.description
                )
            else:
                raise ImportError(f"❌ Search provider '{provider}' is not supported")

        # Also check configuration
        health = self.check_provider_health(provider)
        if not health.healthy:
            if health.error == "Missing API keys":
                info = self.get_provider_info(provider)
                if info and health.details:
                    # Extract missing vars from details string
                    missing_vars = health.details.replace("Missing: ", "").split(", ")
                    raise ProviderConfigurationError(
                        info.name,
                        missing_vars,
                        "Set these environment variables in your .env file",
                    )

    def get_status_report(self) -> str:
        """Get a nicely formatted status report of all providers."""
        report_lines = ["\n--- Search Provider Status ---"]

        for provider in SearchProvider:
            dep_info = self.get_provider_info(provider)
            if dep_info:
                if self.is_provider_available(provider):
                    health = self.check_provider_health(provider)
                    if health.healthy:
                        report_lines.append(
                            f"✓ {dep_info.name}: Installed and configured"
                        )
                    else:
                        error_msg = (
                            health.details or health.error or "Configuration issue"
                        )
                        report_lines.append(
                            f"⚠️ {dep_info.name}: Installed but not configured ({error_msg})"
                        )
                else:
                    report_lines.append(f"✗ {dep_info.name}: Not installed")
                    if dep_info.import_error:
                        report_lines.append(f"  - Error: {dep_info.import_error}")
                    report_lines.append(
                        f"  - To install: {self.get_installation_command(provider)}"
                    )

        report_lines.append("------------------------------\n")
        return "\n".join(report_lines)

    def get_safe_import(self, import_name: str) -> Any | None:
        """Safely import a module, returning None if not available."""
        try:
            return self._try_import(import_name)
        except ImportError:
            return None


# Global instance
dependency_manager = DependencyManager()


# Convenience functions
def is_provider_available(provider: SearchProvider) -> bool:
    """Check if a search provider is available."""
    return dependency_manager.is_provider_available(provider)


def get_available_providers() -> List[SearchProvider]:
    """Get list of available search providers."""
    return dependency_manager.get_available_providers()


def validate_provider_or_raise(provider: SearchProvider) -> None:
    """Validate that a provider is available and configured or raise a helpful error."""
    dependency_manager.validate_provider_or_raise(provider)


def get_status_report() -> str:
    """Generate a status report of all dependencies."""
    return dependency_manager.get_status_report()


def get_safe_import(import_name: str) -> Any | None:
    """Safely import a module, returning None if not available."""
    return dependency_manager.get_safe_import(import_name)
