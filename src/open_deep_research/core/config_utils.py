"""Configuration utilities for extracting and processing config values."""

from typing import Any, Dict, Type, TypeVar, Union, Protocol

from langchain_core.runnables import RunnableConfig

# Type variable for configuration classes
ConfigT = TypeVar('ConfigT')


class ConfigurationProtocol(Protocol):
    """Protocol for configuration classes that can be created from RunnableConfig."""
    
    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "ConfigurationProtocol":
        """Create configuration from RunnableConfig."""
        ...


def get_config_value(value: Any) -> Any:
    """Handle string, dict, and enum cases of configuration values."""
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_search_params(search_api: str, search_api_config: Dict[str, Any] | None) -> Dict[str, Any]:
    """Filter the search_api_config dictionary to include only parameters accepted by the specified search API.

    Args:
        search_api (str): The search API identifier (e.g., "exa", "tavily").
        search_api_config (Optional[Dict[str, Any]]): The configuration dictionary for the search API.

    Returns:
        Dict[str, Any]: A dictionary of parameters to pass to the search function.
    """
    # Define accepted parameters for each search API
    SEARCH_API_PARAMS = {
        "exa": ["max_characters", "num_results", "include_domains", "exclude_domains", "subpages"],
        "tavily": ["max_results", "topic", "search_depth", "chunks_per_source", "time_range", "include_domains", "exclude_domains", "include_images"],
        "perplexity": [],  # Perplexity accepts no additional parameters
        "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
        "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],
        "linkup": ["depth"],
        "googlesearch": ["max_results"],
    }

    # Get the list of accepted parameters for the given search API
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # If no config provided, return an empty dict
    if not search_api_config:
        return {}

    # Filter the config to only include accepted parameters
    return {k: v for k, v in search_api_config.items() if k in accepted_params}


def extract_configuration(
    config: RunnableConfig,
    config_class: Type[ConfigurationProtocol]
) -> ConfigurationProtocol:
    """Extract configuration from RunnableConfig using the specified class.
    
    This pattern appears at the start of almost every node function.
    
    Args:
        config: The RunnableConfig instance
        config_class: The configuration class to instantiate
        
    Returns:
        Instance of the configuration class
    """
    return config_class.from_runnable_config(config)


def get_search_api_params(
    configurable: Any,
) -> Dict[str, Any]:
    """Extract search API parameters from configuration.
    
    Common pattern for getting search API and its parameters.
    
    Args:
        configurable: Configuration object with search_api and search_api_config
        
    Returns:
        Dictionary of search parameters
    """
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}
    return get_search_params(search_api, search_api_config)


def get_model_config_values(
    configurable: Any,
    role: str,
) -> tuple[str, str, Dict[str, Any] | None]:
    """Extract model configuration values for a specific role.
    
    Common pattern for getting provider, model name, and kwargs.
    
    Args:
        configurable: Configuration object
        role: Model role (e.g., "writer", "planner")
        
    Returns:
        Tuple of (provider, model_name, model_kwargs)
    """
    provider_attr = f"{role}_provider"
    model_attr = f"{role}_model"
    kwargs_attr = f"{role}_model_kwargs"
    
    provider = get_config_value(getattr(configurable, provider_attr))
    model_name = get_config_value(getattr(configurable, model_attr))
    model_kwargs = get_config_value(getattr(configurable, kwargs_attr, None) or {})
    
    return provider, model_name, model_kwargs 