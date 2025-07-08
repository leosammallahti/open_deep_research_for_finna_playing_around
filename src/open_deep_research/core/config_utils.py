"""Configuration utilities for extracting and processing config values."""

from typing import Any, Dict, Type, TypeVar

from langchain_core.runnables import RunnableConfig

from open_deep_research.utils import get_config_value, get_search_params

# Type variable for configuration classes
ConfigT = TypeVar('ConfigT')


def extract_configuration(
    config: RunnableConfig,
    config_class: Type[ConfigT]
) -> ConfigT:
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