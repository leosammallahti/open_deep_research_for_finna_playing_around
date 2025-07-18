"""Central registry describing capabilities of models for Open Deep Research.

Each entry maps a model identifier (provider:model_name) to its capabilities so that
UI and runtime code can reason about what role the model can play.

Capabilities:
- supports_tool_choice: can it handle with_structured_output / function-calling?
- recommended_roles: list[str] → ["planner", "writer", "summarizer", "reflection"]
- context_window: int – max tokens (approximate)
"""

from __future__ import annotations

from typing import Dict, List, Literal, TypedDict

ModelRole = Literal["planner", "writer", "summarizer", "reflection"]


class ModelCapabilities(TypedDict, total=False):
    """Type definition for model capabilities and metadata."""

    supports_tool_choice: bool
    recommended_roles: List[ModelRole]
    context_window: int


# NOTE: This is *not* an exhaustive list – extend as needed.
MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    # Anthropic
    "anthropic:claude-3-5-sonnet-20240620": {
        "supports_tool_choice": True,
        "recommended_roles": ["planner", "writer", "reflection"],
        "context_window": 200_000,
    },
    "anthropic:claude-3-haiku-20240307": {
        "supports_tool_choice": True,
        "recommended_roles": ["writer", "summarizer"],
        "context_window": 200_000,
    },
    # OpenAI
    "openai:gpt-4o": {
        "supports_tool_choice": True,
        "recommended_roles": ["planner", "writer", "reflection"],
        "context_window": 128_000,
    },
    "openai:gpt-4": {
        "supports_tool_choice": True,
        "recommended_roles": ["planner", "writer", "reflection"],
        "context_window": 128_000,
    },
    "openai:gpt-3.5-turbo": {
        "supports_tool_choice": True,
        "recommended_roles": ["writer", "summarizer"],
        "context_window": 16_000,
    },
    # Groq – OSS Llama 3
    "groq:llama-3.1-70b-versatile": {
        "supports_tool_choice": True,
        "recommended_roles": ["planner", "writer"],
        "context_window": 131_072,
    },
    # DeepSeek
    "deepseek:deepseek-reasoner": {
        "supports_tool_choice": False,  # Cannot handle tool_call / structured output
        "recommended_roles": ["planner", "reflection"],
        "context_window": 65_536,
    },
    "deepseek:deepseek-chat": {
        "supports_tool_choice": True,
        "recommended_roles": ["writer", "summarizer"],
        "context_window": 65_536,
    },
    # TogetherAI
    "together:meta-llama/Llama-3.3-70B-Instruct-Turbo": {
        "supports_tool_choice": True,
        "recommended_roles": ["planner", "writer", "reflection"],
        "context_window": 128_000,
    },
    "together:meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "supports_tool_choice": True,
        "recommended_roles": ["planner", "writer", "reflection"],
        "context_window": 524_288,
    },
    "together:meta-llama/Llama-4-Scout-17B-16E-Instruct-FP8": {
        "supports_tool_choice": True,
        "recommended_roles": ["writer", "summarizer"],
        "context_window": 300_000,
    },
    "together:Qwen/Qwen2.5-72B-Instruct-Turbo": {
        "supports_tool_choice": True,
        "recommended_roles": ["planner", "writer"],
        "context_window": 32768,
    },
    "together:deepseek-ai/DeepSeek-V3": {
        "supports_tool_choice": True,
        "recommended_roles": ["planner", "reflection"],
        "context_window": 163839,
    },
}


class ModelCombo(TypedDict):
    """Type definition for predefined model combinations."""

    display_name: str
    planner: str
    writer: str
    summarizer: str | None  # Falls back to writer if None


PREDEFINED_COMBOS: Dict[str, ModelCombo] = {
    "together_ai": {
        "display_name": "Together AI (Llama 3.3 70B + Qwen 2.5)",
        "planner": "together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "writer": "together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "summarizer": "together:Qwen/Qwen2.5-72B-Instruct-Turbo",
    },
    "balanced": {
        "display_name": "Balanced (Claude 3.5 Sonnet)",
        "planner": "anthropic:claude-3-5-sonnet-20240620",
        "writer": "anthropic:claude-3-5-sonnet-20240620",
        "summarizer": "anthropic:claude-3-haiku-20240307",
    },
    "max_reasoning": {
        "display_name": "Max Reasoning (DeepSeek + Claude)",
        "planner": "deepseek:deepseek-reasoner",
        "writer": "anthropic:claude-3-5-sonnet-20240620",
        "summarizer": "anthropic:claude-3-haiku-20240307",
    },
    "gpt_4o": {
        "display_name": "GPT-4o (OpenAI)",
        "planner": "openai:gpt-4o",
        "writer": "openai:gpt-4o",
        "summarizer": "openai:gpt-3.5-turbo",
    },
    "all_opensource": {
        "display_name": "All Open-Source (Llama 3)",
        "planner": "groq:llama-3.1-70b-versatile",
        "writer": "groq:llama-3.1-70b-versatile",
        "summarizer": None,
    },
    "budget": {
        "display_name": "Budget (GPT-3.5 Turbo)",
        "planner": "openai:gpt-3.5-turbo",
        "writer": "openai:gpt-3.5-turbo",
        "summarizer": None,
    },
}


def get_supported_models(
    role: ModelRole | None = None, *, require_tool_choice: bool | None = None
) -> List[str]:
    """Return list of model identifiers filtered by role and/or tool-choice capability."""
    models = []
    for model_id, caps in MODEL_CAPABILITIES.items():
        if role and role not in caps.get("recommended_roles", []):
            continue
        if (
            require_tool_choice is not None
            and caps.get("supports_tool_choice", False) != require_tool_choice
        ):
            continue
        models.append(model_id)
    return models


def supports_tool_choice(model_id: str) -> bool:
    """Check if a model supports tool choice/function calling.

    Args:
        model_id: The model identifier (e.g., "openai:gpt-4o")

    Returns:
        True if the model supports tool choice, False otherwise
    """
    return MODEL_CAPABILITIES.get(model_id, {}).get("supports_tool_choice", False)


def get_available_model_combos() -> Dict[str, ModelCombo]:
    """Return only model combinations where all required API keys are available."""
    from open_deep_research.utils import is_api_key_available

    def get_provider(model_str: str) -> str:
        return model_str.split(":")[0]

    def combo_is_available(combo: ModelCombo) -> bool:
        required_providers = [
            get_provider(combo["planner"]),
            get_provider(combo["writer"]),
        ]
        if combo["summarizer"]:
            required_providers.append(get_provider(combo["summarizer"]))

        return all(is_api_key_available(provider) for provider in required_providers)

    return {
        combo_id: combo
        for combo_id, combo in PREDEFINED_COMBOS.items()
        if combo_is_available(combo)
    }


def get_available_models(
    role: ModelRole | None = None, *, require_tool_choice: bool | None = None
) -> List[str]:
    """Return list of model identifiers that have available API keys, filtered by role and/or tool-choice capability."""
    from open_deep_research.utils import is_api_key_available

    models = []
    for model_id, caps in MODEL_CAPABILITIES.items():
        # Check if API key is available
        provider = model_id.split(":")[0]
        if not is_api_key_available(provider):
            continue

        if role and role not in caps.get("recommended_roles", []):
            continue
        if (
            require_tool_choice is not None
            and caps.get("supports_tool_choice", False) != require_tool_choice
        ):
            continue
        models.append(model_id)
    return models
