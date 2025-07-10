"""Message utilities for handling different message formats consistently.

This module provides a unified interface for working with messages regardless
of whether they are dictionaries or BaseMessage objects.
"""

from typing import Any, Dict, List, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from open_deep_research.core.logging_utils import get_logger

logger = get_logger(__name__)


def get_message_content(message: Union[Dict[str, Any], BaseMessage]) -> str:
    """Safely extract content from a message regardless of format.

    Args:
        message: Either a dict with 'content' key or a BaseMessage object

    Returns:
        The message content as a string

    Raises:
        TypeError: If message format is not supported
    """
    if isinstance(message, dict):
        return message.get("content", "")
    elif hasattr(message, "content"):
        content = message.content
        # Handle cases where content might be a complex type
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # For multimodal messages, join text parts
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
            return " ".join(text_parts)
        else:
            return str(content)
    else:
        raise TypeError(
            f"Unsupported message type: {type(message)}. Expected dict or BaseMessage."
        )


def get_message_role(message: Union[Dict[str, Any], BaseMessage]) -> str:
    """Safely extract role from a message regardless of format.

    Args:
        message: Either a dict with 'role' key or a BaseMessage object

    Returns:
        The message role as a string
    """
    if isinstance(message, dict):
        return message.get("role", "user")
    elif isinstance(message, HumanMessage):
        return "user"
    elif isinstance(message, SystemMessage):
        return "system"
    elif isinstance(message, AIMessage):
        return "assistant"
    elif hasattr(message, "type"):
        return message.type
    else:
        return "user"  # Default fallback


def convert_to_dict(message: Union[Dict[str, Any], BaseMessage]) -> Dict[str, Any]:
    """Convert a message to dictionary format.

    Args:
        message: Either a dict or a BaseMessage object

    Returns:
        Dictionary with 'role' and 'content' keys
    """
    if isinstance(message, dict):
        # Ensure it has the required keys
        return {
            "role": message.get("role", "user"),
            "content": message.get("content", ""),
        }
    else:
        return {
            "role": get_message_role(message),
            "content": get_message_content(message),
        }


def convert_to_base_message(message: Union[Dict[str, Any], BaseMessage]) -> BaseMessage:
    """Convert a message to BaseMessage format.

    Args:
        message: Either a dict or a BaseMessage object

    Returns:
        A BaseMessage object
    """
    if isinstance(message, BaseMessage):
        return message

    role = message.get("role", "user")
    content = message.get("content", "")

    if role == "system":
        return SystemMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    else:
        return HumanMessage(content=content)


def normalize_messages(
    messages: List[Union[Dict[str, Any], BaseMessage]], target_format: str = "dict"
) -> List[Union[Dict[str, Any], BaseMessage]]:
    """Normalize a list of messages to a consistent format.

    Args:
        messages: List of messages in mixed formats
        target_format: Either "dict" or "base_message"

    Returns:
        List of messages in the target format
    """
    if target_format == "dict":
        return [convert_to_dict(msg) for msg in messages]
    elif target_format == "base_message":
        return [convert_to_base_message(msg) for msg in messages]
    else:
        raise ValueError(f"Unknown target format: {target_format}")


def count_message_tokens(
    message: Union[Dict[str, Any], BaseMessage], encoding_name: str = "cl100k_base"
) -> int:
    """Count tokens in a message.

    Args:
        message: The message to count tokens for
        encoding_name: The tiktoken encoding to use

    Returns:
        Number of tokens
    """
    import tiktoken

    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    content = get_message_content(message)
    role = get_message_role(message)

    # Count tokens for both role and content
    return len(encoding.encode(role)) + len(encoding.encode(content))


def truncate_message_content(
    content: str, max_tokens: int, encoding_name: str = "cl100k_base"
) -> str:
    """Truncate message content to fit within token limit.

    Args:
        content: The content to truncate
        max_tokens: Maximum number of tokens
        encoding_name: The tiktoken encoding to use

    Returns:
        Truncated content
    """
    import tiktoken

    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(content)

    if len(tokens) <= max_tokens:
        return content

    # Truncate and add ellipsis
    truncated_tokens = tokens[: max_tokens - 3]  # Reserve space for "..."
    truncated_content = encoding.decode(truncated_tokens)

    return truncated_content + "..."


def safe_message_access(
    messages: List[Union[Dict[str, Any], BaseMessage]],
    index: int,
    default: Union[Dict[str, Any], BaseMessage] | None = None,
) -> Union[Dict[str, Any], BaseMessage] | None:
    """Safely access a message by index with bounds checking.

    Args:
        messages: List of messages
        index: Index to access
        default: Default value if index is out of bounds

    Returns:
        The message at index or default value
    """
    try:
        if -len(messages) <= index < len(messages):
            return messages[index]
    except (IndexError, TypeError):
        pass

    return default


def filter_messages_by_role(
    messages: List[Union[Dict[str, Any], BaseMessage]], role: str
) -> List[Union[Dict[str, Any], BaseMessage]]:
    """Filter messages by role.

    Args:
        messages: List of messages
        role: Role to filter by

    Returns:
        List of messages with the specified role
    """
    return [msg for msg in messages if get_message_role(msg) == role]


def merge_system_messages(
    messages: List[Union[Dict[str, Any], BaseMessage]],
) -> List[Union[Dict[str, Any], BaseMessage]]:
    """Merge consecutive system messages into a single message.

    Args:
        messages: List of messages

    Returns:
        List with merged system messages
    """
    if not messages:
        return messages

    result: List[Union[Dict[str, Any], BaseMessage]] = []
    current_system_content = []

    # Determine the target format based on the first message
    target_is_dict = isinstance(messages[0], dict) if messages else True

    for msg in messages:
        if get_message_role(msg) == "system":
            current_system_content.append(get_message_content(msg))
        else:
            # If we have accumulated system messages, merge them
            if current_system_content:
                merged_content = "\n\n".join(current_system_content)
                if target_is_dict:
                    result.append({"role": "system", "content": merged_content})
                else:
                    result.append(SystemMessage(content=merged_content))
                current_system_content = []

            result.append(msg)

    # Don't forget any trailing system messages
    if current_system_content:
        merged_content = "\n\n".join(current_system_content)
        if target_is_dict:
            result.append({"role": "system", "content": merged_content})
        else:
            result.append(SystemMessage(content=merged_content))

    return result
