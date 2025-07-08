from typing import List

import pytest
from langchain.schema import HumanMessage
from langchain.schema.runnable import Runnable
from pydantic import BaseModel, Field

from src.open_deep_research.models import FakeListChatModel
from src.open_deep_research.utils import (
    filter_think_tokens,
    get_structured_output_with_fallback,
)


@pytest.mark.parametrize("input_text, expected_output", [
    # Standard case
    ("Here is some text. <think>This is a thought.</think> Here is more text.", "Here is some text. Here is more text."),
    # No think tokens
    ("This text has no think tokens.", "This text has no think tokens."),
    # Multiple think tokens
    ("<think>First thought.</think>Some content.<think>Second thought.</think>", "Some content."),
    # Think tokens at start and end
    ("<think>Start thought.</think>Content<think>End thought.</think>", "Content"),
    # Nested or malformed (regex should handle greediness correctly)
    ("Text with <think>a <think>nested</think> thought</think> inside.", "Text with inside."),
    # Empty content
    ("", ""),
    # Only think tokens
    ("<think>This is all a thought.</think>", ""),
    # Non-string input
    (123, 123),
    (None, None),
])
def test_filter_think_tokens(input_text, expected_output):
    """Tests the filter_think_tokens utility function."""
    assert filter_think_tokens(input_text) == expected_output


# --- Test get_structured_output_with_fallback ---

# 1. Define a mock schema and model for testing
class MockSchema(BaseModel):
    name: str = Field(description="A name")
    value: int = Field(description="A number")

class MockChatModel(FakeListChatModel):
    """A fake model that returns a predefined response."""
    def __init__(self, responses: List[str]):
        super().__init__(responses=responses)

    def with_structured_output(self, schema, **kwargs) -> Runnable:
        # Simulate a model that doesn't support tool choice by raising an error
        # This is what a real model like deepseek-reasoner would do internally
        raise NotImplementedError("This model does not support tool_choice.")

# 2. Define the test cases
@pytest.mark.asyncio
async def test_fallback_logic_with_non_tool_supporting_model():
    """
    Verify that get_structured_output_with_fallback uses prompt-based extraction
    when the model does not support tool choice.
    """
    # Arrange: Create a mock model that will raise an error on with_structured_output
    # and return a JSON blob in its string response.
    responses = [
        'Some conversational text... here is the JSON you asked for: \n```json\n{"name": "Test", "value": 123}\n```'
    ]
    mock_model = MockChatModel(responses=responses)
    
    # Act: Call the function with a model_id that is known to not support tool choice
    result = await get_structured_output_with_fallback(
        mock_model,
        MockSchema,
        [HumanMessage(content="some prompt")],
        model_id="deepseek:deepseek-reasoner" # This ID signals that fallback is needed
    )

    # Assert: Check that the result is a correctly parsed Pydantic object
    assert isinstance(result, MockSchema)
    assert result.name == "Test"
    assert result.value == 123 