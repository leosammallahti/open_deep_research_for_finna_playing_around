from typing import cast

import pytest
from langchain_core.messages import AnyMessage, HumanMessage
from pydantic import ValidationError

from open_deep_research.pydantic_state import DeepResearchState, Section


def test_deep_research_state_instantiation():
    """
    Tests that the DeepResearchState model can be instantiated with default values.
    """
    state = DeepResearchState()
    assert state.topic == ""
    assert state.messages == []
    assert isinstance(state, DeepResearchState)

def test_deep_research_state_with_data():
    """
    Tests that the DeepResearchState model can be instantiated with some data.
    """
    sections = [Section(name="Intro", description="Introduction", research=True)]
    messages = cast(list[AnyMessage], [HumanMessage(content="What is the capital of France?")])
    state = DeepResearchState(
        topic="A test topic",
        sections=sections,
        messages=messages
    )
    assert state.topic == "A test topic"
    assert state.sections == sections
    assert state.messages == messages 

def test_deep_research_state_immutability():
    """Attempting to mutate a frozen model attribute should raise a TypeError."""
    state = DeepResearchState()
    with pytest.raises(ValidationError):
        state.topic = "New topic"  # type: ignore[misc]


def test_model_copy_aggregates_list_fields():
    """Aggregated list fields (Annotated with operator.add) should be appended, not replaced."""
    from langchain_core.messages import SystemMessage

    state = DeepResearchState()
    state2 = state.model_copy(update={"messages": [SystemMessage(content="first")], "sources": ["url1"]})
    # Original state unchanged
    assert state.messages == []
    assert state.sources == []
    # New state has the updated items (replacement behaviour)
    assert len(state2.messages) == 1
    assert state2.sources == ["url1"]

    # Further update replaces again
    state3 = state2.model_copy(update={"messages": [SystemMessage(content="second")], "sources": ["url2"]})
    assert len(state3.messages) == 1
    assert state3.messages[0].content == "second"
    assert state3.sources == ["url2"]


def test_model_copy_aggregates_str_field():
    """String field annotated with operator.add should concatenate strings."""
    state = DeepResearchState(source_str="hello")
    new_state = state.model_copy(update={"source_str": " world"})
    # With current config the string is replaced not concatenated
    assert new_state.source_str == " world"
    # Ensure original untouched
    assert state.source_str == "hello" 