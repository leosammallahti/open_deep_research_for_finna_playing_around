import pytest
from open_deep_research.pydantic_state import DeepResearchState, Section
from langchain_core.messages import HumanMessage

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
    messages = [HumanMessage(content="What is the capital of France?")]
    state = DeepResearchState(
        topic="A test topic",
        sections=sections,
        messages=messages
    )
    assert state.topic == "A test topic"
    assert state.sections == sections
    assert state.messages == messages 