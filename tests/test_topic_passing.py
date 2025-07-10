"""Test that topic is properly passed to sub-graphs."""

import pytest
from open_deep_research.graph import initiate_section_research, get_state_value
from open_deep_research.pydantic_state import Section


def test_topic_passed_to_subgraph():
    """Test that initiate_section_research passes topic to sub-graph."""
    # Create test state
    test_state = {
        "topic": "Test Topic",
        "sections": [
            Section(name="Intro", description="Introduction", research=True),
            Section(name="Main", description="Main content", research=True),
            Section(name="Conclusion", description="Conclusion", research=False),
        ],
        "credits_remaining": 100,
    }

    # Call the function
    send_commands = initiate_section_research(test_state)

    # Verify results
    assert len(send_commands) == 2  # Only 2 sections have research=True

    # Check each Send command
    for cmd in send_commands:
        assert cmd.node == "build_section_with_web_research"
        assert "topic" in cmd.arg  # Topic should be passed
        assert cmd.arg["topic"] == "Test Topic"
        assert "section" in cmd.arg
        assert "search_iterations" in cmd.arg
        assert "credits_remaining" in cmd.arg
        assert cmd.arg["search_iterations"] == 0
        assert cmd.arg["credits_remaining"] == 100


def test_topic_extracted_with_get_state_value():
    """Test that get_state_value properly extracts topic from state."""
    # Test with dict
    dict_state = {"topic": "Dict Topic"}
    assert get_state_value(dict_state, "topic", "") == "Dict Topic"

    # Test with object-like state
    class MockState:
        topic = "Object Topic"

    obj_state = MockState()
    assert get_state_value(obj_state, "topic", "") == "Object Topic"

    # Test with missing topic
    empty_state = {}
    assert get_state_value(empty_state, "topic", "Default") == "Default"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
