"""Test concurrent state updates in Pydantic models."""

import pytest
from typing import get_type_hints
from open_deep_research.pydantic_state import (
    DeepResearchState,
    SectionResearchState,
    _max_fn,
    _min_fn,
    MaxFn,
    MinFn,
)


def test_reducer_functions():
    """Test that reducer functions work correctly."""
    # Test MaxFn
    assert _max_fn(5, 10) == 10
    assert _max_fn(10, 5) == 10
    assert _max_fn(7, 7) == 7

    # Test MinFn
    assert _min_fn(5, 10) == 5
    assert _min_fn(10, 5) == 5
    assert _min_fn(7, 7) == 7


def test_deep_research_state_annotations():
    """Test that DeepResearchState fields have proper annotations for concurrent updates."""
    hints = get_type_hints(DeepResearchState, include_extras=True)

    # Check search_iterations has MaxFn reducer
    search_iter_type = hints.get("search_iterations")
    assert hasattr(search_iter_type, "__metadata__"), (
        "search_iterations should be Annotated"
    )
    assert search_iter_type.__metadata__[0] == MaxFn, (
        "search_iterations should use MaxFn reducer"
    )

    # Check credits_remaining has MinFn reducer
    credits_type = hints.get("credits_remaining")
    assert hasattr(credits_type, "__metadata__"), (
        "credits_remaining should be Annotated"
    )
    assert credits_type.__metadata__[0] == MinFn, (
        "credits_remaining should use MinFn reducer"
    )


def test_section_research_state_annotations():
    """Test that SectionResearchState fields have proper annotations for concurrent updates."""
    hints = get_type_hints(SectionResearchState, include_extras=True)

    # Check search_iterations has MaxFn reducer
    search_iter_type = hints.get("search_iterations")
    assert hasattr(search_iter_type, "__metadata__"), (
        "search_iterations should be Annotated"
    )
    assert search_iter_type.__metadata__[0] == MaxFn, (
        "search_iterations should use MaxFn reducer"
    )

    # Check credits_remaining has MinFn reducer
    credits_type = hints.get("credits_remaining")
    assert hasattr(credits_type, "__metadata__"), (
        "credits_remaining should be Annotated"
    )
    assert credits_type.__metadata__[0] == MinFn, (
        "credits_remaining should use MinFn reducer"
    )


def test_concurrent_update_simulation():
    """Simulate how LangGraph would handle concurrent updates with our reducers."""
    # Test scenario: Multiple sections updating search_iterations concurrently
    initial_iterations = 0
    concurrent_updates = [3, 1, 5, 2]  # Different sections reporting their iterations

    # Simulate how LangGraph applies updates with MaxFn
    result = initial_iterations
    for update in concurrent_updates:
        result = MaxFn(result, update)

    assert result == 5, "MaxFn should return the highest iteration count"

    # Test scenario: Multiple sections updating credits_remaining concurrently
    initial_credits = 100
    concurrent_credit_updates = [
        80,
        60,
        90,
        70,
    ]  # Different sections reporting remaining credits

    # Simulate how LangGraph applies updates with MinFn
    result = initial_credits
    for update in concurrent_credit_updates:
        result = MinFn(result, update)

    assert result == 60, (
        "MinFn should return the lowest credit count (most conservative)"
    )


def test_state_creation_with_concurrent_fields():
    """Test that states can be created with fields that support concurrent updates."""
    # Create DeepResearchState
    state = DeepResearchState(
        topic="Test Topic", search_iterations=5, credits_remaining=100
    )

    assert state.search_iterations == 5
    assert state.credits_remaining == 100

    # Create SectionResearchState
    section_state = SectionResearchState(
        topic="Test Topic", search_iterations=2, credits_remaining=50
    )

    assert section_state.search_iterations == 2
    assert section_state.credits_remaining == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
