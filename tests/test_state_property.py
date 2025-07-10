# New file tests/test_state_property.py
"""Property‐based tests for state models.

These tests rely on *Hypothesis* to generate random but valid instances of our
state models and verify crucial invariants:

* DeepResearchState and SectionResearchState are immutable (frozen=True) – any
  attempted attribute assignment raises ``TypeError``
* ``model_copy(update=…)`` returns a *new* object with the update applied while
  leaving the original untouched.

Run with::

    pytest -q tests/test_state_property.py
"""

from __future__ import annotations

from typing import List

import hypothesis.strategies as st
from hypothesis import given

from open_deep_research.pydantic_state import (
    DeepResearchState,
    Section,
    SectionResearchState,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Simplified strategy for Section objects (enough for immutability tests)
section_strategy = st.builds(
    Section,
    name=st.text(min_size=1, max_size=20),
    description=st.text(min_size=1, max_size=50),
    research=st.booleans(),
    content=st.text(max_size=200),
)

# Strategy for DeepResearchState – we only set a subset of fields to keep the
# combinatorial explosion in check.
state_strategy = st.builds(
    DeepResearchState,
    topic=st.text(min_size=1, max_size=30),
    sections=st.lists(section_strategy, max_size=5),
    feedback=st.lists(st.text(max_size=50), max_size=3),
)

# Strategy for SectionResearchState – reuses section strategy.
section_state_strategy = st.builds(
    SectionResearchState,
    topic=st.text(min_size=1, max_size=30),
    section=section_strategy,
    search_queries=st.lists(st.text(max_size=50), max_size=3),
    source_str=st.text(max_size=500),
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@given(state=state_strategy)
def test_deepresearchstate_frozen_and_copy(state: DeepResearchState) -> None:
    """˚DeepResearchState should be immutable and copyable."""

    # Attempting to assign should raise TypeError
    try:
        state.topic = "New Topic"  # type: ignore[misc]
    except TypeError:
        pass
    else:
        raise AssertionError("DeepResearchState is not frozen (assignment succeeded)")

    # model_copy should produce a new instance with update
    new_state = state.model_copy(update={"topic": "Updated"})
    assert new_state.topic == "Updated" and state.topic != new_state.topic


@given(state=section_state_strategy)
def test_sectionresearchstate_frozen_and_copy(state: SectionResearchState) -> None:
    """SectionResearchState immutability & copy behaviour."""

    # Immutability check
    try:
        state.topic = "Changed"  # type: ignore[misc]
    except TypeError:
        pass
    else:
        raise AssertionError("SectionResearchState is not frozen")

    # model_copy update test
    new_state = state.model_copy(
        update={"search_iterations": state.search_iterations + 1}
    )
    assert new_state.search_iterations == state.search_iterations + 1
    assert state.search_iterations != new_state.search_iterations
