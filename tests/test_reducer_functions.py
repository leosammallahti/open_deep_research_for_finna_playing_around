from typing import List

import hypothesis.strategies as st
from hypothesis import given

from open_deep_research.pydantic_state import Section, add_sections


@given(
    existing=st.lists(
        st.builds(Section, name=st.text(min_size=1), description=st.text(), research=st.booleans()),
        max_size=10,
    ),
    new=st.lists(
        st.builds(Section, name=st.text(min_size=1), description=st.text(), research=st.booleans()),
        max_size=10,
    ),
)
def test_add_sections_no_duplicates(existing: List[Section], new: List[Section]):
    """add_sections should return a list with unique section names only."""

    merged = add_sections(existing, new)
    names = [s.name for s in merged]
    assert len(names) == len(set(names))  # no duplicate names
    # All originals should still be present
    for sec in existing:
        assert sec.name in names
    for sec in new:
        assert sec.name in names 