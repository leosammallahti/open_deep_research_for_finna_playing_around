"""Test concurrent section writing to ensure no state conflicts."""

from dataclasses import asdict

import pytest
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import SearchAPI, WorkflowConfiguration
from open_deep_research.graph import graph
from open_deep_research.pydantic_state import DeepResearchState, Section


@pytest.mark.asyncio
async def test_parallel_final_sections_no_conflicts():
    """Test that multiple final sections can be written in parallel without conflicts."""

    # Create a state with multiple non-research sections that will be processed in parallel
    initial_state = DeepResearchState(
        topic="Artificial Intelligence",
        sections=[
            Section(name="Introduction", description="Overview of AI", research=True),
            Section(
                name="Conclusion",
                description="Summary and future outlook",
                research=False,
            ),
            Section(
                name="References", description="List of references", research=False
            ),
            Section(
                name="Acknowledgments",
                description="Thanks to contributors",
                research=False,
            ),
        ],
        # Simulate that research sections are already completed
        completed_sections=[
            Section(
                name="Introduction",
                description="Overview of AI",
                research=True,
                content="# Introduction\n\nArtificial Intelligence (AI) is...",
            )
        ],
    )

    # Configure to use a fast model and skip expensive operations
    config = RunnableConfig(
        configurable=asdict(
            WorkflowConfiguration(
                search_api=SearchAPI.NONE,  # Skip actual web searches
                max_search_depth=1,
                writer_provider="together",
                writer_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                include_source_str=False,
            )
        )
    )

    # Track which nodes execute
    executed_nodes = []

    # Run the graph and collect results
    try:
        final_state = None
        async for chunk in graph.astream(initial_state, config):
            executed_nodes.append(list(chunk.keys())[0])
            # Keep the last state
            for node, state in chunk.items():
                if state:
                    final_state = state

    except Exception as e:
        # Check if it's the concurrent update error
        if "Can receive only one value per step" in str(e):
            pytest.fail(f"Concurrent update error occurred: {e}")
        else:
            # Re-raise other errors
            raise

    # Verify the workflow completed successfully
    assert final_state is not None
    assert final_state.get("final_report", "") != ""

    # Verify that write_final_sections was called (potentially multiple times in parallel)
    write_final_sections_count = executed_nodes.count("write_final_sections")
    assert write_final_sections_count >= 2, (
        f"Expected multiple write_final_sections calls, got {write_final_sections_count}"
    )

    # Verify all non-research sections were processed
    assert (
        len(final_state.get("completed_sections", [])) >= 3
    )  # 1 research + at least 2 non-research


@pytest.mark.asyncio
async def test_section_field_isolation():
    """Test that the 'section' field doesn't cause conflicts in parallel execution."""

    # Create a minimal state to test the specific edge case
    initial_state = DeepResearchState(
        topic="Test Topic",
        sections=[
            Section(name="Section A", description="Non-research A", research=False),
            Section(name="Section B", description="Non-research B", research=False),
        ],
        completed_sections=[],
    )

    config = RunnableConfig(
        configurable=asdict(
            WorkflowConfiguration(
                search_api=SearchAPI.NONE,
                writer_provider="together",
                writer_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            )
        )
    )

    # This should not raise any concurrent update errors
    chunks = []
    async for chunk in graph.astream(initial_state, config):
        chunks.append(chunk)

    # The test passes if no exception was raised
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_concurrent_send_no_conflicts():
    """Test that the Send commands don't cause conflicts when updating state."""
    from open_deep_research.graph import initiate_final_section_writing

    # Create test state with multiple non-research sections
    test_state = DeepResearchState(
        topic="Test Topic",
        sections=[
            Section(name="Conclusion", description="Summary", research=False),
            Section(name="References", description="Citations", research=False),
            Section(name="Appendix", description="Additional info", research=False),
        ],
        completed_sections=[],
    )

    # Get the Send commands that would be created
    send_commands = initiate_final_section_writing(test_state)

    # Verify we got multiple Send commands
    assert len(send_commands) == 3

    # Verify each Send command uses 'current_section' not 'section'
    for cmd in send_commands:
        assert cmd.node == "write_final_sections"
        assert "current_section" in cmd.arg
        assert "section" not in cmd.arg  # The old field name should not be used
        assert cmd.arg["current_section"] in test_state.sections
