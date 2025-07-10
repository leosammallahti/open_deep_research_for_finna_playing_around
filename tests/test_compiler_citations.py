import pytest
from dataclasses import asdict

from open_deep_research.pydantic_state import DeepResearchState, Section
from open_deep_research.workflow.unified_compiler import compile_report_unified
from open_deep_research.configuration import WorkflowConfiguration, SearchAPI
from langchain_core.runnables import RunnableConfig


@pytest.mark.asyncio
async def test_compiler_adds_sources_and_raw_block():
    """The unified compiler should append a Sources section and optional raw dump."""

    # Two sections with completed content
    completed = [
        Section(
            name="Background",
            description="desc",
            research=True,
            content="Content citing [1] and [2].",
        ),
        Section(
            name="Results",
            description="desc",
            research=True,
            content="More content citing [2].",
        ),
    ]

    # Simulate aggregated source_str from researcher path
    raw_sources = (
        "URL: https://foo.com/article\n"
        "Title: Foo Article\n\n"
        "URL: https://bar.com/report\n"
        "Title: Bar Report\n"
    )

    state = DeepResearchState(
        topic="Test",
        sections=[s.model_copy(update={"content": ""}) for s in completed],
        completed_sections=completed,
        source_str=raw_sources,
    )

    cfg = WorkflowConfiguration(
        search_api=SearchAPI.NONE,
        include_source_str=True,
        include_raw_source_details=True,
    )
    from typing import cast

    runnable_cfg: RunnableConfig = cast(RunnableConfig, {"configurable": asdict(cfg)})

    patch = await compile_report_unified(state, runnable_cfg)
    report = patch["final_report"]

    # Check sections appear
    assert "# Background" in report and "# Results" in report

    # Check sources section present with numbered URLs
    assert "## Sources" in report
    assert "[1] https://foo.com/article" in report
    assert "[2] https://bar.com/report" in report

    # Check raw source block appended
    assert "## Raw Sources" in report
    assert "https://foo.com/article" in report and "https://bar.com/report" in report
