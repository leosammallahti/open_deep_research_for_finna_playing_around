from types import SimpleNamespace

import pytest

from open_deep_research.graph import graph
from open_deep_research.pydantic_state import (
    Feedback,
    Queries,
    SearchQuery,
    Section,
    SectionOutput,
    Sections,
)


class _StubLLM:
    """Return deterministic structured outputs so the graph completes quickly."""

    async def ainvoke(self, *_, **__):
        return SimpleNamespace(content="stubbed")

    def with_structured_output(self, schema):  # noqa: D401
        async def _ainvoke(__, ___):  # noqa: ANN001
            if schema is Sections:
                return Sections(
                    sections=[
                        Section(
                            name="Intro",
                            description="Intro section",
                            research=False,
                            content="",
                        ),
                        Section(
                            name="Key Findings",
                            description="Key findings",
                            research=True,
                            content="",
                        ),
                        Section(
                            name="Conclusion",
                            description="Conclusion",
                            research=False,
                            content="",
                        ),
                    ]
                )
            if schema is Queries:
                return Queries(queries=[SearchQuery(search_query="test query")])
            if schema is SectionOutput:
                return SectionOutput(section_content="Draft content")
            if schema is Feedback:
                return Feedback(grade="pass", follow_up_queries=[])
            return schema()  # type: ignore[call-arg]

        return SimpleNamespace(ainvoke=_ainvoke)

    def with_retry(self, *_, **__):  # noqa: D401, ANN001
        return self


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(
        "open_deep_research.core.initialize_model",
        lambda *_, **__: _StubLLM(),
    )
    monkeypatch.setattr(
        "open_deep_research.utils.get_structured_output_with_fallback",
        lambda *_, schema_class, **__: _StubLLM()
        .with_structured_output(schema_class)
        .ainvoke([], {}),
    )

    # Stub search to return canned JSON
    canned_response = [
        {
            "query": "test query",
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "content": "Example content",
                    "raw_content": "Full example content",
                }
            ],
        }
    ]
    monkeypatch.setattr(
        "open_deep_research.utils.select_and_execute_search",
        lambda *_,
        **__: "Stubbed Source: Example\nURL: https://example.com\n===\nExample content",
    )
    monkeypatch.setattr(
        "open_deep_research.utils.summarize_search_results",
        lambda s, **_: s,
    )


@pytest.mark.asyncio
async def test_graph_end_to_end_fast():
    """End-to-end run finishes and yields a final_report quickly."""
    final_state = await graph.ainvoke(
        {"topic": "Demo"}, config={"configurable": {"search_budget": 10}}
    )

    final_report = None
    if isinstance(final_state, dict):
        final_report = final_state.get("final_report")
    else:
        final_report = getattr(final_state, "final_report", None)

    assert final_report, "Graph did not produce a final_report"
    assert "Example" in final_report
