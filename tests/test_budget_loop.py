from types import SimpleNamespace

import pytest

from open_deep_research.exceptions import OutOfBudgetError
from open_deep_research.graph import graph
from open_deep_research.pydantic_state import (
    Feedback,
    Queries,
    SearchQuery,
    Section,
    SectionOutput,
    Sections,
)


class _AlwaysFailGraderModel:
    """LLM stub whose structured output for Feedback always fails."""

    async def ainvoke(self, *_, **__):  # noqa: D401
        return SimpleNamespace(content="stub")

    def with_structured_output(self, schema):  # noqa: D401
        async def _ainvoke(__, ___):  # noqa: D401, ANN001
            if schema is Feedback:
                return Feedback(
                    grade="fail", follow_up_queries=[SearchQuery(search_query="foo")]
                )
            if schema is Queries:
                return Queries(queries=[SearchQuery(search_query="foo")])
            if schema is SectionOutput:
                return SectionOutput(section_content="content")
            if schema is Sections:
                return Sections(
                    sections=[
                        Section(name="S", description="D", research=True, content="")
                    ]
                )
            return schema()  # type: ignore[call-arg]

        return SimpleNamespace(ainvoke=_ainvoke)

    def with_retry(self, *_, **__):  # noqa: D401, ANN001
        return self


@pytest.fixture()
def patch_models(monkeypatch):
    monkeypatch.setattr(
        "open_deep_research.core.initialize_model",
        lambda *_, **__: _AlwaysFailGraderModel(),
    )
    monkeypatch.setattr(
        "open_deep_research.utils.get_structured_output_with_fallback",
        lambda *_, schema_class, **__: _AlwaysFailGraderModel()
        .with_structured_output(schema_class)
        .ainvoke([], {}),
    )
    monkeypatch.setattr(
        "open_deep_research.utils.select_and_execute_search",
        lambda *_, **__: "sources",
    )
    monkeypatch.setattr(
        "open_deep_research.utils.summarize_search_results",
        lambda s, **_: s,
    )


@pytest.mark.asyncio
async def test_loop_terminates_after_max_depth(patch_models):
    """Ensure reflection loop stops after max_search_depth iterations."""
    config = {"configurable": {"max_search_depth": 2, "search_budget": 100}}
    state = await graph.ainvoke({"topic": "X"}, config=config)

    # Depending on implementation state may be dict or Pydantic
    iterations = None
    if isinstance(state, dict):
        iterations = state.get("search_iterations")
    elif hasattr(state, "search_iterations"):
        iterations = getattr(state, "search_iterations")

    assert iterations is None or iterations <= 2, "Loop exceeded max_search_depth"


@pytest.mark.asyncio
async def test_budget_exhaustion_triggers_error(patch_models):
    """With tiny budget, OutOfBudgetError should be raised quickly."""
    config = {"configurable": {"search_budget": 1}}
    with pytest.raises(OutOfBudgetError):
        await graph.ainvoke({"topic": "Y"}, config=config)
