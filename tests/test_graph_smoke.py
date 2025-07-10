from types import SimpleNamespace

import pytest

# Import the graph to ensure the module is present and can be executed
from open_deep_research.graph import graph

# Pydantic schemas we need for dummy returns
from open_deep_research.pydantic_state import (
    Feedback,
    Queries,
    SearchQuery,
    Section,
    SectionOutput,
    Sections,
)


@pytest.fixture(autouse=True)
def _patch_external_dependencies(monkeypatch):
    """Monkey-patch external calls so the graph can run offline in a few ms."""

    # --- Dummy LLM implementation -------------------------------------------------
    class _DummyLLM:
        """Minimal async-compatible stub for a chat model."""

        async def ainvoke(self, *args, **kwargs):  # noqa: D401, ANN001
            # Return something with a .content attribute where needed
            return SimpleNamespace(content="stubbed_content")

        # Support `.with_structured_output(schema)` API used in codebase
        def with_structured_output(self, schema):  # noqa: D401, ANN001
            async def _ainvoke(messages, **_kw):  # noqa: ANN001
                # Return the minimal valid object for the requested schema
                if schema is Sections:
                    return Sections(
                        sections=[
                            Section(
                                name="Introduction",
                                description="Intro section",
                                research=False,
                                content="",
                            )
                        ]
                    )
                if schema is Queries:
                    return Queries(queries=[SearchQuery(search_query="example query")])
                if schema is SectionOutput:
                    return SectionOutput(section_content="Draft section content")
                if schema is Feedback:
                    return Feedback(grade="pass", follow_up_queries=[])
                # Fallback: instantiate empty schema
                return schema()  # type: ignore[call-arg]

            # Return an object exposing `ainvoke`
            return SimpleNamespace(ainvoke=_ainvoke)

        # Provide dummy `.with_retry` just returning self
        def with_retry(self, *args, **kwargs):  # noqa: D401, ANN001
            return self

    # Patch model initialisation to return dummy model
    monkeypatch.setattr(
        "open_deep_research.core.initialize_model",
        lambda *args, **kwargs: _DummyLLM(),
    )

    # Patch structured-output helper to skip network calls
    async def _dummy_structured_output(model, schema, messages, model_id=None):  # noqa: D401, ANN001
        # Directly use the dummy model logic above
        dummy_model = _DummyLLM()
        result = await dummy_model.with_structured_output(schema).ainvoke(messages)
        return result

    monkeypatch.setattr(
        "open_deep_research.utils.get_structured_output_with_fallback",
        _dummy_structured_output,
    )

    # Stub out the web-search helper so no external calls occur
    monkeypatch.setattr(
        "open_deep_research.utils.select_and_execute_search",
        lambda *args, **kwargs: "stubbed source string",
    )

    # Summarisation helper can just echo input
    monkeypatch.setattr(
        "open_deep_research.utils.summarize_search_results",
        lambda source_str, **_: source_str,
    )


@pytest.mark.asyncio
async def test_graph_smoke():
    """Ensure the LangGraph pipeline executes end-to-end without external services."""

    # Minimal input – only the topic is strictly required
    output_state = await graph.ainvoke(
        {"topic": "Toy topic"}, config={"recursion_limit": 20}
    )

    # The patched pipeline should always return a dict/Pydantic model with final_report
    final_report = None
    if isinstance(output_state, dict):
        final_report = output_state.get("final_report")
    else:
        final_report = getattr(output_state, "final_report", None)

    assert final_report is not None, "Graph run did not produce a final_report"
