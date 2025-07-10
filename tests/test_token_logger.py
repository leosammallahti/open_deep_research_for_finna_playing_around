import types

import pytest

from open_deep_research.core.model_utils import _TokenLogger


class DummyGenerations:
    """Minimal stand-in for `LLMResult.generations`."""

    def __init__(self, texts):
        self._texts = texts

    def __iter__(self):
        for text in self._texts:
            yield [types.SimpleNamespace(text=text)]


@pytest.mark.parametrize(
    "llm_output,generations",
    [
        # Case 1: Provider returns explicit usage metadata → preferred path.
        (
            {"usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}},
            [],
        ),
        # Case 2: No usage metadata, fallback to char length heuristic.
        (
            None,
            ["hello world"],
        ),
    ],
)
def test_token_logger_on_llm_end_does_not_error(
    tmp_path, monkeypatch, llm_output, generations
):
    """`_TokenLogger.on_llm_end` should gracefully handle both code paths."""

    # Patch CREDIT_LOG_FILE so tests never touch the real log.
    monkeypatch.setenv("CREDIT_LOG_FILE", str(tmp_path / "dummy.jsonl"))

    logger = _TokenLogger("test-provider", "test-model")

    outputs = types.SimpleNamespace(
        llm_output=llm_output,
        generations=DummyGenerations(generations),
    )

    # Must accept *positional* response arg (regression check).
    logger.on_llm_end(outputs)  # should not raise

    # The log file should now exist (even if empty token counts).
    assert (tmp_path / "dummy.jsonl").exists()
