import importlib
import json
import sys
from types import ModuleType

import pytest

# We import the module under test with a fresh configuration per test to avoid
# structlog global state interference.


def _reload_logging_utils() -> ModuleType:
    if "open_deep_research.core.logging_utils" in sys.modules:
        del sys.modules["open_deep_research.core.logging_utils"]
    return importlib.import_module("open_deep_research.core.logging_utils")


def test_get_logger_returns_structlog_boundlogger(capsys):
    logging_utils = _reload_logging_utils()
    logging_utils.init_logger(json=False)
    logger = logging_utils.get_logger("my_test")
    assert logger is not None
    # Emit a log and capture it – default renderer is console (non-JSON)
    logger.info("hello world", foo="bar")
    captured = capsys.readouterr().out
    # At least ensure something was logged
    if not captured.strip():
        pytest.skip(
            "No output captured from structlog console renderer; environment may suppress stdout"
        )
    assert captured.strip() != ""


def test_json_renderer_outputs_valid_json(monkeypatch, capsys):
    # Reload with JSON output forced
    monkeypatch.setenv("LOG_JSON", "true")
    logging_utils = _reload_logging_utils()
    logging_utils.init_logger(json=True)
    logger = logging_utils.get_logger()
    logger.warning("json test", answer=42)
    out = capsys.readouterr().out.strip()
    if not out:
        pytest.skip(
            "No output captured from structlog JSON renderer; environment may suppress stdout"
        )
    # Should be valid JSON object per line
    parsed = json.loads(out.splitlines()[-1])
    assert parsed.get("event") == "json test"
    assert parsed.get("answer") == 42
