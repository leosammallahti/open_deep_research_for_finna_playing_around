import importlib
import os
import sys
from pathlib import Path

import pytest

SETTINGS_MODULE = "open_deep_research.core.settings"


def _reload_settings():
    if SETTINGS_MODULE in sys.modules:
        del sys.modules[SETTINGS_MODULE]
    return importlib.import_module(SETTINGS_MODULE)


def test_env_file_loading(tmp_path: Path, monkeypatch):
    # Create temporary .env file
    env_path = tmp_path / ".env"
    env_path.write_text("TAVILY_API_KEY=abc123\nLOG_LEVEL=DEBUG\n")
    monkeypatch.chdir(tmp_path)  # change CWD so .env is discovered
    settings_module = _reload_settings()
    settings = settings_module.settings
    if os.environ.get("TAVILY_API_KEY") and os.environ["TAVILY_API_KEY"] != "abc123":
        pytest.skip("Environment already defines TAVILY_API_KEY – skipping overwrite test")
    assert settings.tavily_api_key == "abc123"
    assert settings.log_level == "DEBUG"
    # Should also be injected into os.environ
    assert os.environ["TAVILY_API_KEY"] == "abc123" 