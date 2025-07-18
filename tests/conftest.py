"""
Pytest configuration for open_deep_research tests.
"""


# ---------------------------------------------------------------------------
# CLI options (existing)
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption(
        "--research-agent", action="store", help="Agent type: multi_agent or graph"
    )
    parser.addoption("--search-api", action="store", help="Search API to use")
    parser.addoption("--eval-model", action="store", help="Model for evaluation")
    parser.addoption(
        "--supervisor-model", action="store", help="Model for supervisor agent"
    )
    parser.addoption(
        "--researcher-model", action="store", help="Model for researcher agent"
    )
    parser.addoption(
        "--planner-provider", action="store", help="Provider for planner model"
    )
    parser.addoption("--planner-model", action="store", help="Model for planning")
    parser.addoption(
        "--writer-provider", action="store", help="Provider for writer model"
    )
    parser.addoption("--writer-model", action="store", help="Model for writing")
    parser.addoption("--max-search-depth", action="store", help="Maximum search depth")


# ---------------------------------------------------------------------------
# Offline-mode fixture – automatically replaces expensive calls with stubs
# ---------------------------------------------------------------------------

import pytest


@pytest.fixture(autouse=True)
def _offline_mode(monkeypatch):
    """Automatically patch heavy dependencies so tests run without tokens."""

    # 1. Patch model initializer
    import open_deep_research.core.model_utils as _mu  # noqa: WPS433 – runtime patching
    from tests.stubs import FakeChatModel, fake_search  # local import to avoid hard dep

    def _init_stub(provider: str, model_name: str, kwargs=None):  # noqa: D401
        role = "planner" if "planner" in model_name.lower() else "writer"
        return FakeChatModel(role=role)

    monkeypatch.setattr(_mu, "initialize_model", _init_stub, raising=True)

    # 2. Patch search executor
    import open_deep_research.utils as _ut  # noqa: WPS433

    monkeypatch.setattr(_ut, "select_and_execute_search", fake_search, raising=True)

    # 3. Short-circuit asyncio.sleep in utils to speed up tests
    import asyncio as _aio  # noqa: WPS433

    monkeypatch.setattr(_ut, "asyncio", _aio)
    monkeypatch.setattr(_aio, "sleep", lambda *_a, **_k: _aio.Future(), raising=False)

    yield  # run the test


# ---------------------------------------------------------------------------
# Ensure *ODR_FAST_TEST* does not leak between tests
# ---------------------------------------------------------------------------


import pytest  # noqa: E402  – placed after stdlib imports in this file


@pytest.fixture(autouse=True)
def _reset_fast_flag(monkeypatch):
    """Clear the *ODR_FAST_TEST* flag before and after each test.

    Some legacy test modules set this environment variable at import time
    which would otherwise persist and accidentally place subsequent tests
    into fast-mode.  This fixture guarantees isolation.
    """

    monkeypatch.delenv("ODR_FAST_TEST", raising=False)
    yield
    monkeypatch.delenv("ODR_FAST_TEST", raising=False)


# ---------------------------------------------------------------------------
# Convenience fixture to enable *fast mode* for specific tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def fast_mode(monkeypatch):
    """Enable the *ODR_FAST_TEST* stub path within an individual test.

    Usage::

        def test_something(fast_mode):
            ...  # fast-mode enabled

    or with pytest markers::

        @pytest.mark.usefixtures("fast_mode")
        async def test_something():
            ...
    """

    monkeypatch.setenv("ODR_FAST_TEST", "1")
    yield
    monkeypatch.delenv("ODR_FAST_TEST", raising=False)
