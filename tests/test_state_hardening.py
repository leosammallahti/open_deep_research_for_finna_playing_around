import re
from pathlib import Path

from open_deep_research.pydantic_state import DeepResearchState  # type: ignore


def test_deepresearchstate_is_frozen() -> None:
    """The main state model must be immutable to guarantee safe parallel use."""
    assert DeepResearchState.model_config.get("frozen") is True, (
        "DeepResearchState should be frozen"
    )


def test_model_copy_allows_updates() -> None:
    """`model_copy(update=…)` should produce a new object with the mutation applied."""
    original = DeepResearchState(topic="Original")
    mutated = original.model_copy(update={"topic": "Mutated"})

    assert original.topic == "Original"
    assert mutated.topic == "Mutated"


def test_no_dict_style_state_access() -> None:
    """Fail if new `state["foo"]` or `state.get("foo")` patterns creep into runtime code."""
    # Compile regexes once for efficiency
    patterns = [
        re.compile(r"\bstate\[\s*[\'\"]"),
        re.compile(r"\bstate\.get\(\s*[\'\"]"),
    ]

    # Files where internal helpers legitimately use Mapping semantics
    allowed_files = {
        "src/open_deep_research/graph.py",  # helper get_state_value uses Mapping
        "src/open_deep_research/pydantic_state.py",  # legacy Mapping helpers
    }

    project_root = Path(__file__).resolve().parent.parent
    search_roots = [
        project_root / "src" / "open_deep_research",
        project_root / "streamlit_app.py",
    ]
    for root in search_roots:
        if root.is_file():
            files = [root]
        else:
            files = list(root.rglob("*.py"))

        for py_file in files:
            # Skip tests, virtual environments, and allowed helper files
            if (
                "tests" in py_file.parts
                or ".venv" in py_file.parts
                or "site-packages" in py_file.parts
            ):
                continue

            rel_path = py_file.relative_to(project_root).as_posix()
            if rel_path in allowed_files:
                continue

            text = py_file.read_text(encoding="utf-8")
            for pattern in patterns:
                assert pattern.search(text) is None, (
                    f"Disallowed dict-style state access found in {rel_path}"
                )
