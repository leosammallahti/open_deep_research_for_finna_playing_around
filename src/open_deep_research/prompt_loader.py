from __future__ import annotations

"""Utility to load prompt templates from disk.

Markdown files are stored inside the *open_deep_research.prompt_files* package.
Call ``load_prompt("planner")`` to retrieve the *planner.md* template as a
string.
"""

import datetime
import importlib.resources as pkg_resources
from typing import Any, Dict

_PACKAGE: str = "open_deep_research.prompt_files"


def load_prompt_from_file(name: str) -> str:
    file_name = f"{name}.md"
    return pkg_resources.files(_PACKAGE).joinpath(file_name).read_text(encoding="utf-8")

def load_prompt(prompt_name: str, variables: Dict[str, Any] | None = None) -> str:
    prompt = load_prompt_from_file(prompt_name)
    if variables is None:
        variables = {}
    variables['today'] = datetime.date.today().isoformat()
    return prompt.format(**variables)