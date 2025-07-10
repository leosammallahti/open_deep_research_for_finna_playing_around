#!/usr/bin/env python3
"""Demo of the improved MVP - shows configuration and improvements."""

import sys
from pathlib import Path

# Validate settings early to surface configuration errors when running the demo.
from open_deep_research.core.settings import settings

settings.validate_all()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def demo_improvements():
    """Demonstrate the key improvements made to the MVP."""
    # Show configuration improvements
    try:
        from open_deep_research.configuration import SearchAPI

        # Show available search APIs
        for api in SearchAPI:
            pass

    except ImportError:
        pass

    # Show dependency improvements - testing availability
    try:
        __import__("aiohttp")
    except ImportError:
        pass

    try:
        __import__("tiktoken")
    except ImportError:
        pass

    try:
        __import__("langchain_deepseek")
    except ImportError:
        pass


if __name__ == "__main__":
    demo_improvements()
