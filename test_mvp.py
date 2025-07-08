#!/usr/bin/env python3
"""Test script to demonstrate the improved MVP functionality.

This shows how the refactored DeepSeek integration and robust search work.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from open_deep_research.configuration import SearchAPI
from open_deep_research.graph import graph
from open_deep_research.core.model_utils import trace_config


async def test_mvp():
    """Test the improved MVP with different configurations."""
    # Test 1: Basic functionality with DuckDuckGo (no API key needed)
    config = {
        "configurable": {
            "search_api": SearchAPI.DUCKDUCKGO,
            "planner_provider": "anthropic",
            "planner_model": "claude-3-5-haiku-latest",  # Using a smaller model for testing
            "writer_provider": "anthropic", 
            "writer_model": "claude-3-5-haiku-latest",
            "number_of_queries": 1,  # Reduced for faster testing
            "max_search_depth": 1
        }
    }
    
    try:
        # Test with a simple topic
        await graph.ainvoke(
            {"topic": "Benefits of renewable energy"}, 
            config={**config, **trace_config("mvp-test-1")}
        )
        
    except Exception:
        pass
    
    # Test 2: Test DeepSeek integration (if API key available)
    deepseek_config = {
        "configurable": {
            "search_api": SearchAPI.DUCKDUCKGO,  # Use robust default
            "planner_provider": "deepseek",
            "planner_model": "deepseek-chat",  # Using chat model instead of reasoner
            "writer_provider": "deepseek",
            "writer_model": "deepseek-chat",
            "number_of_queries": 1,
            "max_search_depth": 1
        }
    }
    
    try:
        await graph.ainvoke(
            {"topic": "AI in healthcare"}, 
            config={**deepseek_config, **trace_config("mvp-test-2")}
        )
        
    except Exception:
        pass
    

if __name__ == "__main__":
    asyncio.run(test_mvp()) 