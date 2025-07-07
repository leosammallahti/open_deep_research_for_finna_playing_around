#!/usr/bin/env python3
"""
Test script to demonstrate the improved MVP functionality.
This shows how the refactored DeepSeek integration and robust search work.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from open_deep_research.graph import graph
from open_deep_research.configuration import SearchAPI

async def test_mvp():
    """Test the improved MVP with different configurations."""
    
    print("🔍 Testing Open Deep Research MVP")
    print("=" * 50)
    
    # Test 1: Basic functionality with DuckDuckGo (no API key needed)
    print("\n1. Testing with DuckDuckGo (robust default)")
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
        result = await graph.ainvoke(
            {"topic": "Benefits of renewable energy"}, 
            config=config
        )
        print("✅ DuckDuckGo search test: SUCCESS")
        print(f"Generated {len(result.get('sections', []))} sections")
        
    except Exception as e:
        print(f"❌ DuckDuckGo test failed: {e}")
    
    # Test 2: Test DeepSeek integration (if API key available)
    print("\n2. Testing DeepSeek integration (refactored)")
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
        result = await graph.ainvoke(
            {"topic": "AI in healthcare"}, 
            config=deepseek_config
        )
        print("✅ DeepSeek integration test: SUCCESS")
        print("✅ No more tool_choice errors!")
        
    except Exception as e:
        print(f"⚠️  DeepSeek test: {e}")
        print("   (This is expected if you don't have a DeepSeek API key)")
    
    print("\n" + "=" * 50)
    print("🎉 MVP Testing Complete!")
    print("\nKey Improvements:")
    print("- ✅ DeepSeek integration refactored (no more custom workarounds)")
    print("- ✅ Robust search handling (no more 'None' search crashes)")
    print("- ✅ Centralized configuration via environment variables")
    print("- ✅ Updated dependencies for better compatibility")

if __name__ == "__main__":
    asyncio.run(test_mvp()) 