#!/usr/bin/env python3
"""Quick test of the improved MVP with real API keys."""

import sys
if 'pytest' in sys.modules:
    import pytest  # type: ignore
    pytest.skip("quick_test is an example script, not a test", allow_module_level=True)

import asyncio
import sys as _sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from open_deep_research.configuration import SearchAPI
from open_deep_research.graph import graph
from open_deep_research.core.model_utils import trace_config


async def quick_mvp_test():
    """Quick test of the improved MVP."""
    # Test 1: Tavily search with Anthropic models (your default config)
    
    config = {
        "configurable": {
            "search_api": SearchAPI.TAVILY,
            "planner_provider": "anthropic",
            "planner_model": "claude-3-5-sonnet",
            "writer_provider": "anthropic", 
            "writer_model": "claude-3-5-sonnet",
            "number_of_queries": 1,  # Keep it quick
            "max_search_depth": 1
        }
    }
    
    try:
        result = await graph.ainvoke(
            {"topic": "Benefits of AI in education"}, 
            config={**config, **trace_config("quick-test-1")}
        )
        
        
        if 'final_report' in result:
            result['final_report'][:200] + "..." if len(result['final_report']) > 200 else result['final_report']
            
    except Exception:
        pass
    
    # Test 2: DeepSeek integration (now fixed!)
    
    deepseek_config = {
        "configurable": {
            "search_api": SearchAPI.TAVILY,  # Use working search
            "planner_provider": "deepseek",
            "planner_model": "deepseek-chat",
            "writer_provider": "deepseek",
            "writer_model": "deepseek-chat",
            "number_of_queries": 1,
            "max_search_depth": 1
        }
    }
    
    try:
        result = await graph.ainvoke(
            {"topic": "Future of renewable energy"}, 
            config={**deepseek_config, **trace_config("quick-test-2")}
        )
        
    except Exception:
        pass
    

if __name__ == "__main__":
    asyncio.run(quick_mvp_test()) 