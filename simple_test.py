#!/usr/bin/env python3
"""Simple test of our refactored MVP using DuckDuckGo search (no API keys needed)."""
import asyncio
import sys
from pathlib import Path

# Load environment variables if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from open_deep_research.configuration import SearchAPI
from open_deep_research.graph import graph
from open_deep_research.core.model_utils import trace_config


async def simple_test():
    """Simple test using DuckDuckGo search (no API keys needed)."""
    # Test with DuckDuckGo search - no API keys needed
    config = {
        "configurable": {
            "search_api": SearchAPI.DUCKDUCKGO,
            "planner_provider": "anthropic",  # Will fail gracefully if no API key
            "planner_model": "claude-3-5-sonnet-20240620",
            "writer_provider": "anthropic", 
            "writer_model": "claude-3-5-sonnet-20240620",
            "number_of_queries": 1,  # Keep it quick
            "max_search_depth": 1,
            "report_structure": "Brief, concise report"
        }
    }
    
    try:
        
        result = await graph.ainvoke(
            {"topic": "What is artificial intelligence?"}, 
            config={**config, **trace_config("simple-test")}
        )
        
        
        if isinstance(result, dict) and 'final_report' in result:
            result['final_report'][:200] + "..." if len(result['final_report']) > 200 else result['final_report']
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Provide specific guidance
        if "api key" in str(e).lower() or "authentication" in str(e).lower():
            pass
        elif "import" in str(e).lower():
            pass
    

if __name__ == "__main__":
    asyncio.run(simple_test()) 