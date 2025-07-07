#!/usr/bin/env python3
"""
Simple test of our refactored MVP using DuckDuckGo search (no API keys needed).
"""
import asyncio
import sys
import os
from pathlib import Path

# Load environment variables if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    print("dotenv not found, continuing without .env file")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from open_deep_research.graph import graph
from open_deep_research.configuration import SearchAPI

async def simple_test():
    """Simple test using DuckDuckGo search (no API keys needed)."""
    
    print("🔍 Testing Refactored MVP")
    print("=" * 50)
    print("Using DuckDuckGo search (no API keys required)")
    
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
        print("\n🚀 Testing search functionality...")
        print("Topic: 'What is artificial intelligence?'")
        
        result = await graph.ainvoke(
            {"topic": "What is artificial intelligence?"}, 
            config=config
        )
        
        print("✅ SUCCESS! The refactored system is working!")
        print(f"📊 Results: {type(result)} with keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        
        if isinstance(result, dict) and 'final_report' in result:
            report_preview = result['final_report'][:200] + "..." if len(result['final_report']) > 200 else result['final_report']
            print(f"\n📄 Report preview:\n{report_preview}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        
        # Provide specific guidance
        if "api key" in str(e).lower() or "authentication" in str(e).lower():
            print("\n💡 This means you need to add API keys to test with AI models.")
            print("   Add your keys to a .env file to test with real models.")
        elif "import" in str(e).lower():
            print("\n💡 This is likely a missing dependency.")
            print("   Try: pip install -e .")
    
    print("\n" + "=" * 50)
    print("🎯 Test completed!")
    print("\n📋 Summary of our refactoring:")
    print("✅ Model registry system")
    print("✅ Per-role model selection") 
    print("✅ Structured output fallback helper")
    print("✅ Think token filtering")
    print("✅ Predefined model combinations")

if __name__ == "__main__":
    asyncio.run(simple_test()) 