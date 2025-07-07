#!/usr/bin/env python3
"""
Quick test of the improved MVP with real API keys.
"""
import asyncio
import sys
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from open_deep_research.graph import graph
from open_deep_research.configuration import SearchAPI

async def quick_mvp_test():
    """Quick test of the improved MVP."""
    
    print("🔍 Testing Your Improved MVP")
    print("=" * 50)
    
    # Test 1: Tavily search with Anthropic models (your default config)
    print("\n✨ Testing with your API keys:")
    print("   - Search: Tavily")
    print("   - Models: Claude 3.5 Sonnet")
    
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
        print("\n🚀 Running research on 'Benefits of AI in education'...")
        result = await graph.ainvoke(
            {"topic": "Benefits of AI in education"}, 
            config=config
        )
        
        print("✅ SUCCESS! Your MVP is working!")
        print(f"📝 Generated {len(result.get('sections', []))} sections")
        
        if 'final_report' in result:
            report_preview = result['final_report'][:200] + "..." if len(result['final_report']) > 200 else result['final_report']
            print(f"\n📄 Report preview:\n{report_preview}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you've created the .env file with your API keys")
    
    # Test 2: DeepSeek integration (now fixed!)
    print("\n" + "-" * 50)
    print("🧠 Testing DeepSeek integration (refactored)...")
    
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
            config=deepseek_config
        )
        print("✅ DeepSeek test: SUCCESS!")
        print("🎉 No more tool_choice errors!")
        
    except Exception as e:
        print(f"⚠️  DeepSeek test: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 MVP Test Complete!")
    print("\n🚀 Your Streamlit app should be running at: http://localhost:8501")
    print("   Try it out with different models and search providers!")

if __name__ == "__main__":
    asyncio.run(quick_mvp_test()) 