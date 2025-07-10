#!/usr/bin/env python3
"""Simple test script to verify the Open Deep Research fixes.

This script tests the key fixes applied to resolve graph execution issues.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


def check_environment():
    """Check if environment is properly configured."""
    print("🔍 Checking Environment Configuration...")

    # Check API keys
    required_keys = {
        "ANTHROPIC_API_KEY": "Anthropic (for AI models)",
        "LANGSMITH_API_KEY": "LangSmith (for tracing)",
    }

    missing_keys = []
    for key, description in required_keys.items():
        if os.getenv(key):
            print(f"  ✅ {key}: Set")
        else:
            print(f"  ❌ {key}: Not set ({description})")
            missing_keys.append(key)

    # Check optional keys
    optional_keys = {
        "TAVILY_API_KEY": "Tavily search",
        "OPENAI_API_KEY": "OpenAI (fallback)",
    }

    for key, description in optional_keys.items():
        if os.getenv(key):
            print(f"  ✅ {key}: Set ({description})")
        else:
            print(f"  ⚠️  {key}: Not set ({description}) - optional")

    # Check LangSmith configuration
    print("\n🔍 LangSmith Configuration:")
    langsmith_config = {
        "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY"),
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "false"),
        "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT", "default"),
        "LANGSMITH_ENDPOINT": os.getenv(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        ),
    }

    for key, value in langsmith_config.items():
        if key == "LANGSMITH_API_KEY":
            status = "✅ Set" if value else "❌ Not set"
        else:
            status = f"✅ {value}"
        print(f"  {key}: {status}")

    return len(missing_keys) == 0


def test_model_initialization():
    """Test that model initialization works without max_retries error."""
    print("\n🧪 Testing Model Initialization...")

    try:
        from open_deep_research.core.model_utils import initialize_model

        # Test with minimal parameters (should not fail with max_retries=None)
        model = initialize_model(
            provider="anthropic",
            model_name="claude-3-5-sonnet-20240620",
            model_kwargs={},
        )

        print("  ✅ Model initialization successful")
        return True
    except Exception as e:
        print(f"  ❌ Model initialization failed: {e}")
        return False


def test_state_models():
    """Test that state models work correctly."""
    print("\n🧪 Testing State Models...")

    try:
        from open_deep_research.pydantic_state import (
            DeepResearchState,
            SectionResearchState,
            Section,
        )

        # Test DeepResearchState
        state = DeepResearchState(
            topic="Test topic",
            sections=[Section(name="Introduction", description="Intro", research=True)],
        )
        print("  ✅ DeepResearchState creation successful")

        # Test SectionResearchState (should require topic)
        section_state = SectionResearchState(
            topic="Test topic",  # This should be required
            section=Section(name="Test", description="Test", research=True),
        )
        print("  ✅ SectionResearchState creation successful")

        return True
    except Exception as e:
        print(f"  ❌ State model test failed: {e}")
        return False


async def test_simple_graph():
    """Test a simple graph execution."""
    print("\n🧪 Testing Simple Graph Execution...")

    try:
        from open_deep_research.graph import graph
        from open_deep_research.configuration import SearchAPI

        # Simple test input
        test_input = {"topic": "What is Python programming?"}

        # Configuration
        config = {
            "configurable": {
                "search_api": SearchAPI.DUCKDUCKGO,  # No API key needed
                "planner_provider": "anthropic",
                "planner_model": "claude-3-5-sonnet-20240620",
                "writer_provider": "anthropic",
                "writer_model": "claude-3-5-sonnet-20240620",
                "number_of_queries": 1,
                "max_search_depth": 1,
                "search_budget": 10,
                "report_structure": "Brief report with 2 sections",
                "include_source_str": False,
            }
        }

        # Test graph execution with timeout
        print("  📡 Running graph execution (this may take a minute)...")

        try:
            result = await asyncio.wait_for(
                graph.ainvoke(test_input, config=config),
                timeout=120,  # 2 minute timeout
            )

            if isinstance(result, dict) and "final_report" in result:
                report = result["final_report"]
                if report and len(report) > 100:
                    print(
                        f"  ✅ Graph execution successful - Generated report ({len(report)} chars)"
                    )
                    print(f"  📄 Report preview: {report[:200]}...")
                    return True
                else:
                    print(
                        f"  ⚠️  Graph completed but report seems empty or short ({len(report)} chars)"
                    )
                    return False
            else:
                print(
                    f"  ❌ Graph execution failed - unexpected result: {type(result)}"
                )
                return False

        except asyncio.TimeoutError:
            print(
                "  ⚠️  Graph execution timed out (this might be normal for slow models)"
            )
            return False

    except Exception as e:
        print(f"  ❌ Graph execution failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🔧 Open Deep Research - Testing Fixes")
    print("=" * 50)

    # Check environment
    env_ok = check_environment()
    if not env_ok:
        print("\n⚠️  Environment check failed. Please check your .env file.")
        print("See DEBUGGING_FIXES.md for required environment variables.")
        return

    # Test model initialization
    model_ok = test_model_initialization()
    if not model_ok:
        print("\n❌ Model initialization test failed. Check your API keys.")
        return

    # Test state models
    state_ok = test_state_models()
    if not state_ok:
        print("\n❌ State model test failed. Check the fixes are applied.")
        return

    # Test graph execution
    graph_ok = await test_simple_graph()

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)

    results = {
        "Environment Configuration": "✅" if env_ok else "❌",
        "Model Initialization": "✅" if model_ok else "❌",
        "State Models": "✅" if state_ok else "❌",
        "Graph Execution": "✅" if graph_ok else "❌",
    }

    for test, status in results.items():
        print(f"  {status} {test}")

    if all([env_ok, model_ok, state_ok, graph_ok]):
        print(
            "\n🎉 All tests passed! Your Open Deep Research should be working correctly."
        )
        print("You can now run: streamlit run streamlit_app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        print("See DEBUGGING_FIXES.md for troubleshooting guidance.")


if __name__ == "__main__":
    asyncio.run(main())
