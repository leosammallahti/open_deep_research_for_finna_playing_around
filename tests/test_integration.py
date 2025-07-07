"""
Comprehensive integration tests for Open Deep Research.
These tests ensure the full workflow works with different model/search combinations.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_deep_research.graph import graph
from open_deep_research.configuration import SearchAPI


# Test configurations that should work
WORKING_CONFIGURATIONS = [
    {
        "name": "GPT-3.5 + DuckDuckGo",
        "config": {
            "search_api": SearchAPI.DUCKDUCKGO,
            "planner_provider": "openai",
            "planner_model": "gpt-3.5-turbo",
            "writer_provider": "openai",
            "writer_model": "gpt-3.5-turbo",
            "number_of_queries": 1,
            "max_search_depth": 1,
            "report_structure": "Brief report"
        }
    },
    {
        "name": "Claude + None",
        "config": {
            "search_api": SearchAPI.NONE,
            "planner_provider": "anthropic",
            "planner_model": "claude-3-haiku-20240307",
            "writer_provider": "anthropic",
            "writer_model": "claude-3-haiku-20240307",
            "number_of_queries": 1,
            "max_search_depth": 1,
            "report_structure": "Brief report"
        }
    }
]

# Known problematic configurations
PROBLEMATIC_CONFIGURATIONS = [
    {
        "name": "DeepSeek + None",
        "config": {
            "search_api": SearchAPI.NONE,
            "planner_provider": "deepseek",
            "planner_model": "deepseek-chat",
            "writer_provider": "deepseek",
            "writer_model": "deepseek-chat",
            "number_of_queries": 1,
            "max_search_depth": 1,
            "report_structure": "Brief report"
        },
        "expected_error": "object Queries can't be used in 'await' expression",
        "status": "CONFIRMED_BROKEN"
    }
]


class IntegrationTestRunner:
    """Runs integration tests and tracks results."""
    
    def __init__(self):
        self.results = {
            "passed": [],
            "failed": [],
            "skipped": []
        }
        self.topic = "What is machine learning?"  # Simple test topic
    
    async def test_configuration(self, name: str, config: Dict[str, Any], 
                                expected_error: str = None) -> Dict[str, Any]:
        """Test a single configuration."""
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"Config: {config}")
        print(f"{'='*60}")
        
        result = {
            "name": name,
            "config": config,
            "status": "unknown",
            "error": None,
            "output": None,
            "duration": 0
        }
        
        try:
            import time
            start_time = time.time()
            
            # Run the graph
            output = await graph.ainvoke(
                {"topic": self.topic},
                config={"configurable": config}
            )
            
            result["duration"] = time.time() - start_time
            
            # Check output
            if output and isinstance(output, dict):
                if "final_report" in output and output["final_report"]:
                    result["status"] = "passed"
                    result["output"] = output["final_report"][:200] + "..."
                    print(f"✅ PASSED: Generated report in {result['duration']:.2f}s")
                else:
                    result["status"] = "failed"
                    result["error"] = "No final report generated"
                    print(f"❌ FAILED: No final report in output")
            else:
                result["status"] = "failed"
                result["error"] = f"Invalid output type: {type(output)}"
                print(f"❌ FAILED: {result['error']}")
                
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["duration"] = time.time() - start_time if 'start_time' in locals() else 0
            
            # Check if this is an expected error
            if expected_error and expected_error in str(e):
                print(f"⚠️  EXPECTED ERROR: {expected_error}")
                result["status"] = "expected_failure"
            else:
                print(f"❌ FAILED: {str(e)[:200]}")
        
        return result
    
    async def run_all_tests(self):
        """Run all integration tests."""
        print("\n" + "="*80)
        print("INTEGRATION TEST SUITE")
        print("="*80)
        
        # Test working configurations
        print("\n## Testing Working Configurations")
        for config in WORKING_CONFIGURATIONS:
            result = await self.test_configuration(
                config["name"], 
                config["config"]
            )
            
            if result["status"] == "passed":
                self.results["passed"].append(result)
            elif result["status"] == "skipped":
                self.results["skipped"].append(result)
            else:
                self.results["failed"].append(result)
        
        # Test problematic configurations
        print("\n## Testing Known Problematic Configurations")
        for config in PROBLEMATIC_CONFIGURATIONS:
            result = await self.test_configuration(
                config["name"],
                config["config"],
                config.get("expected_error")
            )
            
            # For known broken configs, expected_failure is actually a pass
            if result["status"] == "expected_failure":
                self.results["passed"].append(result)
            elif result["status"] == "skipped":
                self.results["skipped"].append(result)
            else:
                self.results["failed"].append(result)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate a test report."""
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        
        total = len(self.results["passed"]) + len(self.results["failed"]) + len(self.results["skipped"])
        
        print(f"\nTotal Tests: {total}")
        print(f"✅ Passed: {len(self.results['passed'])}")
        print(f"❌ Failed: {len(self.results['failed'])}")
        print(f"⏭️  Skipped: {len(self.results['skipped'])}")
        
        if self.results["failed"]:
            print("\n## Failed Tests:")
            for test in self.results["failed"]:
                print(f"\n- {test['name']}")
                print(f"  Error: {test['error']}")
                print(f"  Config: {test['config']}")
        
        if self.results["skipped"]:
            print("\n## Skipped Tests (Already Fixed):")
            for test in self.results["skipped"]:
                print(f"- {test['name']}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save test results to a file."""
        import json
        from datetime import datetime
        
        results_file = "test_results.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "results": self.results,
            "summary": {
                "total": len(self.results["passed"]) + len(self.results["failed"]) + len(self.results["skipped"]),
                "passed": len(self.results["passed"]),
                "failed": len(self.results["failed"]),
                "skipped": len(self.results["skipped"])
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n📄 Results saved to: {results_file}")


def test_message_handling():
    """Test message handling with different formats."""
    from open_deep_research.utils import get_message_content
    
    print("\n## Testing Message Handling")
    
    # Test dict format
    dict_msg = {"content": "Test content", "role": "user"}
    assert get_message_content(dict_msg) == "Test content"
    print("✅ Dict message handling works")
    
    # Test object format (mock)
    class MockMessage:
        def __init__(self, content):
            self.content = content
    
    obj_msg = MockMessage("Test content")
    assert get_message_content(obj_msg) == "Test content"
    print("✅ Object message handling works")
    
    # Test invalid format
    try:
        get_message_content("invalid")
        print("❌ Invalid message handling failed to raise error")
    except TypeError:
        print("✅ Invalid message handling raises proper error")


async def main():
    """Run all tests."""
    # First run unit tests
    print("Running unit tests...")
    test_message_handling()
    
    # Then run integration tests
    print("\nRunning integration tests...")
    runner = IntegrationTestRunner()
    await runner.run_all_tests()


if __name__ == "__main__":
    # Check for API keys
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print("⚠️  Warning: Missing API keys:", missing_keys)
        print("   Some tests may fail. Add keys to .env file for full testing.")
    
    asyncio.run(main()) 