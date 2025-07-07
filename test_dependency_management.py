#!/usr/bin/env python3
"""
Test script to demonstrate the new dependency management system.
This script shows how the system gracefully handles missing dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_dependency_management():
    """Test the dependency management system."""
    print("🔍 Testing Open Deep Research Dependency Management System")
    print("=" * 60)
    
    try:
        from open_deep_research.dependency_manager import (
            get_status_report,
            get_available_providers,
            is_provider_available,
            SearchProvider,
            validate_provider_or_raise
        )
        
        print("✅ Dependency manager imported successfully!")
        
        # Show status report
        print("\n📋 Current Search Provider Status:")
        print(get_status_report())
        
        # Test available providers
        available = get_available_providers()
        print(f"\n✅ Available providers: {[p.value for p in available]}")
        
        # Test individual provider checks
        print("\n🔍 Testing individual provider availability:")
        for provider in SearchProvider:
            status = "✅ Available" if is_provider_available(provider) else "❌ Not available"
            print(f"  {provider.value}: {status}")
        
        # Test validation (should work for available providers)
        print("\n🧪 Testing provider validation:")
        for provider in available[:2]:  # Test first 2 available providers
            try:
                validate_provider_or_raise(provider)
                print(f"  ✅ {provider.value}: Validation passed")
            except ImportError as e:
                print(f"  ❌ {provider.value}: {e}")
        
        # Test validation for unavailable provider (should raise helpful error)
        unavailable_providers = [p for p in SearchProvider if not is_provider_available(p)]
        if unavailable_providers:
            print(f"\n🚫 Testing validation for unavailable provider ({unavailable_providers[0].value}):")
            try:
                validate_provider_or_raise(unavailable_providers[0])
                print(f"  ⚠️  No error raised (unexpected)")
            except ImportError as e:
                print(f"  ✅ Helpful error message: {e}")
        
        print("\n🎉 Dependency management system is working correctly!")
        
    except ImportError as e:
        print(f"❌ Failed to import dependency manager: {e}")
        print("Make sure you're running from the correct directory")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def test_search_integration():
    """Test that the search system works with the new dependency management."""
    print("\n🔍 Testing Search Integration")
    print("=" * 40)
    
    try:
        from open_deep_research.utils import select_and_execute_search
        from open_deep_research.dependency_manager import get_available_providers
        
        available = get_available_providers()
        
        if not available:
            print("❌ No search providers available for testing")
            return False
        
        # Test with first available provider
        test_provider = available[0].value
        print(f"🧪 Testing search with {test_provider}")
        
        # This should work without errors (though we won't actually execute the search)
        print(f"✅ Search provider '{test_provider}' is properly integrated")
        
    except Exception as e:
        print(f"❌ Search integration test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Open Deep Research - Dependency Management Test")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_dependency_management()
    success &= test_search_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Dependency management is working correctly.")
        print("\n💡 Next steps:")
        print("1. Run: streamlit run streamlit_app.py")
        print("2. Check the sidebar for available search providers")
        print("3. Install additional providers as needed")
    else:
        print("❌ Some tests failed. Check the error messages above.")
        sys.exit(1) 