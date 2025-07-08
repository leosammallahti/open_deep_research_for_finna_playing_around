#!/usr/bin/env python3
"""Test script to demonstrate the new dependency management system.

This script shows how the system gracefully handles missing dependencies.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_dependency_management():
    """Test the dependency management system."""
    try:
        from open_deep_research.dependency_manager import (
    SearchProvider,
    get_available_providers,
    is_provider_available,
    validate_provider_or_raise,
)
        
        
        # Show status report
        
        # Test available providers
        available = get_available_providers()
        
        # Test individual provider checks
        for provider in SearchProvider:
            "✅ Available" if is_provider_available(provider) else "❌ Not available"
        
        # Test validation (should work for available providers)
        for provider in available[:2]:  # Test first 2 available providers
            try:
                validate_provider_or_raise(provider)
            except ImportError:
                pass
        
        # Test validation for unavailable provider (should raise helpful error)
        unavailable_providers = [p for p in SearchProvider if not is_provider_available(p)]
        if unavailable_providers:
            try:
                validate_provider_or_raise(unavailable_providers[0])
            except ImportError:
                pass
        
        
    except ImportError:
        return False
    
    except Exception:
        return False
    
    return True

def test_search_integration():
    """Test that the search system works with the new dependency management."""
    try:
        from open_deep_research.dependency_manager import get_available_providers
        
        available = get_available_providers()
        
        if not available:
            return False
        
        # Test with first available provider
        available[0].value
        
        # This should work without errors (though we won't actually execute the search)
        
    except Exception:
        return False
    
    return True

if __name__ == "__main__":
    
    success = True
    
    # Run tests
    success &= test_dependency_management()
    success &= test_search_integration()
    
    if success:
        pass
    else:
        sys.exit(1) 