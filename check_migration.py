#!/usr/bin/env python3
"""Check migration tracker completion status."""

from src.open_deep_research.migration import MigrationTracker


def main():
    status = MigrationTracker.status()
    print("Migration status:")
    for field_name, completion in status.items():
        print(f"  {field_name}: {completion:.1f}%")
    
    all_complete = all(completion == 100.0 for completion in status.values())
    if all_complete:
        print("\nAll fields migrated! ✅")
    else:
        print("\nSome fields still pending...")
    
    print(f"\nMigration complete: {MigrationTracker.is_complete()}")

if __name__ == "__main__":
    main() 