# Concurrent State Management in LangGraph

## Problem Summary

When using LangGraph's `Send` API for parallel task execution, you may encounter the error:

```
Error: At key 'section': Can receive only one value per step. Use an Annotated key to handle multiple values.
```

Or for numeric fields:

```
Error: At key 'search_iterations': Can receive only one value per step. Use an Annotated key to handle multiple values.
```

This occurs when multiple parallel nodes try to update the same state field without proper annotation for handling concurrent updates.

## Root Cause

In LangGraph, when you use `Send` to create parallel tasks, each task may return state updates. If multiple tasks try to update the same field (like `section` in our case), LangGraph doesn't know how to merge these updates unless the field is properly annotated.

## Solution Implemented

We fixed this issue by:

1. **Renaming the conflicting field**: Changed from `section` to `current_section` in the `Send` command
2. **Backward compatibility**: The `write_final_sections` function checks both field names
3. **Clear separation**: Parallel nodes use a different field name than the main state's `section` field

### Code Changes

#### Before (Causing Error):
```python
def initiate_final_section_writing(state: DeepResearchState) -> List[Send]:
    tasks = [
        Send(
            "write_final_sections",
            {
                "section": s,  # Multiple nodes updating same field!
                "completed_sections": state.completed_sections,
                "topic": state.topic,
            },
        )
        for s in state.sections
        if not s.research
    ]
```

#### After (Fixed):
```python
def initiate_final_section_writing(state: DeepResearchState) -> List[Send]:
    tasks = [
        Send(
            "write_final_sections",
            {
                "current_section": s,  # Different field name, no conflicts
                "completed_sections": state.completed_sections,
                "topic": state.topic,
            },
        )
        for s in state.sections
        if not s.research
    ]
```

## Best Practices for Concurrent State Management

### 1. Use Annotated Fields for Accumulation

When multiple nodes need to add to the same collection:

```python
from typing import Annotated
import operator

class MyState(BaseModel):
    # This field can handle multiple updates
    completed_sections: Annotated[List[Section], operator.add] = Field(
        default_factory=list,
        description="Sections that accumulate from parallel nodes"
    )
```

### 2. Avoid Shared Single-Value Fields

Don't use the same single-value field across parallel nodes:

```python
# BAD: Multiple nodes updating 'current_item'
class BadState(BaseModel):
    current_item: str  # Can't handle parallel updates!

# GOOD: Use different field names or annotated fields
class GoodState(BaseModel):
    items: Annotated[List[str], operator.add]  # Can accumulate
    # OR use node-specific fields
    node_a_item: str | None = None
    node_b_item: str | None = None
```

### 3. Design State for Parallel Execution

When designing your state schema, consider:

- Which fields will be updated by parallel nodes?
- Should these updates accumulate (list/add) or replace?
- Can you use separate field names for clarity?

### 4. Handle Numeric Fields with Custom Reducers

For numeric fields that need concurrent updates, use custom reducer functions:

```python
# Define reducer functions
def _max_fn(old, new):
    """Take the maximum value."""
    return max(old, new)

def _min_fn(old, new):
    """Take the minimum value."""
    return min(old, new)

# Use in state definition
class ResearchState(BaseModel):
    # Track the highest iteration count across parallel sections
    search_iterations: Annotated[int, _max_fn] = Field(
        default=0, description="Maximum iterations performed"
    )
    
    # Track the lowest credits (most conservative) across sections
    credits_remaining: Annotated[int, _min_fn] = Field(
        default=100, description="Minimum credits remaining"
    )
```

This ensures that:
- `search_iterations` always reflects the highest iteration count from any parallel section
- `credits_remaining` always reflects the most conservative (lowest) credit count

### 5. Use Local State for Node-Specific Data

For data that's only relevant within a node:

```python
# Create a separate state class for parallel nodes
class NodeSpecificState(BaseModel):
    local_data: str  # Only used within this node
    shared_context: List[str]  # Passed from parent
```

### 6. Test Parallel Execution

Always test workflows with parallel nodes:

```python
@pytest.mark.asyncio
async def test_parallel_execution():
    """Ensure no concurrent update errors occur."""
    # Create state with multiple parallel tasks
    state = create_state_with_parallel_work()
    
    try:
        async for chunk in graph.astream(state, config):
            pass  # Process normally
    except Exception as e:
        if "Can receive only one value per step" in str(e):
            pytest.fail(f"Concurrent update error: {e}")
```

## Alternative Solutions

### Option 1: Use Annotated Fields

```python
class DeepResearchState(BaseModel):
    # Allow multiple sections to be set
    section: Annotated[Section | None, lambda x, y: y] = None  # Last write wins
    
    # For numeric fields, use appropriate reducers
    max_value: Annotated[int, max] = 0  # Track maximum
    min_value: Annotated[int, min] = 100  # Track minimum
    total_count: Annotated[int, operator.add] = 0  # Sum values
```

### Option 2: Create Separate State Classes

```python
# Main graph state
class MainState(BaseModel):
    sections: List[Section]
    completed_sections: Annotated[List[Section], operator.add]

# Parallel node state  
class ParallelNodeState(BaseModel):
    working_section: Section
    context: List[Section]
```

### Option 3: Use Sub-graphs with Isolated State

Create sub-graphs that have their own state schema, preventing conflicts with the parent graph.

## Debugging Tips

1. **Check Send Commands**: Look for multiple `Send` commands that pass the same field name
2. **Review State Updates**: Ensure parallel nodes aren't updating non-annotated fields
3. **Use Logging**: Add logging to track which nodes are updating which fields
4. **Test Early**: Run parallel workflows early in development to catch issues

## Conclusion

The concurrent state update issue is a common pitfall when using LangGraph's parallel execution features. By following these guidelines and using proper state field design, you can build robust workflows that leverage parallelism without encountering state conflicts.

Remember: **If multiple nodes update it, annotate it!** 