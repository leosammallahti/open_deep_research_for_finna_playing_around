# Open Deep Research Implementation Guide

Open Deep Research provides three different implementations for generating research reports. Each has its own strengths and use cases.

## 🎯 Quick Decision Guide

| If you need... | Use this implementation |
|----------------|------------------------|
| Standard research workflow with human approval | **graph.py** |
| AI to ask clarifying questions before starting | **workflow.py** |
| Parallel research with autonomous agents | **multi_agent.py** |

## 📊 Detailed Comparison

### 1. Graph Implementation (`graph.py`)
**When to use:**
- You want a structured, predictable workflow
- Human approval of the report plan is important
- You prefer sequential research with reflection
- You need fine control over each research step

**Key features:**
- Plan → Human Feedback → Research → Write workflow
- Reflection and iteration on each section
- Supports all search providers
- Clean separation between planning and execution

**Example use case:**
```python
from open_deep_research.graph import graph

result = await graph.ainvoke(
    {"topic": "AI safety research"},
    {"configurable": {
        "search_api": "tavily",
        "planner_model": "claude-3-5-sonnet",
        "writer_model": "claude-3-5-sonnet"
    }}
)
```

### 2. Workflow Implementation (`workflow.py`)
**When to use:**
- You want the AI to clarify requirements before starting
- The research topic might be ambiguous
- You need interactive clarification features
- Message-based interaction is preferred

**Key features:**
- `clarify_with_user` feature for asking questions
- Message-based state management
- Similar to graph.py but with clarification step
- Supports user approval of sections

**Unique capability:**
```python
# The AI can ask clarifying questions like:
# "Should I focus on technical implementation details or high-level business benefits?"
```

**Example use case:**
```python
# In langgraph.json, it's available as "odr_workflow_v2"
result = await workflow.ainvoke(
    {"messages": [{"role": "user", "content": "Research vibe coding"}]},
    {"configurable": {
        "clarify_with_user": True,  # Enable clarification
        "sections_user_approval": True
    }}
)
```

### 3. Multi-Agent Implementation (`multi_agent.py`)
**When to use:**
- You want maximum parallelization
- Autonomous agent behavior is preferred
- Tool-based architecture fits your needs
- You're comfortable with less predictable workflows

**Key features:**
- Supervisor-researcher agent pattern
- Parallel section research
- Tool-calling based architecture
- MCP (Model Context Protocol) support
- Dynamic tool loading

**Example use case:**
```python
from open_deep_research.multi_agent import graph

result = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "Research quantum computing"}]},
    {"configurable": {
        "supervisor_model": "anthropic:claude-3-5-sonnet",
        "researcher_model": "anthropic:claude-3-5-sonnet",
        "search_api": "tavily"
    }}
)
```

## 🔄 Migration Between Implementations

### From workflow.py to graph.py
If you're using workflow.py but don't need the clarification feature:
1. Switch to graph.py (it's the main implementation)
2. Remove `clarify_with_user` configuration
3. Change from message-based to topic-based input

### Adding clarification to graph.py
If you need clarification in graph.py:
1. Use workflow.py instead
2. Enable `clarify_with_user: true`
3. Handle the message-based state

## 🏗️ Architecture Differences

### State Management
- **graph.py**: `DeepResearchState` with `topic` field
- **workflow.py**: `DeepResearchState` with `messages` field
- **multi_agent.py**: `MessagesState` with tool calls

### Execution Pattern
- **graph.py**: Sequential with controlled parallelism
- **workflow.py**: Sequential with clarification step
- **multi_agent.py**: Fully parallel agent execution

### Configuration
- **graph.py**: `WorkflowConfiguration`
- **workflow.py**: `WorkflowConfiguration` (different defaults)
- **multi_agent.py**: `MultiAgentConfiguration`

## 📝 Recommendations

1. **Start with graph.py** - It's the main implementation with the most features
2. **Use workflow.py** if you need the clarification feature
3. **Use multi_agent.py** for maximum parallelism and agent autonomy
4. **All implementations** share core utilities to reduce maintenance

## 🚀 Future Roadmap

- **Short term**: Port `clarify_with_user` feature to graph.py as optional
- **Long term**: Potentially merge graph.py and workflow.py with feature flags
- **Always separate**: multi_agent.py (different paradigm) 