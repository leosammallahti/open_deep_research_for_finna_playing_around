# Development Guide

## 🚀 Getting Started

### **Development Setup**
```bash
# Clone repository
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

### **Running Tests**
```bash
# Run basic tests
python tests/run_test.py --all

# Run specific implementation
python tests/run_test.py --agent multi_agent

# Run with specific model
python tests/run_test.py --agent graph --planner-model "claude-3-5-sonnet"
```

## 🔧 Common Development Tasks

### **1. Adding a New Search Provider**

#### **Step 1: Implement the Search Function**
Add to `src/open_deep_research/utils.py`:
```python
@tool
async def my_search(query: str, custom_param: str = "default") -> str:
    """Search using MyAPI"""
    try:
        # Implementation here
        response = await my_api_client.search(query, custom_param)
        return format_search_results(response)
    except Exception as e:
        return f"Search failed: {str(e)}"
```

#### **Step 2: Update the Search Router**
In `select_and_execute_search()`:
```python
elif search_api.lower() == "myapi":
    return await my_search(query_list[0], **params)
```

#### **Step 3: Add to Configuration**
Update `src/open_deep_research/configuration.py`:
```python
class SearchAPI(Enum):
    # ... existing providers
    MYAPI = "myapi"
```

#### **Step 4: Add Dependencies**
Update `pyproject.toml`:
```toml
[project.optional-dependencies]
myapi = ["myapi-python>=1.0.0"]

all-search = [
    # ... existing deps
    "myapi-python>=1.0.0",
]
```

#### **Step 5: Update Dependency Manager**
Add to `src/open_deep_research/dependency_manager.py`:
```python
SEARCH_PROVIDERS = {
    # ... existing providers
    "myapi": {
        "display_name": "MyAPI",
        "description": "Custom search provider",
        "dependencies": ["myapi"],
        "import_names": ["myapi"],
        "requires_api_key": True,
        "api_key_name": "MYAPI_API_KEY"
    }
}
```

### **2. Adding New AI Models**

#### **Model Configuration**
Models are handled through LangChain's `init_chat_model()`. To add support:

```python
# In configuration classes, add new model options
supervisor_model: str = "newprovider:model-name"
researcher_model: str = "newprovider:model-name"
```

#### **Model-Specific Handling**
For models with special requirements, update the implementations:
```python
# In multi_agent.py or graph.py
if model_name == "special-model":
    # Special handling
    llm = init_chat_model(
        model=model_name,
        special_param=True,
        max_tokens=8000
    )
```

### **3. Modifying Report Structure**

#### **Update State Definitions**
Modify `src/open_deep_research/state.py`:
```python
class Section(BaseModel):
    name: str
    description: str
    research: bool
    content: str
    # Add new fields
    priority: Optional[int] = None
    estimated_length: Optional[int] = None
```

#### **Update Prompts**
Modify `src/open_deep_research/prompts.py`:
```python
report_planner_instructions = """
Generate a report with the following structure:
1. Title and Executive Summary
2. {number_of_sections} main sections
3. Conclusion with key findings
4. Recommendations (new!)

Each section should include:
- Priority level (1-5)
- Estimated word count
...
"""
```

#### **Update Workflow Logic**
Modify the planning and writing functions to handle new fields:
```python
# In graph.py or multi_agent.py
def process_section(section: Section):
    if section.priority and section.priority > 3:
        # High priority sections get more research
        number_of_queries = 5
    else:
        number_of_queries = 2
```

### **4. Adding New Tools**

#### **For Multi-Agent Implementation**
Add to `src/open_deep_research/multi_agent.py`:
```python
class MyCustomTool(BaseModel):
    """Description of what the tool does"""
    parameter: str = Field(description="Parameter description")

# Add to appropriate tool lists
async def get_supervisor_tools(config: RunnableConfig) -> list[BaseTool]:
    tools = [tool(Sections), tool(MyCustomTool)]  # Add here
    # ... rest of function
```

#### **For Workflow Implementation**
Add to `src/open_deep_research/graph.py`:
```python
async def my_custom_node(state: ReportState, config: RunnableConfig):
    """Custom workflow node"""
    # Implementation
    return {"updated_field": "value"}

# Add to graph
builder.add_node("my_custom_node", my_custom_node)
builder.add_edge("previous_node", "my_custom_node")
```

### **5. Modifying the UI**

#### **Adding New Configuration Options**
Update `streamlit_app.py`:
```python
# Add new sidebar option
new_option = st.sidebar.selectbox(
    "New Setting",
    options=["Option1", "Option2", "Option3"],
    index=0
)

# Add to configuration
config = {
    "configurable": {
        # ... existing config
        "new_setting": new_option
    }
}
```

#### **Adding Progress Indicators**
```python
# Custom progress tracking
progress_bar = st.progress(0)
status_text = st.empty()

for i, step in enumerate(steps):
    status_text.text(f"Step {i+1}: {step}")
    progress_bar.progress((i + 1) / len(steps))
    # Process step
```

## 🧪 Testing Your Changes

### **Unit Testing**
```python
# tests/test_my_feature.py
import pytest
from open_deep_research.utils import my_new_function

def test_my_new_function():
    result = my_new_function("test input")
    assert result == "expected output"
```

### **Integration Testing**
```python
# Test full workflow
async def test_full_workflow_with_my_changes():
    config = {
        "configurable": {
            "search_api": "myapi",
            "my_new_setting": "test_value"
        }
    }
    
    from open_deep_research.multi_agent import supervisor_builder
    graph = supervisor_builder.compile()
    
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "Test topic"}]},
        config
    )
    
    assert "final_report" in result
    assert len(result["final_report"]) > 0
```

### **Running Tests**
```bash
# Test your specific changes
python tests/run_test.py --agent multi_agent --search-api myapi

# Run full test suite
python tests/run_test.py --all
```

## 🐛 Debugging Common Issues

### **1. Import Errors**
```python
# Check dependency status
from open_deep_research.dependency_manager import get_status_report
print(get_status_report())

# Safe imports
try:
    from myapi import client
    HAS_MYAPI = True
except ImportError:
    HAS_MYAPI = False
```

### **2. Context Length Issues**
```python
# Add context management
from open_deep_research.utils import truncate_messages_for_context

messages = await truncate_messages_for_context(
    messages,
    model=model_name,
    max_tokens=60000,
    preserve_recent=3
)
```

### **3. Tool Calling Issues**
```python
# Ensure tools are properly defined
@tool
async def my_tool(param: str) -> str:
    """Clear description of what the tool does"""
    return "result"

# Check tool is added to agent
tools = await get_supervisor_tools(config)
tool_names = [tool.name for tool in tools]
assert "my_tool" in tool_names
```

## 📊 Performance Optimization

### **1. Search Result Caching**
```python
# Add caching for expensive operations
from functools import lru_cache

@lru_cache(maxsize=128)
async def cached_search(query: str, provider: str) -> str:
    return await select_and_execute_search(provider, [query], {})
```

### **2. Parallel Processing**
```python
# Use Send() API for parallel execution
return Command(goto=[
    Send("process_section", {"section": section})
    for section in sections
])
```

### **3. Memory Management**
```python
# Clear large variables when done
del large_search_results
gc.collect()
```

## 📋 Code Quality Standards

### **1. Type Hints**
```python
from typing import List, Dict, Any, Optional

async def my_function(
    param1: str,
    param2: List[str],
    param3: Optional[Dict[str, Any]] = None
) -> str:
    """Function with proper type hints"""
    return "result"
```

### **2. Error Handling**
```python
async def robust_function(param: str) -> str:
    """Function with proper error handling"""
    try:
        result = await risky_operation(param)
        return result
    except SpecificError as e:
        logger.error(f"Specific error: {e}")
        return "fallback_value"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### **3. Documentation**
```python
async def well_documented_function(param: str) -> str:
    """Brief description of what the function does.
    
    Args:
        param: Description of the parameter
        
    Returns:
        Description of the return value
        
    Raises:
        SpecificError: When this specific condition occurs
    """
    return "result"
```

## 🔄 Deployment Considerations

### **1. Environment Variables**
```python
# Always provide defaults
API_KEY = os.getenv("MYAPI_API_KEY", "")
if not API_KEY:
    raise ValueError("MYAPI_API_KEY environment variable is required")
```

### **2. Configuration Validation**
```python
def validate_config(config: dict) -> None:
    """Validate configuration before use"""
    required_fields = ["search_api", "supervisor_model"]
    for field in required_fields:
        if field not in config.get("configurable", {}):
            raise ValueError(f"Missing required config field: {field}")
```

### **3. Resource Management**
```python
# Use context managers for resources
async with httpx.AsyncClient() as client:
    response = await client.get(url)
    return response.json()
```

## 🤝 Contributing Guidelines

### **1. Pre-commit Checklist**
- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)

### **2. Pull Request Process**
1. Create feature branch from `main`
2. Implement changes with tests
3. Update documentation
4. Submit PR with clear description
5. Address review feedback

### **3. Code Review Focus**
- Functionality correctness
- Error handling robustness
- Performance implications
- Security considerations
- Documentation completeness

This guide provides the foundation for extending and maintaining the Open Deep Research system! 