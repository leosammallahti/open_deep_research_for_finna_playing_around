# API Reference Guide

## 🔧 Core Functions & Classes

### **Search Interface**

#### `select_and_execute_search()`
```python
async def select_and_execute_search(
    search_api: str,
    queries: List[str], 
    params: Dict[str, Any]
) -> str
```
**Purpose**: Unified interface for all search providers  
**Parameters**:
- `search_api`: Provider name ("tavily", "duckduckgo", "arxiv", "pubmed", etc.)
- `queries`: List of search queries to execute
- `params`: Provider-specific parameters (filtered by `get_search_params()`)

**Returns**: Formatted search results as string

**Example**:
```python
results = await select_and_execute_search(
    "tavily", 
    ["AI research 2024", "machine learning trends"],
    {"num_results": 5}
)
```

#### `get_search_params()`
```python
def get_search_params(search_api: str, search_api_config: Dict[str, Any]) -> Dict[str, Any]
```
**Purpose**: Filter configuration parameters for specific search providers  
**Parameters**:
- `search_api`: Provider name
- `search_api_config`: Raw configuration dictionary

**Returns**: Filtered parameters valid for the specified provider

### **Individual Search Functions**

#### `tavily_search()`
```python
@tool
async def tavily_search(query: str, num_results: int = 5) -> str
```
**Purpose**: Search using Tavily API  
**Tool**: Available as LangChain tool for agents

#### `duckduckgo_search()`
```python
@tool
async def duckduckgo_search(query: str, num_results: int = 5) -> str
```
**Purpose**: Privacy-focused web search  
**Tool**: Available as LangChain tool for agents

#### `arxiv_search()`
```python
@tool
async def arxiv_search(query: str, load_max_docs: int = 5) -> str
```
**Purpose**: Search academic papers on ArXiv  
**Tool**: Available as LangChain tool for agents

#### `pubmed_search()`
```python
@tool
async def pubmed_search(query: str, top_k_results: int = 5) -> str
```
**Purpose**: Search medical literature  
**Tool**: Available as LangChain tool for agents

## 🏗️ State Management

### **ReportState** (Workflow Implementation)
```python
class ReportState(TypedDict):
    topic: str                          # Research topic
    sections: List[Section]             # Report sections
    completed_sections: List[Section]   # Finished sections
    feedback_on_report_plan: List[str] # Human feedback
    final_report: str                   # Completed report
    source_str: str                     # Search sources (if enabled)
```

### **SectionState** (Section Research)
```python
class SectionState(TypedDict):
    topic: str                 # Main topic
    section: Section          # Current section
    search_queries: List[Query] # Generated queries
    source_str: str           # Search results
    search_iterations: int    # Iteration count
```

### **Section** (Pydantic Model)
```python
class Section(BaseModel):
    name: str         # Section title
    description: str  # Research scope
    research: bool    # Whether research is needed
    content: str      # Written content
```

## ⚙️ Configuration Classes

### **WorkflowConfiguration**
```python
@dataclass(kw_only=True)
class WorkflowConfiguration:
    # Search settings
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    number_of_queries: int = 2
    max_search_depth: int = 2
    
    # Model settings
    planner_provider: str = "anthropic"
    planner_model: str = "claude-3-7-sonnet-latest"
    writer_provider: str = "anthropic"
    writer_model: str = "claude-3-7-sonnet-latest"
    
    # Report settings
    report_structure: str = DEFAULT_REPORT_STRUCTURE
    include_source_str: bool = False
    
    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "WorkflowConfiguration"
```

### **MultiAgentConfiguration**
```python
@dataclass(kw_only=True)
class MultiAgentConfiguration:
    # Search settings
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    number_of_queries: int = 2
    
    # Agent settings
    supervisor_model: str = "anthropic:claude-3-5-sonnet"
    researcher_model: str = "anthropic:claude-3-5-sonnet"
    ask_for_clarification: bool = False
    
    # MCP settings
    mcp_server_config: Optional[Dict[str, Any]] = None
    mcp_prompt: Optional[str] = None
    mcp_tools_to_include: Optional[List[str]] = None
    
    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "MultiAgentConfiguration"
```

## 🎯 Core Workflow Functions

### **Workflow Implementation** (`graph.py`)

#### `generate_report_plan()`
```python
async def generate_report_plan(state: ReportState, config: RunnableConfig) -> Dict[str, Any]
```
**Purpose**: Create initial report plan with sections  
**Flow**: Search → Plan → Structure sections  
**Returns**: `{"sections": List[Section]}`

#### `human_feedback()`
```python
def human_feedback(state: ReportState, config: RunnableConfig) -> Command
```
**Purpose**: Get human approval/feedback on report plan  
**Flow**: Present plan → Get feedback → Route to next step  
**Returns**: Command to continue or regenerate plan

#### `write_section()`
```python
async def write_section(state: SectionState, config: RunnableConfig) -> Command
```
**Purpose**: Write section content and evaluate quality  
**Flow**: Write → Grade → Continue or iterate  
**Returns**: Command to complete or research more

### **Multi-Agent Implementation** (`multi_agent.py`)

#### `supervisor()`
```python
async def supervisor(state: ReportState, config: RunnableConfig) -> Dict[str, Any]
```
**Purpose**: Coordinate research and report assembly  
**Flow**: Plan → Delegate → Assemble  
**Returns**: `{"messages": List[BaseMessage]}`

#### `research_agent()`
```python
async def research_agent(state: SectionState, config: RunnableConfig) -> Dict[str, Any]
```
**Purpose**: Research and write individual sections  
**Flow**: Research → Write → Complete  
**Returns**: `{"messages": List[BaseMessage]}`

## 🛠️ Utility Functions

### **Message Handling**
```python
async def truncate_messages_for_context(
    messages: List[BaseMessage],
    model: str,
    max_tokens: int = 60000,
    preserve_recent: int = 3
) -> List[BaseMessage]
```
**Purpose**: Manage context length for large conversations

### **Token Management**
```python
async def count_messages_tokens(
    messages: List[BaseMessage],
    model: str
) -> int
```
**Purpose**: Count tokens in message list for specific model

### **Search Result Processing**
```python
async def summarize_search_results(
    results: str,
    max_tokens: int = 8000,
    model: str = "anthropic:claude-3-5-haiku-latest"
) -> str
```
**Purpose**: Summarize/truncate lengthy search results

## 🔧 Tool Functions

### **Section Writing Tool**
```python
class Section(BaseModel):
    name: str = Field(description="Name for this section")
    description: str = Field(description="Research scope")
    content: str = Field(description="Section content")
```

### **Report Control Tools**
```python
class FinishResearch(BaseModel):
    """Signal research completion"""

class FinishReport(BaseModel):
    """Signal report completion"""

class Question(BaseModel):
    question: str = Field(description="Clarification question")
```

## 📊 LangGraph Integration

### **Graph Building**
```python
# Workflow graph
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_edge(START, "generate_report_plan")

# Multi-agent graph
supervisor_builder = StateGraph(ReportState, input=MessagesState, output=ReportStateOutput)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("research_team", research_builder.compile())
```

### **Parallel Execution**
```python
# Send API for parallel section research
return Command(goto=[
    Send("research_team", {"section": s}) 
    for s in sections_list
])
```

## 🎨 UI Functions

### **Streamlit Interface**
```python
def main():
    """Main Streamlit application"""

def generate_report(topic: str, config: dict) -> str:
    """Generate report with progress tracking"""

def display_provider_status():
    """Show available search providers"""
```

## 📋 Usage Examples

### **Basic Workflow Usage**
```python
from open_deep_research.graph import builder

# Configure
config = {
    "configurable": {
        "search_api": "tavily",
        "planner_model": "claude-3-5-sonnet",
        "writer_model": "claude-3-5-sonnet"
    }
}

# Run
graph = builder.compile()
result = await graph.ainvoke({"topic": "AI research"}, config)
```

### **Multi-Agent Usage**
```python
from open_deep_research.multi_agent import supervisor_builder

# Configure
config = {
    "configurable": {
        "supervisor_model": "anthropic:claude-3-5-sonnet",
        "researcher_model": "anthropic:claude-3-5-sonnet",
        "search_api": "tavily"
    }
}

# Run
graph = supervisor_builder.compile()
result = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "Research AI trends"}]}, 
    config
)
```

### **Custom Search Provider**
```python
# Add new provider to utils.py
@tool
async def my_custom_search(query: str, custom_param: str = "default") -> str:
    """Custom search implementation"""
    # Implementation here
    return "search results"

# Update select_and_execute_search()
elif search_api.lower() == "custom":
    return await my_custom_search(query_list[0], **params)
```

## 🔍 Error Handling

### **Dependency Management**
```python
from open_deep_research.dependency_manager import get_status_report
status = get_status_report()  # Check provider availability
```

### **Common Patterns**
```python
# Safe configuration access
search_api = get_config_value(configurable.search_api)

# Safe parameter filtering
params = get_search_params(search_api, search_api_config or {})

# Error handling in search
try:
    results = await select_and_execute_search(api, queries, params)
except Exception as e:
    logger.error(f"Search failed: {e}")
    results = "No results available"
```

This API reference provides the essential interface information needed to understand and extend the codebase! 