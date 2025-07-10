# Open Deep Research Architecture

## 🏗️ System Overview

Open Deep Research is a sophisticated research assistant with two distinct implementations:

### **Unified Typed Workflow** (`src/open_deep_research/workflow/`)
* **Pattern**: Single LangGraph with adapters; routes dynamically to *workflow* or *multi_agent* back-ends.
* **State Management**: Immutable `DeepResearchState` (Pydantic) with reducer annotations for safe parallel updates.
* **Adapter Layer**: `NodeAdapter` wraps legacy planner/researcher logic so existing code is reused without modification.
* **Execution Modes**: Select by `execution_mode` field, `RunnableConfig.configurable.mode`, or enabling `features["mcp_support"]`.
* **Feature Flags**: Parallel research, citation numbering, raw-source dump—all toggled via `WorkflowConfiguration.features`.
* **Fast/Offline CI**: `ODR_FAST_TEST=1` stubs planner/researcher to avoid network calls.

### **Workflow Implementation** (`src/open_deep_research/graph.py`)
- **Pattern**: Plan → Human Feedback → Execute → Compile
- **State Management**: LangGraph StateGraph with structured state transitions
- **Human-in-the-Loop**: Interactive plan approval with feedback cycles
- **Search Strategy**: Sequential research with reflection and iteration

### **Multi-Agent Implementation** (`src/open_deep_research/multi_agent.py`)
- **Pattern**: Supervisor coordinates parallel Researcher agents
- **State Management**: MessagesState with tool calling
- **Parallelization**: Multiple sections researched simultaneously
- **Search Strategy**: Independent research per section

## 🔄 Data Flow

### Workflow Implementation Flow:
```
Topic Input → Generate Plan → Human Feedback → [Approve/Modify]
    ↓
Parallel Section Research → Query Generation → Web Search → Content Writing
    ↓
Section Quality Check → [Pass/Iterate] → Gather Sections → Final Sections
    ↓
Compile Report → Markdown Output
```

### Multi-Agent Implementation Flow:
```
Topic Input → Supervisor Analysis → Section Planning
    ↓
Parallel Send() to Research Agents → Independent Research → Tool Calls
    ↓
Section Completion → Supervisor Assembly → Introduction/Conclusion
    ↓
Final Report Compilation
```

## 🔧 Key Components

### **State Management**
- `ReportState`: Main workflow state with sections, feedback, and progress
- `SectionState`: Individual section research state
- `MessagesState`: Multi-agent communication state

### **Configuration System**
- `WorkflowConfiguration`: Settings for graph-based implementation
- `MultiAgentConfiguration`: Settings for agent-based implementation
- Environment variable integration with `.env` support

### **Search Integration**
- `select_and_execute_search()`: Unified search interface for 9+ providers
- `get_search_params()`: Provider-specific parameter filtering
- Smart dependency management with optional providers

### **Model Integration**
- `init_chat_model()`: Universal model initialization
- Support for OpenAI, Anthropic, Groq, DeepSeek, and others
- Structured output generation for sections and plans

## 🧩 Core Modules

### **`graph.py`** - Workflow Implementation
- **Nodes**: `generate_report_plan`, `human_feedback`, `build_section_with_web_research`
- **Edges**: Conditional routing based on feedback and completion
- **Sub-graphs**: Section research as independent workflow

### **`multi_agent.py`** - Multi-Agent Implementation  
- **Agents**: `supervisor` (coordination), `research_agent` (section research)
- **Tools**: Dynamic tool loading based on configuration
- **Communication**: Message passing with tool call results

### **`utils.py`** - Shared Utilities
- Search provider implementations
- Message handling and context management
- Result formatting and processing

### **`prompts.py`** - LLM Instructions
- System prompts for planning, research, and writing
- Configurable templates with dynamic parameters
- Model-specific optimizations

### **`configuration.py`** - Settings Management
- Dataclass-based configuration
- Environment variable integration
- Provider-specific settings

## 🔍 Search Provider Architecture

### **Unified Interface**
All search providers implement a common interface through `select_and_execute_search()`:

```python
async def select_and_execute_search(
    search_api: str,
    queries: List[str], 
    params: Dict[str, Any]
) -> str
```

### **Provider Types**
- **Web Search**: Tavily, DuckDuckGo, Google, Perplexity
- **Academic**: ArXiv, PubMed  
- **Semantic**: Exa
- **Enterprise**: Azure AI Search
- **Real-time**: Linkup

### **Smart Dependencies**
- Optional dependencies with graceful fallbacks
- Runtime provider detection
- Helpful error messages with installation instructions

## 🎛️ Configuration Flow

### **Environment Variables**
- API keys loaded from `.env` file
- Model selection via environment variables
- Search provider configuration

### **Runtime Configuration**
- LangGraph `RunnableConfig` for dynamic settings
- Configuration classes with defaults
- Parameter validation and filtering

## 🔄 Error Handling

### **Robust Error Management**
- Dependency validation with helpful messages
- Model-specific error handling
- Search provider fallbacks
- Context length management

### **Debugging Support**
- Comprehensive logging
- State inspection capabilities
- Progress tracking
- Performance monitoring

## 🧪 Testing Architecture

### **Two-Tier Testing**
1. **Pytest System**: Binary pass/fail for CI/CD
2. **LangSmith Evaluate**: Detailed quality assessment

### **Test Coverage**
- Unit tests for individual components
- Integration tests for full workflows
- Model compatibility tests
- Search provider validation

## 📊 UI Architecture

### **Streamlit Interface**
- `streamlit_app.py`: Main UI application
- Real-time progress tracking
- Dynamic provider selection
- Configuration management
- Report download functionality

### **User Experience**
- Provider status detection
- Helpful error messages
- Progress indicators
- Modern, responsive design

## 🔗 Integration Points

### **LangGraph Integration**
- State management with type safety
- Conditional routing and branching
- Parallel execution with Send() API
- Human-in-the-loop interrupts

### **LangChain Integration**
- Universal model initialization
- Tool calling and structured outputs
- Message handling and formatting
- Provider abstraction

### **MCP (Model Context Protocol)**
- External tool integration
- Local file system access
- Database connections
- API integrations

## 🚀 Deployment Options

### **Local Development**
- LangGraph server with Studio UI
- Streamlit web application
- Jupyter notebook examples

### **Production Deployment**
- LangGraph Platform support
- Containerized deployment
- Environment configuration
- API endpoint access

## 📈 Performance Considerations

### **Optimization Strategies**
- Parallel section research
- Context length management
- Search result summarization
- Model selection optimization

### **Resource Management**
- Token usage tracking
- Rate limiting compliance
- Memory-efficient processing
- Async/await patterns

This architecture enables flexible, scalable research workflows while maintaining code quality and user experience. 