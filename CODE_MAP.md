# Code Map - Quick Reference Guide

## 🗺️ Finding Functionality Fast

### **🔍 Search & Research**
| What | Where | Key Function |
|------|-------|--------------|
| **Search Provider Interface** | `src/open_deep_research/utils.py` | `select_and_execute_search()` |
| **Tavily Search** | `src/open_deep_research/utils.py` | `tavily_search()` |
| **DuckDuckGo Search** | `src/open_deep_research/utils.py` | `duckduckgo_search()` |
| **ArXiv Search** | `src/open_deep_research/utils.py` | `arxiv_search()` |
| **PubMed Search** | `src/open_deep_research/utils.py` | `pubmed_search()` |
| **Exa Search** | `src/open_deep_research/utils.py` | `exa_search()` |
| **Search Parameter Filtering** | `src/open_deep_research/utils.py` | `get_search_params()` |

### **🤖 Core Implementations**
| What | Where | Key Components |
|------|-------|----------------|
| **Workflow Implementation** | `src/open_deep_research/graph.py` | `generate_report_plan()`, `human_feedback()`, `write_section()` |
| **Multi-Agent Implementation** | `src/open_deep_research/multi_agent.py` | `supervisor()`, `research_agent()` |
| **State Definitions** | `src/open_deep_research/state.py` | `ReportState`, `SectionState`, `Queries` |
| **Configuration** | `src/open_deep_research/configuration.py` | `WorkflowConfiguration`, `MultiAgentConfiguration` |

### **🎯 Key Functions by Task**

#### **Report Generation**
```python
# Workflow approach
src/open_deep_research/graph.py:
- generate_report_plan()      # Creates initial plan
- human_feedback()           # Gets user approval
- write_section()            # Writes individual sections
- compile_final_report()     # Assembles final report

# Multi-agent approach  
src/open_deep_research/multi_agent.py:
- supervisor()               # Coordinates research
- research_agent()           # Researches sections
- supervisor_tools()         # Handles tool calls
```

#### **Search Operations**
```python
src/open_deep_research/utils.py:
- select_and_execute_search()  # Main search interface
- get_search_params()         # Provider-specific params
- process_search_results()    # Format results
- summarize_search_results()  # Truncate long results
```

#### **Model Integration**
```python
src/open_deep_research/configuration.py:
- WorkflowConfiguration.from_runnable_config()
- MultiAgentConfiguration.from_runnable_config()

# Model initialization uses LangChain's init_chat_model()
```

#### **UI Components**
```python
streamlit_app.py:
- main()                     # Main app interface
- generate_report()          # Report generation UI
- display_provider_status()  # Show available providers
```

### **📁 Directory Structure**
```
src/open_deep_research/
├── __init__.py              # Package initialization
├── graph.py                 # Workflow implementation  
├── multi_agent.py           # Multi-agent implementation
├── state.py                 # State definitions
├── configuration.py         # Settings management
├── utils.py                 # Search & utility functions
├── prompts.py               # LLM instructions
├── dependency_manager.py    # Smart dependency handling
└── workflow/                # Alternative workflow (legacy)
    ├── configuration.py
    ├── prompts.py
    ├── state.py
    └── workflow.py
```

### **🔧 Configuration & Setup**
| Component | File | Purpose |
|-----------|------|---------|
| **Dependencies** | `pyproject.toml` | Package dependencies with optional extras |
| **Environment** | `.env` | API keys and configuration |
| **LangGraph Config** | `langgraph.json` | LangGraph server settings |
| **Dependency Management** | `src/open_deep_research/dependency_manager.py` | Smart import handling |

### **🧪 Testing & Evaluation**
| Component | File | Purpose |
|-----------|------|---------|
| **Test Runner** | `tests/run_test.py` | Main testing interface |
| **Quality Tests** | `tests/test_report_quality.py` | Report quality validation |
| **Evaluators** | `tests/evals/evaluators.py` | Detailed evaluation functions |
| **Evaluation Runner** | `tests/evals/run_evaluate.py` | LangSmith evaluation system |

### **🎨 User Interface**
| Component | File | Purpose |
|-----------|------|---------|
| **Main UI** | `streamlit_app.py` | Primary Streamlit interface |
| **Examples** | `examples/` | Usage examples for different scenarios |
| **Notebooks** | `src/open_deep_research/graph.ipynb` | Jupyter notebook examples |
| **Notebooks** | `src/open_deep_research/multi_agent.ipynb` | Multi-agent notebook |

## 🔍 Common Tasks & Where to Find Them

### **Adding a New Search Provider**
1. Add provider function to `src/open_deep_research/utils.py`
2. Update `select_and_execute_search()` with new provider
3. Add dependency to `pyproject.toml` optional extras
4. Update `dependency_manager.py` with new provider info

### **Modifying Report Structure**
1. Update prompts in `src/open_deep_research/prompts.py`
2. Modify state definitions in `src/open_deep_research/state.py`
3. Update configuration in `src/open_deep_research/configuration.py`

### **Changing Model Configuration**
1. Update configuration classes in `src/open_deep_research/configuration.py`
2. Modify model initialization in implementations
3. Update environment variable handling

### **Adding New Tools**
1. For workflow: Add to `src/open_deep_research/graph.py`
2. For multi-agent: Add to tool lists in `src/open_deep_research/multi_agent.py`
3. Consider MCP integration for external tools

### **Debugging Issues**
1. Check logs in implementation files
2. Use test runner: `python tests/run_test.py`
3. Check dependency status: `python -c "from open_deep_research.dependency_manager import get_status_report; print(get_status_report())"`
4. Verify configuration: Check `.env` file and environment variables

## 📊 Performance & Optimization

### **Context Length Management**
- `src/open_deep_research/utils.py`: `truncate_messages_for_context()`
- `src/open_deep_research/utils.py`: `count_messages_tokens()`

### **Search Result Processing**
- `src/open_deep_research/utils.py`: `summarize_search_results()`
- `src/open_deep_research/utils.py`: `process_search_results()`

### **Model-Specific Optimizations**
- `src/open_deep_research/multi_agent.py`: Model-specific handling
- `src/open_deep_research/graph.py`: Thinking budget allocation

## 🚀 Entry Points

### **LangGraph Server**
```bash
langgraph dev  # Starts server with Studio UI
```

### **Streamlit UI**
```bash
streamlit run streamlit_app.py
```

### **Jupyter Notebooks**
- `src/open_deep_research/graph.ipynb`
- `src/open_deep_research/multi_agent.ipynb`

### **Command Line Testing**
```bash
python tests/run_test.py --help
```

This map provides quick navigation to any functionality within the codebase! 