# Open Deep Research - Directory Structure

## Root Directory
```
open_deep_research_for_finna_playing_around/
├── src/open_deep_research/          # Main package source code
├── tests/                           # Test files
├── examples/                        # Example configurations and usage
├── .notes/                          # Project documentation for AI context
├── .cursor/                         # Cursor IDE configuration (if using new format)
├── .cursorrules                     # Cursor AI behavior rules
├── .cursorignore                    # Files for Cursor to ignore
├── README.md                        # Main project documentation
├── pyproject.toml                   # Python project configuration
├── uv.lock                          # Dependency lock file
└── streamlit_app.py                 # Streamlit web interface
```

## Source Code Structure (`src/open_deep_research/`)

### Core Files
- **`__init__.py`** - Package initialization
- **`state.py`** - State management definitions
- **`pydantic_state.py`** - Immutable Pydantic state models
- **`graph.py`** - Graph-based workflow implementation
- **`multi_agent.py`** - Multi-agent workflow implementation
- **`configuration.py`** - Configuration management
- **`model_registry.py`** - Model provider registry
- **`utils.py`** - Utility functions
- **`prompts.py`** - AI model prompts
- **`tavily_tools.py`** - Tavily search integration
- **`message_utils.py`** - Message handling utilities
- **`dependency_manager.py`** - Dependency management

### Core Module (`src/open_deep_research/core/`)
- **`config_utils.py`** - Configuration utilities
- **`format_utils.py`** - Output formatting utilities
- **`logging_utils.py`** - Logging configuration
- **`model_utils.py`** - Model interaction utilities

### Workflow Module (`src/open_deep_research/workflow/`)
- **`configuration.py`** - Workflow-specific configuration
- **`prompts.py`** - Workflow-specific prompts
- **`workflow.py`** - Workflow node definitions

### Additional Resources
- **`files/`** - Static files and resources
- **`unified_search_example.py`** - Example search implementations

## Test Structure (`tests/`)
- **`conftest.py`** - Pytest configuration
- **`test_integration.py`** - Integration tests
- **`test_pydantic_state.py`** - State management tests
- **`test_report_quality.py`** - Report quality tests
- **`test_utils.py`** - Utility function tests
- **`run_test.py`** - Test runner

### Evaluation Tests (`tests/evals/`)
- **`evaluators.py`** - Evaluation framework
- **`prompts.py`** - Evaluation prompts
- **`run_evaluate.py`** - Evaluation runner
- **`target.py`** - Evaluation targets

## Examples Directory (`examples/`)
- **`arxiv.md`** - ArXiv search examples
- **`pubmed.md`** - PubMed search examples
- **`inference-market.md`** - Inference market examples
- **`inference-market-gpt45.md`** - GPT-4.5 specific examples

## Documentation Files
- **`README.md`** - Main project documentation
- **`ARCHITECTURE.md`** - System architecture overview
- **`API_REFERENCE.md`** - API documentation
- **`IMPLEMENTATION_GUIDE.md`** - Implementation details
- **`DEVELOPMENT_GUIDE.md`** - Development setup guide
- **`CONFIGURATION_GUIDE.md`** - Configuration instructions
- **`DEPENDENCY_MANAGEMENT.md`** - Dependency management guide

## Configuration Files
- **`pyproject.toml`** - Python project and dependency configuration
- **`uv.lock`** - Dependency lock file (auto-generated)
- **`mypy.ini`** - MyPy type checking configuration
- **`langgraph.json`** - LangGraph configuration
- **`.gitignore`** - Git ignore patterns

## Key Design Patterns

### State Management
- Immutable state updates using Pydantic models
- Centralized state definition in `pydantic_state.py`
- Type-safe state transitions

### Workflow Organization
- Separation of graph-based and multi-agent implementations
- Modular workflow node definitions
- Configurable workflow parameters

### Search Integration
- Unified search interface across providers
- Individual tool files for each search provider
- Extensible search provider architecture

### Configuration Management
- Centralized configuration in `configuration.py`
- Environment variable support
- Flexible model and provider selection

## Development Workflow
1. Core logic in `src/open_deep_research/`
2. Tests in `tests/` with matching structure
3. Examples in `examples/` for demonstration
4. Documentation in root and `.notes/` for AI context 