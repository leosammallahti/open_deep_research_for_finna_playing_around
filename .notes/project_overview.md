# Open Deep Research - Project Overview

## Project Purpose
Open Deep Research is an experimental, fully open-source research assistant that automates deep research and produces comprehensive reports on any topic. It's designed to be flexible, configurable, and capable of working with multiple AI models and search providers.

## Key Architecture Components

### 1. LangGraph Workflow System
- **Framework**: Built on LangGraph for orchestrating complex workflows
- **State Management**: Uses immutable Pydantic models (DeepResearchState)
- **Workflow Types**: Supports both graph-based and multi-agent implementations

### 2. Multi-Model Support
- **Flexible Model Selection**: Choose different models for planning, writing, and summarization
- **Provider Agnostic**: Works with any model supported by `init_chat_model()` API
- **Fallback Strategies**: Handles models that don't support function calling

### 3. Search Integration
- **Multiple APIs**: Tavily, Perplexity, Exa, ArXiv, PubMed, Linkup, DuckDuckGo, Google Search
- **Unified Interface**: Consistent search interface across all providers
- **MCP Support**: Model Context Protocol for extended data access

### 4. Report Generation
- **Structured Output**: Generates comprehensive markdown reports
- **Section-based**: Breaks down topics into logical sections
- **Human-in-the-Loop**: Optional human feedback and approval steps

## Current Implementation Status

### Completed Features
- ✅ Graph-based workflow implementation
- ✅ Multi-agent implementation with supervisor-researcher pattern
- ✅ Multiple search API integrations
- ✅ Flexible model configuration system
- ✅ Immutable state management with Pydantic
- ✅ Streamlit UI for easy interaction
- ✅ Comprehensive error handling and logging

### Active Development Areas
- 🔄 MCP server integration enhancements
- 🔄 Performance optimizations
- 🔄 Additional search provider integrations
- 🔄 UI/UX improvements

## Technology Stack
- **Language**: Python 3.11+
- **Framework**: LangGraph for workflow orchestration
- **State Management**: Pydantic models with immutable patterns
- **UI**: Streamlit for web interface
- **Package Management**: uv (modern Python packaging)
- **Testing**: pytest with comprehensive test coverage

## User Workflow
1. **Topic Input**: User provides research topic
2. **Planning**: AI generates structured research plan
3. **Human Review**: Optional feedback and approval step
4. **Research**: Parallel or sequential section research
5. **Report Generation**: Comprehensive markdown report output

## Key Design Principles
- **Modularity**: Separate concerns for planning, research, and writing
- **Flexibility**: Support for multiple models and search providers
- **Reliability**: Robust error handling and fallback strategies
- **Extensibility**: Easy to add new search providers and model integrations
- **Type Safety**: Comprehensive type hints and Pydantic validation 