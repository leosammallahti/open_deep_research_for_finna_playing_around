# Open Deep Research

Open Deep Research is an experimental, fully open-source research assistant that automates deep research and produces comprehensive reports on any topic. It features a highly flexible architecture that allows you to customize the entire research and writing process.

### Key Features
- **🤖 Flexible, Per-Role Model Selection**: Choose the best AI model for each phase of the research process—planning, writing, and summarization. Use powerful reasoning models like DeepSeek for planning and state-of-the-art writers like Claude 3.5 Sonnet for content generation.
- **🚀 Predefined Model Combinations**: Get started quickly with curated model setups like "Max Reasoning" or "All Open-Source," or create your own custom combination.
- **🛠️ Robust Tool & Model Compatibility**: The system intelligently handles models that don't natively support function calling (like `deepseek-reasoner`) by using a fallback strategy, ensuring smooth, error-free operation.
- **✨ Automatic Output Cleaning**: Automatically filters out artifacts like `<think>` tokens from models, ensuring clean, usable output.
- **🔌 Multiple Search APIs**: Supports a wide array of search providers, including Tavily, Perplexity, Google Search, ArXiv, and more.
- ** HUMAN-IN-THE-LOOP**: Optional human feedback steps to review and approve the AI's research plan before execution.

#### Workflow

![open-deep-research-overview](https://github.com/user-attachments/assets/a171660d-b735-4587-ab2f-cd771f773756)

####  Multi-agent

![multi-agent-researcher](https://github.com/user-attachments/assets/3c734c3c-57aa-4bc0-85dd-74e2ec2c0880)

### 🚀 Quickstart

Clone the repository:
```bash
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
```

Then edit the `.env` file to customize the environment variables (for model selection, search tools, and other configuration settings):
```bash
cp .env.example .env
```

Launch the assistant with the LangGraph server locally, which will open in your browser:

#### Mac

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and start the LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

#### Windows / Linux

```powershell
# Install dependencies 
pip install -e .
pip install -U "langgraph-cli[inmem]" 

# Start the LangGraph server
langgraph dev
```

Use this to open the Studio UI:
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

### ⚙️ Configuring Your Assistant

From the UI, you can easily configure the assistant:
1.  **Choose a Model Combination**: Select a predefined combo like `Max Reasoning` for the best results, or `All Open-Source` to use models like Llama 3. Select `Custom` to configure each model individually.
2.  **Select a Search Provider**: Choose the search engine you want the assistant to use.
3.  **Adjust Advanced Settings**: Control the report style (Comprehensive, Concise, etc.) and the number of search queries per section.

#### Multi-agent

(1) Chat with the agent about your topic of interest, and it will initiate report generation:

<img width="1326" alt="input" src="https://github.com/user-attachments/assets/dc8f59dd-14b3-4a62-ac18-d2f99c8bbe83" />

(2) The report is produced as markdown.

#### Workflow

(1) Provide a `Topic`:

<img width="1326" alt="input" src="https://github.com/user-attachments/assets/de264b1b-8ea5-4090-8e72-e1ef1230262f" />

(2) This will generate a report plan and present it to the user for review.

(3) We can pass a string (`"..."`) with feedback to regenerate the plan based on the feedback.

<img width="1326" alt="feedback" src="https://github.com/user-attachments/assets/c308e888-4642-4c74-bc78-76576a2da919" />

(4) Or, we can just pass `true` to the JSON input box in Studio accept the plan.

<img width="1480" alt="accept" src="https://github.com/user-attachments/assets/ddeeb33b-fdce-494f-af8b-bd2acc1cef06" />

(5) Once accepted, the report sections will be generated.

<img width="1326" alt="report_gen" src="https://github.com/user-attachments/assets/74ff01cc-e7ed-47b8-bd0c-4ef615253c46" />

The report is produced as markdown.

<img width="1326" alt="report" src="https://github.com/user-attachments/assets/92d9f7b7-3aea-4025-be99-7fb0d4b47289" />

### Search Tools

Available search tools:

* [Tavily API](https://tavily.com/) - General web search
* [Perplexity API](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api) - General web search
* [Exa API](https://exa.ai/) - Powerful neural search for web content
* [ArXiv](https://arxiv.org/) - Academic papers in physics, mathematics, computer science, and more
* [PubMed](https://pubmed.ncbi.nlm.nih.gov/) - Biomedical literature from MEDLINE, life science journals, and online books
* [Linkup API](https://www.linkup.so/) - General web search
* [DuckDuckGo API](https://duckduckgo.com/) - General web search
* [Google Search API/Scrapper](https://google.com/) - Create custom search engine [here](https://programmablesearchengine.google.com/controlpanel/all) and get API key [here](https://developers.google.com/custom-search/v1/introduction)
* [Microsoft Azure AI Search](https://azure.microsoft.com/en-us/products/ai-services/ai-search) - Cloud based vector database solution 

Open Deep Research is compatible with many different LLMs: 

* You can select any model that is integrated [with the `init_chat_model()` API](https://python.langchain.com/docs/how_to/chat_models_universal_init/)
* See full list of supported integrations [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)

### Using the package

```bash
pip install open-deep-research
```

See [src/open_deep_research/graph.ipynb](src/open_deep_research/graph.ipynb) and [src/open_deep_research/multi_agent.ipynb](src/open_deep_research/multi_agent.ipynb) for example usage in a Jupyter notebook:

## Open Deep Research Implementations

Open Deep Research provides three different implementations for generating research reports. See the [Implementation Guide](IMPLEMENTATION_GUIDE.md) for detailed guidance on choosing the right implementation for your needs.

### Quick Overview

## 1. Graph-based Workflow Implementation (`src/open_deep_research/graph.py`)

The graph-based implementation follows a structured plan-and-execute workflow:

- **Planning Phase**: Uses a planner model to analyze the topic and generate a structured report plan
- **Human-in-the-Loop**: Allows for human feedback and approval of the report plan before proceeding
- **Sequential Research Process**: Creates sections one by one with reflection between search iterations
- **Section-Specific Research**: Each section has dedicated search queries and content retrieval
- **Supports Multiple Search Tools**: Works with all search providers (Tavily, Perplexity, Exa, ArXiv, PubMed, Linkup, etc.)

This implementation provides a more interactive experience with greater control over the report structure, making it ideal for situations where report quality and accuracy are critical.

You can customize the research assistant workflow through several parameters:

- `report_structure`: Define a custom structure for your report (defaults to a standard research report format)
- `number_of_queries`: Number of search queries to generate per section (default: 2)
- `max_search_depth`: Maximum number of reflection and search iterations (default: 2)
- `planner_provider`: Model provider for planning phase (default: "anthropic", but can be any provider from supported integrations with `init_chat_model` as listed [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html))
- `planner_model`: Specific model for planning (default: "claude-3-7-sonnet-latest")
- `planner_model_kwargs`: Additional parameter for planner_model
- `writer_provider`: Model provider for writing phase (default: "anthropic", but can be any provider from supported integrations with `init_chat_model` as listed [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html))
- `writer_model`: Model for writing the report (default: "claude-3-5-sonnet-latest")
- `writer_model_kwargs`: Additional parameter for writer_model
- `search_api`: API to use for web searches (default: "tavily", options include "perplexity", "exa", "arxiv", "pubmed", "linkup")

## 2. Multi-Agent Implementation (`src/open_deep_research/multi_agent.py`)

The multi-agent implementation uses a supervisor-researcher architecture:

- **Supervisor Agent**: Manages the overall research process, plans sections, and assembles the final report
- **Researcher Agents**: Multiple independent agents work in parallel, each responsible for researching and writing a specific section
- **Parallel Processing**: All sections are researched simultaneously, significantly reducing report generation time
- **Specialized Tool Design**: Each agent has access to specific tools for its role (search for researchers, section planning for supervisors)
- **Search and MCP Support**: Works with Tavily/DuckDuckGo for web search, MCP servers for local/external data access, or can operate without search tools using only MCP tools

This implementation focuses on efficiency and parallelization, making it ideal for faster report generation with less direct user involvement.

You can customize the multi-agent implementation through several parameters:

- `supervisor_model`: Model for the supervisor agent (default: "anthropic:claude-3-5-sonnet-latest")
- `researcher_model`: Model for researcher agents (default: "anthropic:claude-3-5-sonnet-latest") 
- `number_of_queries`: Number of search queries to generate per section (default: 2)
- `search_api`: API to use for web searches (default: "tavily", options include "duckduckgo", "none")
- `ask_for_clarification`: Whether the supervisor should ask clarifying questions before research (default: false) - **Important**: Set to `true` to enable the Question tool for the supervisor agent
- `mcp_server_config`: Configuration for MCP servers (optional)
- `mcp_prompt`: Additional instructions for using MCP tools (optional)
- `mcp_tools_to_include`: Specific MCP tools to include (optional)

## MCP (Model Context Protocol) Support

The multi-agent implementation (`src/open_deep_research/multi_agent.py`) supports MCP servers to extend research capabilities beyond web search. MCP tools are available to research agents alongside or instead of traditional search tools, enabling access to local files, databases, APIs, and other data sources.

**Note**: MCP support is currently only available in the multi-agent (`src/open_deep_research/multi_agent.py`) implementation, not in the workflow-based workflow implementation (`src/open_deep_research/graph.py`).

### Key Features

- **Tool Integration**: MCP tools are seamlessly integrated with existing search and section-writing tools
- **Research Agent Access**: Only research agents (not supervisors) have access to MCP tools
- **Flexible Configuration**: Use MCP tools alone or combined with web search
- **Disable Default Search**: Set `search_api: "none"` to disable web search tools entirely
- **Custom Prompts**: Add specific instructions for using MCP tools

### Filesystem Server Example

#### SKK

```python
config = {
    "configurable": {
        "search_api": "none",  # Use "tavily" or "duckduckgo" to combine with web search
        "mcp_server_config": {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "/path/to/your/files"
                ],
                "transport": "stdio"
            }
        },
        "mcp_prompt": "Step 1: Use the `list_allowed_directories` tool to get the list of allowed directories. Step 2: Use the `read_file` tool to read files in the allowed directory.",
        "mcp_tools_to_include": ["list_allowed_directories", "list_directory", "read_file"]  # Optional: specify which tools to include
    }
}
```

#### Studio

MCP server config: 
```
{
  "filesystem": {
    "command": "npx",
    "args": [
      "-y",
      "@modelcontextprotocol/server-filesystem",
      "/Users/rlm/Desktop/Code/open_deep_research/src/open_deep_research/files"
    ],
    "transport": "stdio"
  }
}
```

MCP prompt: 
```
CRITICAL: You MUST follow this EXACT sequence when using filesystem tools:

1. FIRST: Call `list_allowed_directories` tool to discover allowed directories
2. SECOND: Call `list_directory` tool on a specific directory from step 1 to see available files  
3. THIRD: Call `read_file` tool to read specific files found in step 2

DO NOT call `list_directory` or `read_file` until you have first called `list_allowed_directories`. You must discover the allowed directories before attempting to browse or read files.
```

MCP tools: 
```
list_allowed_directories
list_directory 
read_file
```

Example test topic and follow-up feedback that you can provide that will reference the included file: 

Topic:
```
I want an overview of vibe coding
```

Follow-up to the question asked by the research agent: 

```
I just want a single section report on vibe coding that highlights an interesting / fun example
```

Resulting trace: 

https://smith.langchain.com/public/d871311a-f288-4885-8f70-440ab557c3cf/r

### Configuration Options

- **`mcp_server_config`**: Dictionary defining MCP server configurations (see [langchain-mcp-adapters examples](https://github.com/langchain-ai/langchain-mcp-adapters#client-1))
- **`mcp_prompt`**: Optional instructions added to research agent prompts for using MCP tools
- **`mcp_tools_to_include`**: Optional list of specific MCP tool names to include (if not set, all tools from all servers are included)
- **`search_api`**: Set to `"none"` to use only MCP tools, or keep existing search APIs to combine both

### Common Use Cases

- **Local Documentation**: Access project documentation, code files, or knowledge bases
- **Database Queries**: Connect to databases for specific data retrieval
- **API Integration**: Access external APIs and services
- **File Analysis**: Read and analyze local files during research

The MCP integration allows research agents to incorporate local knowledge and external data sources into their research process, creating more comprehensive and context-aware reports.

## Search API Configuration

Not all search APIs support additional configuration parameters. Here are the ones that do:

- **Exa**: `max_characters`, `num_results`, `include_domains`, `exclude_domains`, `subpages`
  - Note: `include_domains` and `exclude_domains` cannot be used together
  - Particularly useful when you need to narrow your research to specific trusted sources, ensure information accuracy, or when your research requires using specified domains (e.g., academic journals, government sites)
  - Provides AI-generated summaries tailored to your specific query, making it easier to extract relevant information from search results
- **ArXiv**: `load_max_docs`, `get_full_documents`, `load_all_available_meta`
- **PubMed**: `top_k_results`, `email`, `api_key`, `doc_content_chars_max`
- **Linkup**: `depth`

Example with Exa configuration:
```python
thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                           "search_api": "exa",
                           "search_api_config": {
                               "num_results": 5,
                               "include_domains": ["nature.com", "sciencedirect.com"]
                           },
                           # Other configuration...
                           }}
```

## 3. Unified Typed Workflow (`src/open_deep_research/workflow/`)

**Status: Experimental – under active migration.**  This path unifies the
graph-based and multi-agent approaches under one typed LangGraph powered by
immutable `DeepResearchState`.

Key points:

* **Adapter Layer** – Existing planner/researcher logic is reused via
  `NodeAdapter` wrappers so proven code continues to work.
* **Execution Modes** – Select between *workflow* (legacy graph) and
  *multi_agent* (supervisor-researcher) by either:
  1. Setting `state.execution_mode = "multi_agent"` before invoking the
     graph, **or**
  2. Supplying `{"configurable": {"mode": "multi_agent"}}` in
     `RunnableConfig`, **or**
  3. Enabling `features["mcp_support"] = True` in `WorkflowConfiguration`.
* **Fast/Offline CI** – When `ODR_FAST_TEST=1` the planner & researcher fall
  back to synthetic stubs so the full test-suite runs without network calls
  or model tokens.

Example usage (Python):

```python
from open_deep_research.pydantic_state import DeepResearchState
from open_deep_research.workflow.unified_workflow import unified_planner_graph

# Build initial state
state = DeepResearchState(topic="Future of Quantum Computing")

# Execute in multi-agent mode
config = {
    "configurable": {
        "mode": "multi_agent",  # or leave blank and set features["mcp_support"]
    }
}

final_state = unified_planner_graph.invoke(state, config)
print(final_state.final_report)
```

The unified workflow currently supports:

* Sequential *or* multi-agent research paths.
* Parallel section research via the `parallel_research` feature flag.
* Citation numbering and optional raw source block.

Refactor progress is tracked in
[`UNIFIED_WORKFLOW_PROGRESS.md`](UNIFIED_WORKFLOW_PROGRESS.md).

## Model Considerations

(1) You can use models supported with [the `init_chat_model()` API](https://python.langchain.com/docs/how_to/chat_models_universal_init/). See full list of supported integrations [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html).

(2) ***The workflow planner and writer models need to support structured outputs***: Check whether structured outputs are supported by the model you are using [here](https://python.langchain.com/docs/integrations/chat/).

(3) ***The agent models need to support tool calling:*** Ensure tool calling is well supoorted; tests have been done with Claude 3.7, o3, o3-mini, and gpt4.1. See [here](https://smith.langchain.com/public/adc5d60c-97ee-4aa0-8b2c-c776fb0d7bd6/d).

(4) With Groq, there are token per minute (TPM) limits if you are on the `on_demand` service tier:
- The `on_demand` service tier has a limit of `6000 TPM`
- You will want a [paid plan](https://github.com/cline/cline/issues/47#issuecomment-2640992272) for section writing with Groq models

(5) `deepseek-R1` [is not strong at function calling](https://api-docs.deepseek.com/guides/reasoning_model), which the assistant uses to generate structured outputs for report sections and report section grading. See example traces [here](https://smith.langchain.com/public/07d53997-4a6d-4ea8-9a1f-064a85cd6072/r).  
- Consider providers that are strong at function calling such as OpenAI, Anthropic, and certain OSS models like Groq's `llama-3.3-70b-versatile`.
- If you see the following error, it is likely due to the model not being able to produce structured outputs (see [trace](https://smith.langchain.com/public/8a6da065-3b8b-4a92-8df7-5468da336cbe/r)):
```
groq.APIError: Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.
```

(6) Follow [here[(https://github.com/langchain-ai/open_deep_research/issues/75#issuecomment-2811472408) to use with OpenRouter.

(7) For working with local models via Ollama, see [here](https://github.com/langchain-ai/open_deep_research/issues/65#issuecomment-2743586318).

## 🔍 Troubleshooting & Error Analysis

If you encounter errors while running Open Deep Research, use these built-in tools to quickly identify if issues have already been resolved:

### Error Analysis Tools
```bash
# Analyze error logs for known issues
python error_log_analyzer.py

# Analyze specific log file
python error_log_analyzer.py streamlit_error.log

# Confirm which fixes actually work (user testing required)
python confirm_fix.py
```

### Issue Status System
- **✅ USER_CONFIRMED** - User has personally tested and confirmed the fix works
- **❌ CONFIRMED_BROKEN** - User has confirmed the issue still exists despite attempted fixes  
- **🔧 NEEDS_VERIFICATION** - Code fix implemented but not confirmed by user
- **⚠️ ONGOING** - Known issues with workarounds but no permanent fix
- **❓ UNKNOWN** - New error patterns that haven't been categorized yet

### Documentation
- **📋 `RESOLVED_ISSUES.md`** - Complete tracker of all issues with their current status
- **🤖 `error_log_analyzer.py`** - Automated script that identifies resolved vs. ongoing issues
- **🧪 `confirm_fix.py`** - Interactive tool for users to confirm which fixes actually work

**Important:** Issues are only marked as truly resolved when users personally test and confirm they work. This prevents wasting time on "fixes" that don't actually solve the problem.

## Evaluation Systems

Open Deep Research includes two comprehensive evaluation systems to assess report quality and performance:

### 1. Pytest-based Evaluation System

A developer-friendly testing framework that provides immediate feedback during development and testing cycles.

#### **Features:**
- **Rich Console Output**: Formatted tables, progress indicators, and color-coded results
- **Binary Pass/Fail Testing**: Clear success/failure criteria for CI/CD integration
- **LangSmith Integration**: Automatic experiment tracking and logging
- **Flexible Configuration**: Extensive CLI options for different testing scenarios
- **Real-time Feedback**: Live output during test execution

#### **Evaluation Criteria:**
The system evaluates reports against 9 comprehensive quality dimensions:
- Topic relevance (overall and section-level)
- Structure and logical flow
- Introduction and conclusion quality
- Proper use of structural elements (headers, citations)
- Markdown formatting compliance
- Citation quality and source attribution
- Overall research depth and accuracy

#### **Usage:**
```bash
# Run all agents with default settings
python tests/run_test.py --all

# Test specific agent with custom models
python tests/run_test.py --agent multi_agent \
  --supervisor-model "anthropic:claude-3-7-sonnet-latest" \
  --search-api tavily

# Test with OpenAI o3 models
python tests/run_test.py --all \
  --supervisor-model "openai:o3" \
  --researcher-model "openai:o3" \
  --planner-provider "openai" \
  --planner-model "o3" \
  --writer-provider "openai" \
  --writer-model "o3" \
  --eval-model "openai:o3" \
  --search-api "tavily"
```

#### **Key Files:**
- `tests/run_test.py`: Main test runner with rich CLI interface
- `tests/test_report_quality.py`: Core test implementation
- `tests/conftest.py`: Pytest configuration and CLI options

### 2. LangSmith Evaluate API System

A comprehensive batch evaluation system designed for detailed analysis and comparative studies.

#### **Features:**
- **Multi-dimensional Scoring**: Four specialized evaluators with 1-5 scale ratings
- **Weighted Criteria**: Detailed scoring with customizable weights for different quality aspects
- **Dataset-driven Evaluation**: Batch processing across multiple test cases
- **Performance Optimization**: Caching with extended TTL for evaluator prompts
- **Professional Reporting**: Structured analysis with improvement recommendations

#### **Evaluation Dimensions:**

1. **Overall Quality** (7 weighted criteria):
   - Research depth and source quality (20%)
   - Analytical rigor and critical thinking (15%)
   - Structure and organization (20%)
   - Practical value and actionability (10%)
   - Balance and objectivity (15%)
   - Writing quality and clarity (10%)
   - Professional presentation (10%)

2. **Relevance**: Section-by-section topic relevance analysis with strict criteria

3. **Structure**: Assessment of logical flow, formatting, and citation practices

4. **Groundedness**: Evaluation of alignment with retrieved context and sources

#### **Usage:**
```bash
# Run comprehensive evaluation on LangSmith datasets
python tests/evals/run_evaluate.py
```

#### **Key Files:**
- `tests/evals/run_evaluate.py`: Main evaluation script
- `tests/evals/evaluators.py`: Four specialized evaluator functions
- `tests/evals/prompts.py`: Detailed evaluation prompts for each dimension
- `tests/evals/target.py`: Report generation workflows

### When to Use Each System

**Use Pytest System for:**
- Development and debugging cycles
- CI/CD pipeline integration
- Quick model comparison experiments
- Interactive testing with immediate feedback
- Gate-keeping before production deployments

**Use LangSmith System for:**
- Comprehensive model evaluation across datasets
- Research and analysis of system performance
- Detailed performance profiling and benchmarking
- Comparative studies between different configurations
- Production monitoring and quality assurance

Both evaluation systems complement each other and provide comprehensive coverage for different use cases and development stages.

## UX

### Local deployment

Follow the [quickstart](#-quickstart) to start LangGraph server locally.

### Hosted deployment
 
You can easily deploy to [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options). 

## Configuration at a Glance

All runtime knobs now live in a single typed object: `open_deep_research.core.settings.settings`.

• **Defaults** are loaded once from environment variables / a `.env` file into that object when the package is imported.
• **Per-run overrides** can still be supplied via `config={"configurable": …}` when you call the LangGraph pipeline.

To change behaviour you therefore have two options:
1.  Set an env-var (or add it to your `.env`) – e.g. `SEARCH_BUDGET=50`.
2.  Pass an override at call-time – e.g. `graph.ainvoke(input, config={"configurable": {"search_budget": 50}})`.

You never need to read or mutate `os.environ` directly – import `settings` instead:
```python
from open_deep_research.core.settings import settings
print(settings.search_budget)
```

---

### Developer workflow (pre-commit)

Run this once after cloning so the same quality gates enforced in CI run locally:
```bash
pre-commit install
```
The hook chain formats with Ruff, lints, then runs MyPy before every commit.

---
