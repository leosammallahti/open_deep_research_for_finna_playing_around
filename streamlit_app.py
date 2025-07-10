"""Streamlit web application for Open Deep Research.

This module provides the main web interface for the Open Deep Research application,
allowing users to configure research settings and generate AI-powered research reports.
"""

import asyncio
import datetime
import os
import sys
import time
import traceback
from pathlib import Path

import streamlit as st

# Validate environment configuration early – this will raise a clear error
# if API keys contain whitespace or other issues, preventing expensive graph
# execution later.
from open_deep_research.core.settings import settings

# *Fail fast*: if the settings are invalid the following call will raise a
# ValueError and Streamlit will display the traceback at startup.
try:
    settings.validate_all()
except Exception as e:
    st.error(f"⚠️ Configuration error: {e}")
    # Re-raise to stop app initialisation – safer than running with bad config.
    raise

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from open_deep_research.configuration import (
    COMPREHENSIVE_REPORT_STRUCTURE,
    DEFAULT_REPORT_STRUCTURE,
    EXECUTIVE_SUMMARY_STRUCTURE,
    SearchAPI,
)
from open_deep_research.core.model_utils import trace_config
from open_deep_research.dependency_manager import (
    SearchProvider,
    get_available_providers,
    get_status_report,
)
from open_deep_research.exceptions import OutOfBudgetError
from open_deep_research.graph import get_state_value, graph
from open_deep_research.model_registry import (
    get_available_model_combos,
    get_available_models,
    supports_tool_choice,
)

# Safe environment variable loading with error handling
ENV_LOAD_ERROR = None
try:
    from dotenv import load_dotenv

    # By specifying the path, we prevent dotenv from searching parent directories
    # and finding a potentially mis-encoded .env file.
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
except UnicodeDecodeError as e:
    ENV_LOAD_ERROR = f"Environment file encoding error: {str(e)}. Please check your .env file encoding (should be UTF-8)."
except FileNotFoundError:
    ENV_LOAD_ERROR = "No .env file found. This is optional - you can set environment variables manually."
except Exception as e:
    ENV_LOAD_ERROR = f"Environment file error: {str(e)}"

# Error tracker was removed - using simple error handling instead

# Set page configuration
st.set_page_config(
    page_title="Open Deep Research",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Show environment loading error if there was one
if ENV_LOAD_ERROR:
    st.error(f"⚠️ **Environment Configuration Issue**: {ENV_LOAD_ERROR}")
    st.info(
        "💡 **How to fix**: Create a `.env` file in the project root with your API keys in UTF-8 encoding, or set environment variables manually."
    )

    with st.expander("🔧 Example .env file format"):
        st.code(
            """
# Save this as .env in UTF-8 encoding
DEEPSEEK_API_KEY=your_deepseek_key_here
TAVILY_API_KEY=your_tavily_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
""",
            language="bash",
        )

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .report-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 2rem;
        margin-top: 2rem;
    }
    h1 {
        color: #1e3a8a;
    }

    /* Hide the link icon that Streamlit automatically adds to headers */
    div[data-testid="stHeading"] a {
        display: none;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "report" not in st.session_state:
    st.session_state.report = None
if "research_in_progress" not in st.session_state:
    st.session_state.research_in_progress = False
if "research_task" not in st.session_state:
    st.session_state.research_task = None
if "should_stop_research" not in st.session_state:
    st.session_state.should_stop_research = False

# Header
st.title("🔍 Open Deep Research")
st.markdown(
    "**AI-powered research assistant** that helps you create comprehensive reports on any topic"
)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- Search Provider Selection ---
    # Use the SearchProvider enum directly instead of manual mapping
    PROVIDER_DISPLAY_NAMES = {
        SearchProvider.GOOGLE: "Google Search",
        SearchProvider.DUCKDUCKGO: "DuckDuckGo",
        SearchProvider.ARXIV: "ArXiv (Academic Papers)",
        SearchProvider.PUBMED: "PubMed (Medical Research)",
        SearchProvider.PERPLEXITY: "Perplexity",
        SearchProvider.TAVILY: "Tavily",
        SearchProvider.EXA: "Exa",
        SearchProvider.LINKUP: "Linkup",
        SearchProvider.AZURE: "Azure AI Search",
    }

    available_providers = get_available_providers()
    available_provider_names = [
        PROVIDER_DISPLAY_NAMES[provider]
        for provider in available_providers
        if provider in PROVIDER_DISPLAY_NAMES
    ]

    if len(available_provider_names) < len(PROVIDER_DISPLAY_NAMES):
        st.warning("⚠️ Some search providers are not installed.")
        with st.expander("📋 View Provider Status"):
            st.text(get_status_report())

    # Determine default search provider from environment or fallback
    default_search_api = os.environ.get("SEARCH_API", "tavily")
    default_search_provider = next(
        (
            provider
            for provider in available_providers
            if provider.value == default_search_api
        ),
        SearchProvider.TAVILY,
    )
    default_search_name = PROVIDER_DISPLAY_NAMES.get(default_search_provider, "Tavily")

    try:
        default_index = available_provider_names.index(default_search_name)
    except ValueError:
        default_index = 0 if available_provider_names else -1

    # If no providers are available, add Tavily as a fallback
    if not available_provider_names:
        available_provider_names = ["Tavily"]
        st.warning("⚠️ No search providers are configured. Using Tavily as fallback.")

    search_provider_name = st.selectbox(
        "🔎 Search Provider",
        options=available_provider_names,
        index=max(0, default_index),
        help="Choose a search engine. Configure with SEARCH_API in your .env file.",
    )

    # Get the provider enum value from the display name
    search_provider_api = next(
        (
            provider.value
            for provider, name in PROVIDER_DISPLAY_NAMES.items()
            if name == search_provider_name
        ),
        "duckduckgo",
    )

    # --- Advanced Settings -------------------------------------------------
    with st.expander("🔧 Advanced settings"):
        adv_search_budget = st.number_input(
            "Search budget (credits)",
            min_value=1,
            value=int(os.environ.get("SEARCH_BUDGET", 100)),
        )
        adv_max_search_depth = st.slider(
            "Max search depth",
            min_value=1,
            max_value=5,
            value=int(os.environ.get("MAX_SEARCH_DEPTH", 2)),
        )
        adv_recursion_limit = st.number_input(
            "LangGraph recursion limit",
            min_value=50,
            value=int(os.environ.get("RECURSION_LIMIT", 100)),
        )

    st.session_state.adv_settings = {
        "search_budget": adv_search_budget,
        "max_search_depth": adv_max_search_depth,
        "recursion_limit": adv_recursion_limit,
    }

    # --- Model Selection ---
    st.subheader("🤖 AI Models")

    # Get only available model combos
    available_combos = get_available_model_combos()

    if not available_combos:
        st.error(
            "❌ No model combinations available. Please check your API keys in the .env file."
        )
        st.info(
            "Required API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY"
        )
        st.stop()

    # Predefined Model Combos
    combo_options = {
        key: combo["display_name"] for key, combo in available_combos.items()
    }
    combo_options["custom"] = "🧑‍🔧 Custom"

    selected_combo_key = st.selectbox(
        "Model Combination",
        options=list(combo_options.keys()),
        format_func=lambda key: combo_options[key],
        index=0,
        help="Select a curated set of models for different tasks, or choose 'Custom' to configure them yourself.",
    )

    if selected_combo_key == "custom":
        # Build model options dynamically per role using registry with API key filtering
        planner_model_options = get_available_models("planner")
        writer_model_options = get_available_models("writer")

        if not planner_model_options:
            st.error(
                "❌ No planner models available. Please configure API keys for at least one model provider."
            )
            st.stop()

        if not writer_model_options:
            st.error(
                "❌ No writer models available. Please configure API keys for at least one model provider."
            )
            st.stop()

        # Planner (Supervisor) Model
        default_planner = os.environ.get("PLANNER_MODEL", planner_model_options[0])
        if default_planner not in planner_model_options:
            default_planner = planner_model_options[0]
        supervisor_model = st.selectbox(
            "Planner / Supervisor Model",
            options=planner_model_options,
            index=planner_model_options.index(default_planner),
            help="Model that plans / coordinates the report. Configure with PLANNER_MODEL in your .env file.",
        )

        # Writer (Researcher) Model
        default_writer = os.environ.get("WRITER_MODEL", writer_model_options[0])
        if default_writer not in writer_model_options:
            default_writer = writer_model_options[0]
        researcher_model = st.selectbox(
            "Writer / Researcher Model",
            options=writer_model_options,
            index=writer_model_options.index(default_writer),
            help="Model that writes sections. Configure with WRITER_MODEL in your .env file.",
        )

        # Add a warning if the writer model doesn't support tool choice
        if not supports_tool_choice(researcher_model):
            st.warning(
                f"**Warning:** The selected writer model (`{researcher_model}`) does not support tool calling. This may lead to errors or poor performance as it cannot reliably generate structured output."
            )

        # Optional summarizer drop-down if users want custom summarization model
        with st.expander("📝 Summarisation Model (optional)"):
            summarizer_model_options = (
                get_available_models("summarizer") or writer_model_options
            )
            default_summarizer = os.environ.get(
                "SUMMARIZER_MODEL", summarizer_model_options[0]
            )
            if default_summarizer not in summarizer_model_options:
                default_summarizer = summarizer_model_options[0]
            summarizer_model = st.selectbox(
                "Summarizer Model",
                options=summarizer_model_options,
                index=summarizer_model_options.index(default_summarizer),
                help="Model used to summarise large search results. Configure with SUMMARIZER_MODEL in your .env file.",
            )
    else:
        # Use a predefined combo
        combo = available_combos[selected_combo_key]
        supervisor_model = combo["planner"]
        researcher_model = combo["writer"]
        summarizer_model = combo.get("summarizer")  # Can be None
        st.caption(
            f"Using **{combo['planner']}** for planning and **{combo['writer']}** for writing."
        )

    # --- Advanced Settings ---
    with st.expander("🔧 Advanced Settings"):
        # Report Length / Style
        report_style_options = {
            "Concise": DEFAULT_REPORT_STRUCTURE,
            "Comprehensive": COMPREHENSIVE_REPORT_STRUCTURE,
            "Executive Summary": EXECUTIVE_SUMMARY_STRUCTURE,
        }
        report_style_key = st.selectbox(
            "Report Style",
            options=list(report_style_options.keys()),
            index=0,
            help="Controls the length and detail of the final report.",
        )
        report_structure_prompt = report_style_options[report_style_key]

        # Number of Queries
        default_queries = int(os.environ.get("NUMBER_OF_QUERIES", 1))
        number_of_queries = st.slider(
            "Number of search queries per section",
            min_value=1,
            max_value=5,
            value=default_queries,
            help="More queries = more comprehensive research. Configure with NUMBER_OF_QUERIES in your .env file.",
        )

        ask_for_clarification = st.checkbox(
            "Allow AI to ask clarifying questions",
            value=False,
            help="Enable if you want the AI to ask for more details about your topic",
        )

        include_sources = st.checkbox(
            "Include sources and citations",
            value=True,
            help="Include numbered citations and source URLs in the final report",
        )

        include_raw_sources = st.checkbox(
            "Include full raw sources content (very large)",
            value=False,
            help="If enabled, the full unprocessed search results will be appended under a 'Raw Sources' section. This can exceed tens of thousands of characters.",
        )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Research Topic")
    research_topic = st.text_area(
        "What would you like to research?",
        placeholder="Enter your research topic here... For example:\n- The impact of artificial intelligence on healthcare\n- Climate change mitigation strategies\n- The history of quantum computing",
        height=150,
    )

with col2:
    st.header("ℹ️ How it works")
    st.info("""
    1. **Enter your topic** - Be specific about what you want to research
    2. **Configure settings** - Choose search provider and report preferences
    3. **Start research** - The AI will search, analyze, and write a comprehensive report
    4. **Get your report** - Receive a well-structured report with introduction, body sections, and conclusion
    """)

# Research buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button(
        "🚀 Start Research",
        type="primary",
        disabled=st.session_state.research_in_progress,
    ):
        if not research_topic:
            st.error("⚠️ Please enter a research topic!")
        else:
            st.session_state.research_in_progress = True
            st.session_state.report = None
            st.session_state.should_stop_research = False

with col2:
    if st.button("⏹️ Stop Research", disabled=not st.session_state.research_in_progress):
        st.session_state.should_stop_research = True
        st.session_state.research_in_progress = False

        # Cancel the research task if it exists
        if "research_task" in st.session_state and st.session_state.research_task:
            try:
                st.session_state.research_task.cancel()
            except Exception:
                pass  # Task might already be done or cancelled

        st.warning("🛑 Research stopped by user. You can start a new research session.")
        st.rerun()

# Run research if button was clicked
if st.session_state.research_in_progress:
    # Create configuration for graph.py implementation
    # Extract provider and model from the model strings (e.g., "anthropic:claude-3.5-sonnet-20240620")

    # --- Economy settings for Tavily -------------------------------
    # When the user selects the *Concise* report style we default to
    # the cheapest Tavily parameters: one result at basic depth.  This
    # typically costs ~5 credits (1 result * 1 credit for basic).
    # Users can still override via .env or Advanced Settings in future.

    search_api_config: dict[str, object] | None = None
    if report_style_key == "Concise" and search_provider_api == "tavily":
        search_api_config = {
            "search_depth": "basic",  # cheapest depth tier
            "max_results": 1,
        }

    # ----------------------------------------------------------------

    config_values: dict[str, object] = {
        "search_api": SearchAPI(search_provider_api)
        if search_provider_api
        else SearchAPI.DUCKDUCKGO,
        "planner_provider": supervisor_model.split(":", 1)[0],
        "planner_model": supervisor_model.split(":", 1)[1],
        "writer_provider": researcher_model.split(":", 1)[0],
        "writer_model": researcher_model.split(":", 1)[1],
        "summarization_model_provider": summarizer_model.split(":", 1)[0]
        if summarizer_model
        else None,
        "summarization_model": summarizer_model.split(":", 1)[1]
        if summarizer_model
        else None,
        "report_structure": report_structure_prompt,
        "number_of_queries": number_of_queries,
        "ask_for_clarification": ask_for_clarification,
        "include_source_str": include_sources,
        "include_raw_source_details": include_raw_sources,
        "search_budget": st.session_state.adv_settings["search_budget"],
        "max_search_depth": st.session_state.adv_settings["max_search_depth"],
    }

    # Inject economy search settings if we built them earlier.
    if search_api_config is not None:
        config_values["search_api_config"] = search_api_config

    config = {"configurable": config_values}

    # Create progress container
    progress_container = st.container()

    with progress_container:
        st.info("🔍 Research in progress...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Run the research
        async def run_research():
            """Execute the research workflow asynchronously with progress tracking.

            Returns:
                The final research report or error message
            """
            if st.session_state.should_stop_research:
                return None

            try:
                # Debug logging
                with open("streamlit_debug.log", "a") as f:
                    f.write(f"\n\n{'=' * 50}\n")
                    f.write(f"Starting research at {datetime.datetime.now()}\n")
                    f.write(f"Topic: {research_topic}\n")
                    f.write(f"Config: {config}\n")

                status_text.text("Initializing research agents...")
                progress_bar.progress(10)

                status_text.text("Planning report structure...")
                progress_bar.progress(20)

                # Prepare live status placeholders
                live_status = st.status("Running research…", state="running")
                iteration_placeholder = st.empty()
                credits_placeholder = st.empty()
                elapsed_placeholder = st.empty()

                start_ts = time.time()

                # Stream graph events so we can update the UI progressively
                # Use stream_mode="values" to get the full state at each step, not just updates
                latest_state = None
                async for step_state in graph.astream(
                    {"topic": research_topic},
                    config={
                        **trace_config("streamlit-session"),
                        "configurable": config["configurable"],
                        "recursion_limit": st.session_state.adv_settings[
                            "recursion_limit"
                        ],
                    },
                    stream_mode="values",  # Get full state, not just updates
                ):
                    latest_state = step_state

                    # Extract metrics gracefully from dict or model
                    # Prefer attribute access but fall back to Mapping support via helper
                    iter_count = get_state_value(step_state, "search_iterations", 0)
                    credits_remaining = get_state_value(step_state, "credits_remaining")

                    budget = config["configurable"].get("search_budget", 100)

                    elapsed = int(time.time() - start_ts)

                    iteration_placeholder.text(f"🔄 Iteration: {iter_count}")
                    if credits_remaining is not None:
                        credits_used = budget - credits_remaining
                        credits_placeholder.text(
                            f"💰 Credits: {credits_used} / {budget} (remaining {credits_remaining})"
                        )
                    elapsed_placeholder.text(f"⏱️ Elapsed: {elapsed}s")

                final_state = latest_state

                progress_bar.progress(100)
                live_status.update(label="Research complete!", state="complete")
                status_text.text("Research complete!")

                # Access the final_report from the state object (can be Pydantic model or dictionary)
                if final_state is None:
                    return "Error: Graph execution returned no result. Please check your configuration and try again."

                # Handle both Pydantic model and dictionary returns
                # Unified state access irrespective of Mapping vs object
                final_report = get_state_value(final_state, "final_report")
                sections = get_state_value(final_state, "sections", [])

                # Debug logging
                with open("streamlit_debug.log", "a") as f:
                    f.write(f"\nFinal state type: {type(final_state)}\n")
                    if hasattr(final_state, "__dict__"):
                        f.write(f"State attributes: {list(vars(final_state).keys())}\n")
                    elif isinstance(final_state, dict):
                        f.write(f"State keys: {list(final_state.keys())}\n")
                    f.write(f"Final report found: {bool(final_report)}\n")
                    f.write(
                        f"Final report length: {len(final_report) if final_report else 0}\n"
                    )
                    f.write(f"Sections found: {len(sections)}\n")

                if final_report:
                    return final_report
                else:
                    # Try to get sections if final_report is not available
                    if sections:
                        return f"Report generation incomplete. Generated {len(sections)} sections but final compilation failed."
                    else:
                        return "No report generated. The research process may have failed to complete."

            except OutOfBudgetError:
                st.session_state.research_in_progress = False
                st.warning(
                    "🛑 Search budget exhausted. Consider increasing the budget or lowering search depth/results."
                )
            except Exception as e:
                # Log the error to console for debugging
                traceback.print_exc()

                error_msg = str(e)

                # Log the error to file with more robust error handling
                try:
                    with open("streamlit_error.log", "a", encoding="utf-8") as f:
                        f.write(f"\n\n{'=' * 50}\n")
                        f.write(f"Error at {datetime.datetime.now()}\n")
                        f.write(f"Topic: {research_topic}\n")
                        f.write(
                            f"Models: {config['configurable']['planner_provider']}:{config['configurable']['planner_model']} / {config['configurable']['writer_provider']}:{config['configurable']['writer_model']}\n"
                        )
                        f.write(f"Error: {error_msg}\n")
                        f.write(f"Traceback:\n{traceback.format_exc()}\n")
                        f.flush()  # Ensure the log is written immediately
                except Exception:
                    pass

                raise e

        # Run the async function with proper cancellation handling
        try:
            # Create a new event loop for the research task
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Create a task that can be cancelled
                research_task = loop.create_task(run_research())
                st.session_state.research_task = research_task

                # Run until complete or cancelled
                report = loop.run_until_complete(research_task)

                if report:
                    st.session_state.report = report
                    st.success("✅ Research completed successfully!")
                    st.session_state.research_in_progress = False
                    # Remove problematic st.rerun() to prevent auto-restart
                else:
                    # Don't rerun on error - let user see the error message
                    st.session_state.research_in_progress = False
            except asyncio.CancelledError:
                st.session_state.research_in_progress = False
                st.info("🛑 Research was cancelled by user.")
            except RuntimeError as e:
                error_msg = str(e)
                if (
                    "cannot schedule new futures after shutdown" in error_msg
                    or "Event loop is closed" in error_msg
                ):
                    st.session_state.research_in_progress = False
                    st.warning(
                        "🛑 Research was stopped. The system is cleaning up resources."
                    )
                    # Log the error for debugging
                    try:
                        with open("streamlit_error.log", "a", encoding="utf-8") as f:
                            f.write(f"\n\n{'=' * 50}\n")
                            f.write(
                                f"Event loop shutdown error at {datetime.datetime.now()}: {error_msg}\n"
                            )
                            f.flush()
                    except Exception:
                        pass
                else:
                    raise
            finally:
                # Clean up the event loop
                try:
                    # Cancel any pending tasks
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()

                    # Give tasks a chance to clean up
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass  # Ignore errors during cleanup

                # Close the loop
                loop.close()

        except asyncio.CancelledError:
            st.session_state.research_in_progress = False
            st.info("🛑 Research was cancelled successfully.")
        except RuntimeError as e:
            if "cannot schedule new futures after shutdown" in str(e):
                st.session_state.research_in_progress = False
                st.warning(
                    "🛑 Research was stopped. The system is cleaning up resources."
                )
            else:
                raise
        except Exception as e:
            # Handle final error after all retries
            st.session_state.research_in_progress = False

            # --- Simple Error Analysis ---
            error_msg = str(e)

            # Provide specific error messages for common issues
            if "context length" in error_msg.lower():
                st.error(
                    "❌ **Context Length Error**: The research topic is too complex for the selected model. Try:"
                )
                st.error(
                    "• Using a model with larger context (e.g., Claude 3.5 Sonnet)"
                )
                st.error("• Simplifying your research topic")
                st.error("• Reducing the report length setting")
                st.error("• Reducing the number of search queries")
            elif "rate limit" in error_msg.lower() or "RateLimitError" in error_msg:
                st.error(
                    "❌ **Rate Limit Error**: The API rate limit was exceeded. Try:"
                )
                st.error("• Waiting 5-10 minutes before retrying")
                st.error(
                    "• Using a different model provider (switch from OpenAI to Anthropic or vice versa)"
                )
                st.error("• Reducing the number of search queries per section")
                st.error("• Using a model with higher rate limits")
                st.info(
                    "💡 **Tip**: OpenAI has stricter rate limits than Anthropic for some models."
                )
            elif (
                "authentication" in error_msg.lower() or "api key" in error_msg.lower()
            ):
                st.error("❌ **Authentication Error**: API key issue. Try:")
                st.error("• Checking your API keys in the .env file")
                st.error("• Making sure there are no extra spaces or newlines")
                st.error("• Verifying your API key is valid and has sufficient credits")
            elif "timeout" in error_msg.lower():
                st.error("❌ **Network Timeout**: Search service is slow. Try:")
                st.error("• Retrying with a different search provider")
                st.error("• Checking your internet connection")
                st.error("• Using a simpler research topic")
            elif "overloaded" in error_msg.lower() or "529" in error_msg:
                st.error(
                    "❌ **API Overloaded**: The AI service is currently overloaded. Try:"
                )
                st.error("• Waiting 1-2 minutes before retrying")
                st.error(
                    "• Using a different AI model (e.g., switching from Claude to GPT)"
                )
                st.error("• Reducing the complexity of your research topic")
                st.error("• Trying again during off-peak hours")
                st.info(
                    "💡 **Tip**: The app automatically retries with backoff delays, but high demand can still cause failures."
                )
            elif "cannot schedule new futures after shutdown" in error_msg.lower():
                st.error(
                    "❌ **System Shutdown Error**: The research was interrupted during execution. Try:"
                )
                st.error("• Restarting the Streamlit app")
                st.error("• Using a simpler research topic")
                st.error("• Avoiding rapid start/stop operations")
                st.info(
                    "💡 **Tip**: This usually happens when research is stopped abruptly. The app is designed to handle this gracefully."
                )
            else:
                st.error(f"❌ **Unexpected Error**: {str(e)}")
                # Also print traceback to console for unexpected errors
                traceback.print_exc()
                st.error("Check streamlit_error.log for full details")

# Display the report if available
if st.session_state.report:
    st.markdown("---")
    st.header("📊 Research Report")

    # Add download button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        st.download_button(
            label="📥 Download Report (Markdown)",
            data=st.session_state.report,
            file_name="research_report.md",
            mime="text/markdown",
        )
    with col2:
        if st.button("🔄 New Research"):
            st.session_state.report = None
            st.rerun()

    # Display report
    st.markdown(st.session_state.report)

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666;'>
    <p>Powered by Open Deep Research | Built with ❤️ using LangGraph and Streamlit</p>
    <p style='font-size: 0.8em;'>Note: Make sure you have configured your API keys in the .env file</p>
</div>
""",
    unsafe_allow_html=True,
)
