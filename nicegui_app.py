"""Professional Research Platform - NiceGUI Implementation

A modern, professional research platform interface built with NiceGUI.
Features a clean sidebar navigation, structured workflow, and dark theme.
"""
import asyncio
import os
import sys
import traceback
from pathlib import Path

from nicegui import app, ui, run
from nicegui.events import ValueChangeEventArguments

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from open_deep_research.configuration import (
    COMPREHENSIVE_REPORT_STRUCTURE,
    DEFAULT_REPORT_STRUCTURE,
    EXECUTIVE_SUMMARY_STRUCTURE,
    SearchAPI,
)
from open_deep_research.dependency_manager import (
    SearchProvider,
    get_available_providers,
)
from open_deep_research.exceptions import OutOfBudgetError
from open_deep_research.graph import get_state_value, graph
from open_deep_research.model_registry import (
    get_available_model_combos,
    get_available_models,
    supports_tool_choice,
)
from langchain_core.runnables import RunnableConfig

# Custom CSS for the modern dark-theme sidebar
ui.add_head_html(
    """
    <style>
        /* Sidebar container scroll */
        .sidebar-scroll {
            max-height: 100vh;
            overflow-y: auto;
        }

        /* Primary sidebar card */
        .sidebar-card {
            background: var(--nicegui-default-bg);
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            padding: 16px;
        }

        /* Header block with gradient background */
        .sidebar-header {
            background: linear-gradient(180deg, var(--nicegui-default-bg) 0%, var(--nicegui-default-bg-light) 100%);
            border: 1px solid var(--nicegui-default-border);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            text-align: center;
        }

        .sidebar-header .title {
            font-size: 18px;
            font-weight: 700;
            color: var(--nicegui-default-text);
            margin: 0;
        }

        .sidebar-header .subtitle {
            font-size: 12px;
            font-weight: 500;
            color: var(--nicegui-default-text-secondary);
            margin-top: 4px;
        }

        /* Filter buttons group */
        .filter-buttons-group {
            display: flex;
            flex-direction: column;
            gap: 4px;
            margin-bottom: 16px;
        }

        .filter-buttons-group button {
            background: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 6px !important;
            padding: 12px 16px !important;
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
            transition: all 0.2s ease !important;
            position: relative !important;
        }

        .filter-buttons-group button:hover {
            background: var(--nicegui-default-bg-hover) !important;
            border-color: var(--nicegui-default-border) !important;
            transform: translateY(-1px);
        }

        .filter-buttons-group button.active {
            background: var(--nicegui-default-bg-active) !important;
            font-weight: 600 !important;
            padding-left: 20px !important;
        }

        .filter-buttons-group button.active::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            height: 100%;
            width: 3px;
            background: var(--nicegui-default-primary);
            border-radius: 3px 0 0 3px;
        }

        /* Section controls */
        .section-controls {
            background: var(--nicegui-default-bg-light);
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 12px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        /* Main layout improvements for integrated sidebar */
        .q-page-container, .q-page {
            padding: 0 !important;
            margin: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        
        /* Remove any default body margins */
        body {
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Responsive behavior */
        @media (max-width: 991px) {
            .sidebar-card {
                width: 100% !important;
                margin-bottom: 20px;
            }
        }

        @media (min-width: 992px) {
            .sidebar-card {
                width: 280px !important;
            }
        }
    </style>
    """
)

# Application state with comprehensive configuration
app_state = {
    "current_page": "research",
    "research_topic": "",
    
    # Search Configuration
    "search_provider": "tavily",
    "search_budget": 100,
    "max_search_depth": 2,
    "number_of_queries": 1,
    
    # Report Configuration
    "report_style": "Concise",
    "ask_for_clarification": False,
    "include_sources": True,
    "include_raw_sources": False,
    
    # Model Configuration
    "model_combo": "budget",
    "custom_planner_model": None,
    "custom_writer_model": None,
    "custom_summarizer_model": None,
    
    # System Configuration
    "recursion_limit": 100,
    
    # Runtime State
    "research_in_progress": False,
    "final_report": "",
    "progress": 0,
    "status_message": "Ready to start research",
    "active_sidebar_section": "search", # currently active section in sidebar
}

# Provider display names mapping
PROVIDER_DISPLAY_NAMES = {
    SearchProvider.GOOGLE: "Google Search",
    SearchProvider.DUCKDUCKGO: "DuckDuckGo",
    SearchProvider.ARXIV: "ArXiv",
    SearchProvider.PUBMED: "PubMed",
    SearchProvider.PERPLEXITY: "Perplexity",
    SearchProvider.TAVILY: "Tavily",
    SearchProvider.EXA: "Exa",
    SearchProvider.LINKUP: "Linkup",
    SearchProvider.AZURE: "Azure AI Search",
}

def get_provider_api(provider_name):
    """Get the provider enum value from the display name."""
    return next(
        (
            provider.value
            for provider, name in PROVIDER_DISPLAY_NAMES.items()
            if name == provider_name
        ),
        "tavily",
    )

def create_sidebar():
    """Create a clean, flat sidebar with direct configuration options"""
    with ui.column().classes('w-full').style('width: 280px; min-height: 100vh; background-color: var(--card); border-right: 1px solid var(--border); padding: 24px;'):
        
        # Header
        with ui.column().classes('w-full mb-8'):
            ui.label('Configure your research parameters').classes('text-sm text-gray-400')
        
        # Get available providers for direct selection
        available_providers = get_available_providers()
        available_provider_names = [
            PROVIDER_DISPLAY_NAMES[p]
            for p in available_providers
            if p in PROVIDER_DISPLAY_NAMES
        ] or ["Tavily"]

        default_search_api = os.environ.get("SEARCH_API", "tavily")
        default_search_provider = next(
            (p for p in available_providers if p.value == default_search_api),
            SearchProvider.TAVILY,
        )
        current_provider = PROVIDER_DISPLAY_NAMES.get(default_search_provider, "Tavily")
        
        # Search Provider - styled like expansion item
        with ui.column().classes('w-full mb-4'):
            ui.label('Search Provider').classes('text-white font-medium mb-2')
            provider_select = ui.select(
                options=available_provider_names,
                value=current_provider
            ).classes('w-full expansion-select')
            
            def _select_provider(e):
                app_state['search_provider'] = get_provider_api(e.value)
            
            provider_select.on('change', _select_provider)
        
        # Report Style - styled like expansion item
        with ui.column().classes('w-full mb-4'):
            ui.label('Report Style').classes('text-white font-medium mb-2')
            report_style_options = ["Concise", "Comprehensive", "Executive Summary"]
            style_select = ui.select(
                options=report_style_options,
                value=app_state.get('report_style', 'Concise')
            ).classes('w-full expansion-select')
            
            def _select_style(e):
                app_state['report_style'] = e.value
            
            style_select.on('change', _select_style)

        # Model Configuration - styled like expansion item
        combos = get_available_model_combos()
        if combos:
            combo_opts = {k: v['display_name'] for k, v in combos.items()}
            combo_opts['custom'] = 'Custom'
            current_combo = app_state.get('model_combo', 'budget')
            if current_combo not in combo_opts:
                current_combo = next(iter(combo_opts))
            
            with ui.column().classes('w-full mb-4'):
                ui.label('Model Configuration').classes('text-white font-medium mb-2')
                model_select = ui.select(
                    options=combo_opts,
                    value=current_combo
                ).classes('w-full expansion-select')
                
                def _select_model(e):
                    app_state['model_combo'] = e.value
                
                model_select.on('change', _select_model)

        # Advanced Settings - Simple expansion
        with ui.expansion('Advanced Settings').classes('w-full mt-4'):
            # Search Budget
            with ui.column().classes('w-full mb-4'):
                ui.label('Search Budget').classes('text-white font-medium mb-2')
                _budget = ui.number(value=app_state.get('search_budget', 100), min=1, max=1000, step=10).classes('w-full')
                _budget.on('change', lambda e: app_state.update(search_budget=int(e.value)))  # type: ignore[attr-defined]
                ui.label('Maximum search credits to use').classes('text-xs text-gray-500 mt-1')

            # Max Search Depth
            with ui.column().classes('w-full mb-4'):
                ui.label('Max Search Depth').classes('text-white font-medium mb-2')
                _depth = ui.slider(min=1, max=5, value=app_state.get('max_search_depth', 2), step=1).classes('w-full')
                _depth.on('change', lambda e: app_state.update(max_search_depth=int(e.value)))  # type: ignore[attr-defined]
                ui.label('Higher values = more thorough but slower').classes('text-xs text-gray-500 mt-1')

            # Number of Queries
            with ui.column().classes('w-full mb-4'):
                ui.label('Search Queries').classes('text-white font-medium mb-2')
                _queries = ui.slider(min=1, max=5, value=app_state.get('number_of_queries', 1), step=1).classes('w-full')
                _queries.on('change', lambda e: app_state.update(number_of_queries=int(e.value)))  # type: ignore[attr-defined]
                ui.label('More queries = more comprehensive research').classes('text-xs text-gray-500 mt-1')

            # Report Options
            with ui.column().classes('w-full'):
                ui.label('Report Options').classes('text-white font-medium mb-2')
                _clar_cb = ui.checkbox('Allow AI to ask clarifying questions', value=app_state.get('ask_for_clarification', False)).classes('mb-2')
                _clar_cb.on('change', lambda e: app_state.update(ask_for_clarification=e.value))  # type: ignore[attr-defined]

                _src_cb = ui.checkbox('Include sources and citations', value=app_state.get('include_sources', True)).classes('mb-2')
                _src_cb.on('change', lambda e: app_state.update(include_sources=e.value))  # type: ignore[attr-defined]

                _raw_cb = ui.checkbox('Include full raw sources (large reports)', value=app_state.get('include_raw_sources', False))
                _raw_cb.on('change', lambda e: app_state.update(include_raw_sources=e.value))  # type: ignore[attr-defined]

        # Status Footer
        ui.space()
        with ui.column().classes('w-full pt-4 border-t border-gray-700'):
            ui.label('Status').classes('text-sm font-medium text-gray-400 mb-2')
            ui.label().bind_text_from(app_state, 'status_message').classes('text-xs text-gray-500')

def create_research_workflow_card():
    """Create the research workflow steps card"""
    with ui.card().classes('w-full bg-gray-800 border-gray-700'):
        ui.label('Research Workflow').classes('text-xl font-bold text-white mb-4')
        
        workflow_steps = [
            ('Topic Definition', 'Define your research scope and objectives with precision'),
            ('Source Configuration', 'Configure search parameters and source preferences'),
            ('Analysis Execution', 'AI conducts comprehensive research and data analysis'),
            ('Report Generation', 'Structured report with findings, recommendations, and insights'),
        ]
        
        for i, (title, description) in enumerate(workflow_steps, 1):
            with ui.row().classes('w-full mb-6 items-start'):
                with ui.column().classes('items-center mr-4'):
                    ui.badge(str(i), color='blue').classes('text-white font-bold')
                with ui.column().classes('flex-1'):
                    ui.label(title).classes('font-semibold text-white text-base')
                    ui.label(description).classes('text-gray-400 text-sm mt-1')
        
        ui.separator().classes('border-gray-700 my-4')
        
        with ui.expansion('Professional Research Tools').classes('w-full'):
            ui.label('Advanced configuration options available in sidebar').classes('text-gray-400 text-sm')

def create_main_research_interface():
    """Create the main research interface"""
    with ui.column().classes('flex-1 p-8').style('min-height: 100vh; overflow-x: hidden;'):
        # Header
        with ui.row().classes('w-full items-center mb-8'):
            with ui.column():
                ui.label('Research Platform').classes('text-3xl font-bold text-white')
        
        # Main content area - using flexbox layout that fills available space
        with ui.row().classes('w-full gap-8 items-start').style('min-height: 60vh;'):
            # Left side - Research input (takes up more space)
            with ui.column().classes('flex-1 min-w-0'):
                with ui.card().classes('w-full bg-gray-800 border-gray-700'):
                    ui.label('Research Query').classes('text-white font-medium mb-2')
                    
                    research_input = ui.textarea(
                        placeholder='Enter your research topic or question. Be specific for optimal results.',
                        value=app_state.get('research_topic', '')
                    ).classes('w-full bg-gray-700 text-white custom-textarea-padding')
                    research_input.style('''
                        min-height: 120px; 
                        resize: vertical;
                        scrollbar-width: thin;
                        scrollbar-color: #4a5568 #2d3748;
                    ''')
                    
                    def update_research_topic(e):
                        app_state['research_topic'] = e.value
                    
                    research_input.on('change', update_research_topic)
                    
                    # Start Research Button - prominent and left-aligned
                    with ui.row().classes('w-full mt-8'):
                        start_btn = ui.button('▶ Start Research').classes('text-gray-900 font-semibold px-6 py-3').style('''
                            background-color: #f7fafc;
                            border: 1px solid #e2e8f0;
                            border-radius: 8px;
                            width: 180px;
                            font-size: 14px;
                            transition: all 0.2s ease;
                        ''')
                        start_btn.on('click', start_research)
                        
                        # Progress indicator (hidden by default)
                        with ui.column().classes('ml-6').bind_visibility_from(app_state, 'research_in_progress'):
                            ui.linear_progress().bind_value_from(app_state, 'progress').classes('w-48')
                            ui.label().bind_text_from(app_state, 'status_message').classes('text-gray-400 text-sm')
            
            # Right side - Workflow (fixed width, no flex)
            with ui.column().classes('w-80 flex-shrink-0'):
                create_research_workflow_card()

async def start_research():
    """Start the research process"""
    if not app_state['research_topic'].strip():
        ui.notify('Please enter a research topic', type='negative')
        return
    
    app_state['research_in_progress'] = True
    app_state['progress'] = 0
    app_state['status_message'] = 'Initializing research...'
    
    try:
        # Get model configuration
        available_combos = get_available_model_combos()
        if not available_combos:
            ui.notify('No models available. Please check API configuration.', type='negative')
            return
        
        # Determine model configuration
        combo_key = app_state.get('model_combo', list(available_combos.keys())[0])
        
        if combo_key == 'custom':
            # Use custom model selection
            supervisor_model = app_state.get('custom_planner_model')
            researcher_model = app_state.get('custom_writer_model')
            summarizer_model = app_state.get('custom_summarizer_model')
            
            if not supervisor_model or not researcher_model:
                ui.notify('Please configure custom models in the sidebar', type='negative')
                return
        else:
            # Use predefined combo
            combo = available_combos.get(combo_key, list(available_combos.values())[0])
            supervisor_model = combo['planner']
            researcher_model = combo['writer']
            summarizer_model = combo.get('summarizer')
        
        # Prepare report structure
        report_style_map = {
            'Concise': DEFAULT_REPORT_STRUCTURE,
            'Comprehensive': COMPREHENSIVE_REPORT_STRUCTURE,
            'Executive Summary': EXECUTIVE_SUMMARY_STRUCTURE,
        }
        
        # Economy settings for Tavily
        search_api_config = None
        if app_state['report_style'] == "Concise" and app_state['search_provider'] == "tavily":
            search_api_config = {
                "search_depth": "basic",
                "max_results": 1,
            }
        
        config_values = {
            'search_api': SearchAPI(app_state['search_provider']),
            'planner_provider': supervisor_model.split(':', 1)[0],
            'planner_model': supervisor_model.split(':', 1)[1],
            'writer_provider': researcher_model.split(':', 1)[0],
            'writer_model': researcher_model.split(':', 1)[1],
            'summarization_model_provider': summarizer_model.split(':', 1)[0] if summarizer_model else None,
            'summarization_model': summarizer_model.split(':', 1)[1] if summarizer_model else None,
            'report_structure': report_style_map[app_state['report_style']],
            'search_budget': app_state['search_budget'],
            'max_search_depth': app_state['max_search_depth'],
            'number_of_queries': app_state['number_of_queries'],
            'ask_for_clarification': app_state['ask_for_clarification'],
            'include_source_str': app_state['include_sources'],
            'include_raw_source_details': app_state['include_raw_sources'],
        }
        
        # Add economy search settings if configured
        if search_api_config:
            config_values['search_api_config'] = search_api_config
        
        # Run research
        app_state['status_message'] = 'Conducting research...'
        app_state['progress'] = 0.2
        
        astream_config: RunnableConfig = {
            'configurable': config_values,
            'recursion_limit': app_state['recursion_limit'],
        }
        
        latest_state = None
        async for step_state in graph.astream(
            {'topic': app_state['research_topic']}, 
            config=astream_config, 
            stream_mode='values'
        ):
            latest_state = step_state
            iter_count = get_state_value(step_state, 'search_iterations', 0)
            app_state['status_message'] = f'Research iteration {iter_count}'
            app_state['progress'] = min(0.2 + (iter_count / 5) * 0.7, 0.9)
        
        if latest_state:
            app_state['final_report'] = get_state_value(latest_state, 'final_report', 'No report generated')
            app_state['status_message'] = 'Research completed successfully!'
            app_state['progress'] = 1.0
            ui.notify('Research completed!', type='positive')
        else:
            raise Exception('No results returned from research process')
            
    except Exception as e:
        app_state['status_message'] = f'Error: {str(e)}'
        ui.notify(f'Research failed: {str(e)}', type='negative')
        traceback.print_exc()
    finally:
        app_state['research_in_progress'] = False

@ui.page('/')
@ui.page('/research')
def research_page():
    """Main research page"""
    ui.dark_mode().enable()
    
    # Custom CSS for professional styling
    ui.add_head_html('''
    <style>
        /* === Professional Research Platform Design System === */
        :root {
            /* Core color system (HSL) */
            --background: hsl(220 12% 8%);   /* #141619 */
            --foreground: hsl(220 8% 92%);   /* #E8E9EA */
            --primary: hsl(220 8% 85%);      /* #D4D6D9 */
            --card: hsl(220 10% 10%);        /* #1A1C20 */
            --muted: hsl(220 8% 12%);        /* #1D1F23 */
            --border: hsl(220 8% 18%);       /* #2A2D31 */
            --shadow: hsla(220 12% 4% / 0.8);
        }
        /* Base layout overrides */
        html, body, .nicegui-content {
            background-color: var(--background) !important;
            color: var(--foreground) !important;
            font-family: "Inter", "Segoe UI", Helvetica, Arial, sans-serif;
            font-size: 14px;
            line-height: 1.5;
        }
        /* Ensure full width layout */
        .nicegui-content, .q-page-container, .q-page {
            width: 100% !important;
            max-width: 100% !important;
        }
        h1 {
            font-size: 24px;
            font-weight: 600;
            line-height: 1.2;
            color: var(--primary);
        }
        h2 {
            font-size: 18px;
            font-weight: 600;
            color: var(--primary);
        }
        /* Quasar component theming */
        .q-drawer {
            background-color: var(--card) !important;
            border-right: 1px solid var(--border) !important;
        }
        .q-card {
            background-color: var(--card) !important;
            border: 1px solid var(--border) !important;
            box-shadow: 0 1px 3px var(--shadow) !important;
            border-radius: 8px !important;
        }
        .q-expansion-item__container {
            background-color: var(--card) !important;
            border: 1px solid var(--border) !important;
        }
        .q-expansion-item__header {
            background-color: var(--muted) !important;
        }
        /* Buttons */
        .q-btn {
            background-color: var(--primary) !important;
            color: var(--background) !important;
            transition: opacity 0.2s ease;
        }
        .q-btn:hover {
            opacity: 0.9 !important;
        }
        /* Inputs / Selects */
        .q-field__control, .q-field__native {
            background-color: var(--muted) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            color: var(--foreground) !important;
        }
        
        /* Fix textarea padding specifically - comprehensive targeting */
        textarea, 
        .q-textarea textarea, 
        .q-field__native, 
        .q-field__control textarea,
        .q-textarea .q-field__native,
        .q-textarea .q-field__control,
        .q-textarea .q-field__control .q-field__native,
        .q-field--filled .q-field__control,
        .q-field--filled .q-field__control .q-field__native {
            padding: 12px 16px !important;
            box-sizing: border-box !important;
        }
        
        /* Extra aggressive targeting for textarea padding */
        .q-textarea .q-field__control::before,
        .q-textarea .q-field__control::after {
            padding-left: 16px !important;
            padding-right: 16px !important;
        }
        
        /* Custom class for textarea with proper padding - direct application */
        .custom-textarea-padding textarea,
        .custom-textarea-padding .q-field__native {
            padding: 12px 16px !important;
        }
        
        /* Remove any container padding that creates intermediate layers */
        .custom-textarea-padding .q-field__control {
            padding: 0 !important;
        }
        /* Badges for workflow steps */
        .q-badge {
            background-color: var(--primary) !important;
            color: var(--background) !important;
        }
        
        /* Enhanced scrollbar styling for textarea */
        .q-field__native::-webkit-scrollbar {
            width: 8px;
        }
        
        .q-field__native::-webkit-scrollbar-track {
            background: #2d3748;
            border-radius: 4px;
        }
        
        .q-field__native::-webkit-scrollbar-thumb {
            background: #4a5568;
            border-radius: 4px;
        }
        
        .q-field__native::-webkit-scrollbar-thumb:hover {
            background: #718096;
        }
        
        /* Enhanced card styling with better shadows and borders */
        .q-card {
            background-color: var(--card) !important;
            border: 1px solid var(--border) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
            border-radius: 12px !important;
        }
        
        /* Button hover effects */
        .q-btn:hover {
            opacity: 0.9 !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        
        /* Override dropdown button styles to look like text options */
        .q-expansion-item .q-btn {
            background-color: transparent !important;
            color: var(--foreground) !important;
            border: none !important;
            box-shadow: none !important;
            font-size: 14px !important;
            font-weight: normal !important;
            text-transform: none !important;
            padding: 6px 12px !important;
            border-radius: 4px !important;
            min-height: auto !important;
        }
        
        .q-expansion-item .q-btn:hover {
            background-color: var(--muted) !important;
            color: var(--foreground) !important;
            transform: none !important;
            box-shadow: none !important;
        }
        
        .q-expansion-item .q-btn .q-btn__content {
            font-size: 14px !important;
            font-weight: normal !important;
            text-transform: none !important;
        }
        
        /* Make select dropdowns look EXACTLY like expansion items */
        .expansion-select {
            background-color: var(--card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
        }
        
        .expansion-select .q-field {
            background-color: var(--muted) !important;
            border: none !important;
            border-radius: 6px !important;
            color: var(--foreground) !important;
            padding: 12px 16px !important;
            min-height: 48px !important;
            box-shadow: none !important;
        }
        
        .expansion-select .q-field__control {
            background-color: transparent !important;
            border: none !important;
            border-radius: 6px !important;
            color: var(--foreground) !important;
            padding: 0 !important;
            min-height: 48px !important;
            box-shadow: none !important;
            outline: none !important;
        }
        
        .expansion-select .q-field__native {
            background-color: transparent !important;
            border: none !important;
            color: var(--foreground) !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            padding: 0 !important;
            width: 100% !important;
            box-shadow: none !important;
            outline: none !important;
        }
        
        .expansion-select .q-field__append {
            color: var(--foreground) !important;
            opacity: 0.7 !important;
        }
        
        .expansion-select .q-icon {
            color: var(--foreground) !important;
            font-size: 16px !important;
        }
        
        /* Remove all field decorations */
        .expansion-select .q-field__before,
        .expansion-select .q-field__after,
        .expansion-select .q-field__bottom {
            display: none !important;
        }
        
        /* Remove focus states that don't match expansion */
        .expansion-select .q-field--focused .q-field__control,
        .expansion-select .q-field:focus-within .q-field__control,
        .expansion-select .q-field--focused,
        .expansion-select .q-field:focus-within {
            outline: none !important;
            box-shadow: none !important;
            border: none !important;
        }
        
        /* Remove borders and intermediate containers from textarea */
        .q-textarea .q-field__control {
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        
        .q-textarea .q-field__control:focus-within {
            border: none !important;
            box-shadow: none !important;
        }
        
        /* Remove intermediate wrapper padding and styling */
        .q-textarea .q-field__inner {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        .q-textarea .q-field__marginal {
            display: none !important;
        }
        
        /* Remove field wrapper styling */
        .q-field--filled .q-field__inner {
            padding: 0 !important;
        }
        
        .q-field--filled .q-field__control {
            padding: 0 !important;
        }
        
        /* Style the dropdown popup */
        .q-menu {
            background-color: var(--card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        }
        
        .q-item {
            background-color: transparent !important;
            color: var(--foreground) !important;
            padding: 8px 16px !important;
            min-height: 36px !important;
        }
        
        .q-item:hover {
            background-color: var(--muted) !important;
        }
        
        .q-item.q-manual-focusable--focused {
            background-color: var(--muted) !important;
        }
        
        .q-item__label {
            color: var(--foreground) !important;
            font-size: 14px !important;
        }
    </style>
    ''')
    
    # Create main layout with integrated sidebar
    with ui.row().classes('w-full min-h-screen').style('margin: 0; padding: 0;'):
        create_sidebar()
        create_main_research_interface()

@ui.page('/sources')
def sources_page():
    """Sources management page"""
    ui.dark_mode().enable()
    
    with ui.row().classes('w-full min-h-screen').style('margin: 0; padding: 0;'):
        create_sidebar()
        with ui.column().classes('flex-1 p-8'):
            ui.label('Sources').classes('text-3xl font-bold text-white mb-4')
            ui.label('Manage your research sources and references').classes('text-gray-400')

@ui.page('/reports')
def reports_page():
    """Reports page"""
    ui.dark_mode().enable()
    
    with ui.row().classes('w-full min-h-screen').style('margin: 0; padding: 0;'):
        create_sidebar()
        with ui.column().classes('flex-1 p-8'):
            ui.label('Reports').classes('text-3xl font-bold text-white mb-4')
            ui.label('View and manage your research reports').classes('text-gray-400')
            
            # Show current report if available
            if app_state['final_report']:
                with ui.card().classes('w-full mt-6 bg-gray-800 border-gray-700'):
                    ui.label('Latest Research Report').classes('text-xl font-bold text-white mb-4')
                    ui.markdown(app_state['final_report']).classes('text-gray-300')

@ui.page('/analytics')
def analytics_page():
    """Analytics page"""
    ui.dark_mode().enable()
    
    with ui.row().classes('w-full min-h-screen').style('margin: 0; padding: 0;'):
        create_sidebar()
        with ui.column().classes('flex-1 p-8'):
            ui.label('Analytics').classes('text-3xl font-bold text-white mb-4')
            ui.label('Research analytics and insights').classes('text-gray-400')

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title='Research Platform',
        port=8080,
        host='0.0.0.0',
        dark=True,
        storage_secret='research_platform_secret_key'
    ) 