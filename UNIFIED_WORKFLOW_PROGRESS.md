# Unified Workflow Migration – Progress & Next Steps

_Last updated: 2025-07-10 (session 2)_

## Overview
We are migrating Open Deep Research from multiple parallel workflows (`graph.py`, `multi_agent.py`, legacy `workflow/workflow.py`) into **one configurable, typed LangGraph workflow**.  The goal is to provide

* 📐 **Type-safety** via immutable Pydantic `DeepResearchState`  
* 🔌 **Pluggable execution modes** (single-workflow, multi-agent) behind adapters  
* 🛠️ **Feature flags & compatibility checks** for experimental options  
* 🚀 **Fast/offline CI** with the `ODR_FAST_TEST=1` stub path

## What’s Done so Far
1. **Architecture agreed** – pros/cons and incremental rollout plan documented.
2. **Foundational tooling**  
   * `migration/state_tracker.py` for field-level progress tracking.  
   * `feature_compatibility.py` to validate feature flag combinations.  
   * `LazyNodeLoader` stub + `LazyLoadError` for deferred imports.
3. **Configuration updates**  
   * `WorkflowConfiguration` & `MultiAgentConfiguration` now have `features` dict and merging logic.  
   * Added `search_api` enum (none/perplexity/tavily/…).
4. **Typed models / adapters**  
   * `unified_models.py` ➜ `PlannerResult`, `ResearchResult`, `RouterDecision`.  
   * `node_adapter.py` bridges legacy planner/research nodes ↔ new state.
5. **Minimal unified graph** (`src/open_deep_research/workflow/`)
   * `unified_planner.py`  
   * `unified_researcher.py`  
   * `unified_compiler.py`  
   * `unified_workflow.py` (START → plan → research → compile → END)
6. **Researcher adapter enhanced (previous session)**  
   * Uses **real** `build_section_with_web_research` when _not_ in fast mode and `search_api ≠ none`.  
   * Falls back to placeholder sections for CI/offline.  
   * Correctly accumulates `source_str` for citations.
7. **Multi-agent parity (this session)**  
   * Full implementation of `NodeAdapter.multi_agent_researcher` incl. fast-mode stub.  
   * Planner & researcher nodes now respect `execution_mode="multi_agent"` and `features["mcp_support"]`.  
   * Added `tests/test_multi_agent_adapter.py` covering stub + real paths.
8. **Tests & CI**  
   * Smoke test (`tests/test_unified_workflow_smoke.py`) ensures graph runs in fast mode.  
   * New offline-no-search test validates placeholder path.  
   * GitHub Action runs Ruff, MyPy (strict), pytest in fast mode, and migration completeness check.

## Outstanding Work
The items below are **not** blocking CI but required for full feature parity.

| Priority | Task | Owner | Notes |
|----------|------|-------|-------|
| ~~🔴 High | Replace stub researcher content with call to `build_section_with_web_research` **in parallel** (async gather) | — | Will unlock faster prod runs ~~ **✅ Implemented 2025-07-10 — behind `parallel_research` feature flag** |
| 🟠 Med  | ~~Add citation numbering + raw source inclusion in `unified_compiler.py`~~ **✅ Implemented 2025-07-10** | — | Uses `format_utils.extract_unique_urls` for numbering |
| ~~🟠 Med  | Flesh out **multi-agent** path in `NodeAdapter.multi_agent_*`~~ **✅ Implemented 2025-07-10 (session 2)** | — | Researcher adapter + routing + tests |
| 🟡 Low  | Documentation pass across README / Architecture docs | — | Link to this file |
| 🟡 Low  | Remove legacy `workflow/` directory once parity reached | — | Requires migration tracker 100% |

## Developer Notes
* Use `ODR_FAST_TEST=1` for offline/quick iterations.  
* Config overrides can be passed via `RunnableConfig` ➜ `{"configurable": {...}}`.  
* Keep new fields _immutable_ via `model_copy()` when updating state.

## Recent Feedback Incorporated
> “Incrementally build the unified workflow; ensure fast mode stubs remain, but wire the real researcher so we can validate actual content early. Provide a clear todo list for citations and parallelisation.”

This session delivered the researcher wiring and added the **offline test** to lock behaviour in place.

---

### How to Continue
1. Open a new chat and import this file for context.  
2. Pick an item from **Outstanding Work** (e.g., parallelised research).  
3. Update code + tests; increment migration tracker if new fields covered.  
4. Run `pytest -q` with and without `ODR_FAST_TEST` to validate. 

## Implementation Guidelines & Constraints

To keep the migration smooth and reviewable we follow these **non-negotiable rules** (summarised from earlier reviewer feedback and project memories):

1. 🧩 **Incremental, test-driven steps**  
   • Land small, isolated PRs.  
   • Add/adjust **pytest** coverage for every new node or reducer.  
   • Keep `ODR_FAST_TEST`-driven paths green in CI before tackling real-API behaviour.

2. 🔒 **Do _not_ modify upstream/library code**  
   • Touch only files introduced inside this fork unless absolutely required.  
   • If you _must_ change legacy code, wrap it behind adapters so the original logic stays intact.

3. 🐛 **Fix root causes, not symptoms**  
   • Before patching a failing test, audit the state reducer / node logic for the underlying race or type bug.

4. ✨ **Type-safety first**  
   • Pydantic models are `frozen=True`; always clone with `model_copy(update=...)`.  
   • MyPy runs in strict mode – keep the build green.

5. 🔁 **Reducer etiquette**  
   • Single-value fields ➜ use `ReplaceFn` (last-write-wins).  
   • Accumulators (lists) ➜ provide custom merge fn (e.g. `add_sections`).  
   • Numeric counters ➜ `MaxFn` / `MinFn` as appropriate.

6. 🚦 **CI expectations**  
   • `ruff format --check`, `ruff`, `mypy`, `pytest -q` all must pass.  
   • `MigrationTracker.is_complete()` gate will fail the build if any old field is still untouched – increment progress when you migrate something.

7. 🌐 **Fast vs Real mode**  
   • Keep the stubbed planner/researcher path (`ODR_FAST_TEST=1`) working _always_.  
   • When adding real API calls, guard them behind `search_api != none` so offline contributors aren’t blocked.

8. 📚 **Documentation**  
   • Update this file **and** code-level docstrings in the same PR.  
   • Larger design decisions → `ARCHITECTURE.md` or new ADR-style markdown.

Refer back to this checklist before opening a PR to ensure consistency with project guidelines. 