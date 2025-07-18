from __future__ import annotations

"""Adapters that bridge the legacy nodes and the upcoming unified state.

This keeps the migration low-risk: we don’t rewrite proven nodes, we
simply marshal data into/out of the new typed models.
"""

from typing import Any, Dict

from langchain_core.runnables import RunnableConfig

from open_deep_research.feature_compatibility import FeatureCompatibility
from open_deep_research.pydantic_state import DeepResearchState
from open_deep_research.unified_models import PlannerResult

__all__ = ["NodeAdapter"]


class NodeAdapter:
    """Collection of static helpers – not meant to be instantiated."""

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    @staticmethod
    async def workflow_planner(
        state: DeepResearchState, config: RunnableConfig
    ) -> PlannerResult:  # noqa: D401
        """Call the existing *generate_report_plan* and wrap the output."""

        import os

        # Fast path for CI / offline tests – if env var set, synthesize plan
        if os.getenv("ODR_FAST_TEST") == "1":
            from open_deep_research.pydantic_state import Section

            dummy_sections = [
                Section(name="Introduction", description="intro", research=False),
                Section(name="Key Findings", description="findings", research=True),
                Section(name="Conclusion", description="summary", research=False),
            ]
            return PlannerResult(sections=dummy_sections)

        from open_deep_research.graph import generate_report_plan

        raw: Dict[str, Any] = await generate_report_plan(state, config)
        return PlannerResult(sections=raw.get("sections", []))

    @staticmethod
    async def multi_agent_planner(
        state: DeepResearchState, config: RunnableConfig
    ) -> PlannerResult:  # noqa: D401
        """Use the multi-agent *supervisor* to obtain the plan."""

        import os

        # ------------------------------------------------------------------
        # Fast/offline stub – identical to workflow planner dummy path
        # ------------------------------------------------------------------
        if os.getenv("ODR_FAST_TEST") == "1":
            from open_deep_research.pydantic_state import Section

            dummy_sections = [
                Section(name="Intro", description="Overview", research=False),
                Section(name="Research", description="Main body", research=True),
                Section(name="Outro", description="Wrap-up", research=False),
            ]
            return PlannerResult(sections=dummy_sections)

        # ------------------------------------------------------------------
        # Real execution path – call supervisor
        # ------------------------------------------------------------------
        from open_deep_research.multi_agent import supervisor

        try:
            raw: Dict[str, Any] = await supervisor(state, config)  # type: ignore[arg-type]
            # The supervisor returns a new state; extract *sections*
            sections = raw.get("sections") if isinstance(raw, dict) else []
            return PlannerResult(sections=sections or [])
        except Exception as exc:  # noqa: BLE001
            # Defensive fallback – log and delegate to workflow planner
            import logging

            logging.getLogger(__name__).warning(
                "multi_agent_planner failed (%s); falling back to workflow planner", exc
            )
            return await NodeAdapter.workflow_planner(state, config)

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------

    @staticmethod
    async def workflow_researcher(state: DeepResearchState, config: RunnableConfig):
        """Research adapter for workflow mode.

        Behaviour:
        * If ``ODR_FAST_TEST=1`` **or** ``search_api == NONE`` – return
          placeholder content (keeps CI fast/offline).
        * Otherwise – delegate to the legacy "build_section_with_web_research"
          sub-graph for each research section and aggregate the results.
        """

        import os

        from open_deep_research.configuration import SearchAPI, WorkflowConfiguration
        from open_deep_research.unified_models import (
            ResearchResult,
            Section,
        )  # local import to avoid cycles

        # ------------------------------------------------------------------
        # Decide which path to take
        # ------------------------------------------------------------------
        fast_env = os.getenv("ODR_FAST_TEST") == "1"

        # Extract configuration (may be absent)
        cfg = (
            WorkflowConfiguration.from_runnable_config(config)
            if isinstance(config, dict)
            else WorkflowConfiguration()
        )

        if fast_env or cfg.search_api == SearchAPI.NONE:
            # --------------------------------------------------------------
            # Placeholder content path
            # --------------------------------------------------------------
            completed: list[Section] = []
            for sec in state.sections:
                if sec.research:
                    completed.append(
                        Section(
                            name=sec.name,
                            description=sec.description,
                            research=sec.research,
                            content=f"# {sec.name}\n\n(placeholder content)",
                        )
                    )
            return ResearchResult(completed_sections=completed, source_str="")

        # --------------------------------------------------------------
        # Real research path – call legacy sub-graph
        # --------------------------------------------------------------
        from open_deep_research.graph import (
            graph as legacy_graph,
        )  # heavy import only here

        # Honour feature flag – default to *sequential* for backwards-compatibility
        parallel = (
            (
                cfg.features.get("parallel_research", False)
                if cfg.features and FeatureCompatibility.is_allowed(
                    "parallel_research",
                    mode="workflow",
                    active_features=cfg.features,
                )
                else False
            )
        )

        section_runnable = legacy_graph.get_node("build_section_with_web_research")  # type: ignore[attr-defined]

        aggregated: list[Section] = []
        source_accum = ""

        if parallel:
            # ----------------------------------------------------------
            # Run research for all sections concurrently
            # ----------------------------------------------------------
            import asyncio

            # Build coroutines for research sections only
            tasks = [
                section_runnable.ainvoke({"topic": state.topic, "section": sec}, config)
                for sec in state.sections
                if sec.research
            ]

            # Nothing to do – early exit keeps behaviour identical
            if not tasks:
                return ResearchResult(completed_sections=[], source_str="")

            # Execute concurrently and aggregate results preserving order
            results = await asyncio.gather(*tasks)
            for res in results:
                aggregated.extend(res.get("completed_sections", []))
                source_accum += res.get("source_str", "")
        else:
            # ----------------------------------------------------------
            # Fallback: sequential execution (previous behaviour)
            # ----------------------------------------------------------
            for sec in state.sections:
                if not sec.research:
                    continue

                patch = await section_runnable.ainvoke(
                    {"topic": state.topic, "section": sec}, config
                )

                # patch includes completed_sections (list) + source_str
                aggregated.extend(patch.get("completed_sections", []))
                source_accum += patch.get("source_str", "")

        return ResearchResult(completed_sections=aggregated, source_str=source_accum)

    @staticmethod
    async def multi_agent_researcher(state: DeepResearchState, config: RunnableConfig):
        """Research adapter for *multi_agent* execution mode.

        Behaviour mirrors :pymeth:`NodeAdapter.workflow_researcher` for the
        fast/offline path but, in normal operation, delegates the heavy
        lifting to the multi-agent LangGraph workflow defined in
        ``open_deep_research.multi_agent``.  The resulting *completed_sections*
        and *source_str* are normalised into a :class:`~open_deep_research.unified_models.ResearchResult`.
        """

        import os

        from open_deep_research.configuration import SearchAPI, WorkflowConfiguration
        from open_deep_research.unified_models import ResearchResult

        # ------------------------------------------------------------------
        # Fast/offline path – identical stub behaviour as workflow_researcher
        # ------------------------------------------------------------------
        fast_env = os.getenv("ODR_FAST_TEST") == "1"
        cfg = (
            WorkflowConfiguration.from_runnable_config(config)
            if isinstance(config, dict)
            else WorkflowConfiguration()
        )

        if fast_env or cfg.search_api == SearchAPI.NONE:
            # Re-use the existing workflow stub implementation
            return await NodeAdapter.workflow_researcher(state, config)

        # ------------------------------------------------------------------
        # Real execution path – run the multi-agent LangGraph
        # ------------------------------------------------------------------
        from open_deep_research.multi_agent import graph as multi_agent_graph

        try:
            # Execute the graph.  The compiled LangGraph object implements the
            # *Runnable* protocol so we can pass *config* straight through.
            final_state = await multi_agent_graph.ainvoke(state, config)

            # The graph returns an updated state (MultiAgentReportState) that
            # contains the completed sections and, optionally, accumulated
            # source strings.  We defensively access the attributes to remain
            # compatible with future schema tweaks.
            completed_sections = getattr(final_state, "completed_sections", [])
            source_str = getattr(final_state, "source_str", "")

            return ResearchResult(
                completed_sections=completed_sections,  # type: ignore[arg-type]
                source_str=source_str,
            )
        except Exception as exc:  # noqa: BLE001
            # Log and gracefully degrade to the single‐workflow researcher
            import logging

            logging.getLogger(__name__).warning(
                "multi_agent_researcher failed (%s); falling back to workflow researcher",
                exc,
            )
            return await NodeAdapter.workflow_researcher(state, config)
