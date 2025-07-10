from __future__ import annotations

"""Track migration progress for dual‐state period.

During the transition from the original ``DeepResearchState`` to the
future ``UnifiedState`` we need to ensure that new fields are
propagated to every workflow node.  This module provides light-weight
book-keeping that can be surfaced in CI.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

__all__ = ["StateFieldMigration", "MigrationTracker"]


@dataclass
class StateFieldMigration:
    """Metadata for a single state-field migration.

    Attributes
    ----------
    field_name
        Name of the *new* field being introduced.
    added_date
        Timestamp when the field first appeared in the codebase.  Used
        only for informational purposes.
    unified_only
        If *True* the field currently exists **only** on the new
        ``UnifiedState``.  Once the field is ported to
        ``DeepResearchState`` this value should be flipped to *False*.
    nodes_updated
        Set of workflow-node names (strings) that have been audited and
        updated to read / write this field correctly.
    nodes_pending
        Counter-set of nodes that still need work.  CI can fail if this
        set is non-empty, guaranteeing forward progress.
    """

    field_name: str
    added_date: datetime
    unified_only: bool = True
    nodes_updated: Set[str] = field(default_factory=set)
    nodes_pending: Set[str] = field(default_factory=set)

    def completion_ratio(self) -> float:
        """Return completion ratio in the *range [0, 1]*."""
        total = len(self.nodes_updated) + len(self.nodes_pending)
        return len(self.nodes_updated) / total if total else 1.0


class MigrationTracker:
    """Central registry that CI pulls from.

    The class variable ``_registry`` is deliberately *not* frozen so
    developers can append new entries in follow-up PRs without touching
    class code.
    """

    # NOTE: keep entries sorted alphabetically for readability.
    _registry: Dict[str, StateFieldMigration] = {
        # Example seed entry – extend in follow-up PRs
        "execution_mode": StateFieldMigration(
            field_name="execution_mode",
            added_date=datetime(2025, 1, 10),
            unified_only=True,
            nodes_updated={"unified_planner"},
            nodes_pending={"workflow_researcher", "compile_report"},
        ),
    }

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    @classmethod
    def status(cls) -> Dict[str, float]:
        """Return ``{field_name: completion_percentage}``."""

        return {
            fname: mig.completion_ratio() * 100 for fname, mig in cls._registry.items()
        }

    # ------------------------------------------------------------------
    @classmethod
    def is_complete(cls) -> bool:
        """True if **all** tracked fields are fully migrated."""

        return all(mig.completion_ratio() == 1.0 for mig in cls._registry.values())

    # ------------------------------------------------------------------
    @classmethod
    def serialize_markdown(cls, path: str | Path) -> None:
        """Emit a quick Markdown report for humans.

        Parameters
        ----------
        path
            Target file-path (.md).  Parents are created automatically.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = ["# Migration Progress\n"]
        for name, mig in sorted(cls._registry.items()):
            pct = mig.completion_ratio() * 100
            lines.append(f"## `{name}` – {pct:.0f}%\n")
            if mig.nodes_pending:
                lines.append(
                    "*Pending nodes*: " + ", ".join(sorted(mig.nodes_pending)) + "\n"
                )
            if mig.nodes_updated:
                lines.append(
                    "*Completed*: " + ", ".join(sorted(mig.nodes_updated)) + "\n"
                )
        path.write_text("\n".join(lines), encoding="utf-8")
