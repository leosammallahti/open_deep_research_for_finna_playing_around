from __future__ import annotations

"""Declarative feature-compatibility helpers used by configuration classes.

This indirection makes it trivial to expand / tweak the compatibility
matrix without touching validation code scattered across the codebase.
"""

from typing import Dict, List, Tuple, cast

__all__ = ["FeatureCompatibility"]


class FeatureCompatibility:
    """Compatibility rules for optional runtime features."""

    # ------------------------------------------------------------------
    # Pair-wise compatibility matrix
    #   (feature_a, feature_b)  ->  allowed?
    # ------------------------------------------------------------------
    MATRIX: Dict[Tuple[str, str], bool] = {
        ("parallel_research", "human_feedback"): False,
        ("mcp_support", "section_grading"): False,
        ("clarification", "parallel_research"): False,
    }

    # ------------------------------------------------------------------
    # Mode restrictions – maps *execution_mode* -> list[feature] disallowed
    # ------------------------------------------------------------------
    MODE_RESTRICTIONS: Dict[str, List[str]] = {
        "workflow": ["mcp_support"],
        "multi_agent": ["human_feedback", "section_grading"],
        "hybrid": [],
    }

    # ------------------------------------------------------------------
    @classmethod
    def validate_features(cls, features: Dict[str, bool]) -> List[str]:
        """Return a list of detected incompatibilities (empty if valid)."""
        active = [name for name, enabled in features.items() if enabled]
        errors: List[str] = []

        # Pair-wise checks (order-independent)
        for i, feat1 in enumerate(active):
            for feat2 in active[i + 1 :]:
                # Ensure the tuple length is statically recognised as 2 for type checkers
                key = cast(Tuple[str, str], tuple(sorted((feat1, feat2))))
                allowed = cls.MATRIX.get(key, True)
                if not allowed:
                    errors.append(f"'{feat1}' incompatible with '{feat2}'")

        return errors

    # ------------------------------------------------------------------
    @classmethod
    def validate_mode(cls, mode: str, features: Dict[str, bool]) -> List[str]:
        """Validate that *features* are allowed under *mode*."""
        disallowed = cls.MODE_RESTRICTIONS.get(mode, [])
        return [
            f"Mode '{mode}' does not support '{f}'"
            for f in disallowed
            if features.get(f, False)
        ]

    @classmethod
    def is_allowed(
        cls,
        feature: str,
        *,
        mode: str | None = None,
        active_features: Dict[str, bool] | None = None,
    ) -> bool:
        """Return *True* if *feature* can be enabled given *mode* and *active_features*.

        Parameters
        ----------
        feature
            Name of the feature we want to enable.
        mode
            Execution mode ("workflow", "multi_agent", or "hybrid").  If *None* the
            mode‐level restrictions are skipped.
        active_features
            Mapping of *currently* enabled feature flags.  The helper checks
            pair-wise compatibility between *feature* and every *other* active
            feature.  If *None* we assume no other features are active.
        """

        # Check mode restriction first – a single disallow is enough to reject.
        if mode is not None:
            disallowed = cls.MODE_RESTRICTIONS.get(mode, [])
            if feature in disallowed:
                return False

        # If no other active features we are good.
        if not active_features:
            return True

        # Evaluate pair-wise matrix with every *other* active feature that is True.
        for other, enabled in active_features.items():
            if not enabled or other == feature:
                continue
            key = cast(Tuple[str, str], tuple(sorted((feature, other))))
            if cls.MATRIX.get(key, True) is False:
                return False

        return True
