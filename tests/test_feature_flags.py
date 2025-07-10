import pytest

from open_deep_research.configuration import (
    WorkflowConfiguration,
    MultiAgentConfiguration,
    SearchAPI,
)


def test_workflow_default_features_pass():
    # Should construct without error
    cfg = WorkflowConfiguration()
    assert cfg.features["human_feedback"] is True


def test_invalid_pair_raises():
    with pytest.raises(ValueError, match="incompatible"):
        WorkflowConfiguration(
            features={"parallel_research": True, "human_feedback": True}
        )


def test_mode_restriction_violation():
    # mcp_support not allowed in workflow mode
    with pytest.raises(ValueError, match="workflow"):
        WorkflowConfiguration(features={"mcp_support": True})


def test_multi_agent_defaults_pass():
    cfg = MultiAgentConfiguration()
    assert cfg.features["parallel_research"] is True


def test_multi_agent_restriction_violation():
    # human_feedback not allowed in multi-agent mode
    with pytest.raises(ValueError):
        MultiAgentConfiguration(features={"human_feedback": True})
