import pytest
from open_deep_research.utils import TokenBudgetManager

def test_budget_allocation():
    mgr = TokenBudgetManager(total=100)
    assert mgr.allocate(30) == 30
    assert mgr.remaining == 70
    assert not mgr.exhausted()
    assert mgr.allocate(80) == 70  # only remaining granted
    assert mgr.remaining == 0
    assert mgr.exhausted()

    with pytest.raises(ValueError):
        mgr.allocate(-5)

    with pytest.raises(ValueError):
        TokenBudgetManager(total=-1) 