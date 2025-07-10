# Contributing to Open Deep Research

Thank you for helping improve the project!  This short guide explains the local workflow, especially our **two-tier test strategy** that keeps CI fast while preserving full coverage.

---

## 1 – Environment setup

```bash
# Clone your fork and install the dev extras
pip install -e ".[dev]"
```

The `dev` extra pulls in tools like **Hypothesis** used for property-based testing.

---

## 2 – Two-tier test suite

| Tier | Command | What runs | Typical duration |
|------|---------|-----------|-------------------|
| **Fast** | `ODR_FAST_TEST=1 pytest -q -m "not slow"` | Smoke tests, unit tests, *excludes* any test marked `slow` | ⩽30 s |
| **Full** | `pytest -q` | Entire suite, including Hypothesis property tests and other `slow` tests | ~2–3 min |

### Marking slow tests

*Resource-intensive* or *stochastic* tests should be marked:

```python
import pytest
pytestmark = pytest.mark.slow
```

The CI workflow executes the **fast tier** only, so these tests must remain green but can take longer.

### Property-based tests & Hypothesis

Property-based tests live in `tests/test_state_property.py`.  They are guarded so that they:

1. **Skip** if Hypothesis isn’t installed (`pytest.importorskip("hypothesis")`).
2. **Skip** automatically when `ODR_FAST_TEST` is set.

---

## 3 – Pre-commit hooks

Install hooks once:

```bash
pre-commit install
```

Hooks run **Ruff** (format + lint) and **MyPy** in strict mode.  They do **not** run the full pytest suite to remain snappy; remember to execute the *full* tests before opening a pull request:

```bash
pytest -q  # or `pytest -m slow` selectively
```

---

## 4 – Coding conventions

The project enforces:

* PEP 8 via Ruff formatter.
* Strict typing via MyPy (`mypy --strict`).
* Immutable state updates using `model_copy()`.

See `.cursorrules` for the complete architecture & style guide.

---

## 5 – Opening a PR

1. Ensure **both** test tiers pass locally.
2. Summarise your changes and reference any issues.
3. Avoid touching legacy code outside the fork unless necessary. 