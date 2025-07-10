import datetime as _dt
import json
import os
import threading
from typing import Any, Dict

"""Minimal utilities for tracking Tavily / model credit spend.

This is **approximate** – Tavily does not yet return a usage field, so we
apply the pricing table from the 2025-05 docs.  Numbers will drift if Tavily
changes pricing, but they give us a useful order-of-magnitude view.

The module writes newline-delimited JSON (one record per credit event) to the
file specified by the env var ``CREDIT_LOG_FILE`` (default
``credit_usage.jsonl``).  Each record looks like::

    {"ts": "2025-07-08T21:55:02.312Z", "provider": "tavily",
     "event": "search", "credits": 3,
     "details": {"query": "…", "search_depth": "advanced"}}

A simple `aggregate()` helper is provided to summarise totals.
"""

_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _log_file() -> str:
    """Return the path for the credit log file.

    This indirection ensures we honor changes to the ``CREDIT_LOG_FILE``
    environment variable *after* module import (e.g. in unit tests that
    monkey-patch the env var).
    """
    return os.getenv("CREDIT_LOG_FILE", "credit_usage.jsonl")


def _append(record: Dict[str, Any]) -> None:
    """Append *record* (dict) to the log file as JSON-line."""
    record = {**record, "ts": _dt.datetime.utcnow().isoformat()}
    with _lock:
        with open(_log_file(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Public API – Tavily
# ---------------------------------------------------------------------------

# Static pricing table (credits per request)
_TAVILY_PRICING = {
    ("search", "basic"): 1,
    ("search", "advanced"): 3,
    ("extract", "basic"): 2,
    ("extract", "advanced"): 3,
}


def record_tavily(event: str, depth: str, count: int = 1, **details: Any) -> None:
    """Record *count* Tavily events of *event* (search|extract) at *depth*.

    Args:
        event: "search" or "extract".
        depth: "basic" or "advanced".
        count: How many calls of this type.
        **details: Extra context to log (query, urls, etc.).
    """
    event = event.lower()
    depth = depth.lower()
    credits_per = _TAVILY_PRICING.get((event, depth), 0)
    _append(
        {
            "provider": "tavily",
            "event": event,
            "depth": depth,
            "count": count,
            "credits": credits_per * count,
            "details": details,
        }
    )


# ---------------------------------------------------------------------------
# Public API – Model usage (tokens / price estimation)
# ---------------------------------------------------------------------------

# For now we just log token counts; pricing varies by provider.


def record_model(
    provider: str, model: str, input_tokens: int, output_tokens: int
) -> None:
    """Log token usage for a model call."""
    _append(
        {
            "provider": provider,
            "model": model,
            "event": "model_call",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    )


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------


def aggregate() -> Dict[str, Any]:
    """Return aggregated totals from the log file.

    (Expensive for large files; fine for dev/analysis.)
    """
    log_path = _log_file()
    if not os.path.exists(log_path):
        return {}
    totals: Dict[str, Any] = {"tavily": 0, "input_tokens": 0, "output_tokens": 0}
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("provider") == "tavily":
                totals["tavily"] += rec.get("credits", 0)
            elif rec.get("event") == "model_call":
                totals["input_tokens"] += rec.get("input_tokens", 0)
                totals["output_tokens"] += rec.get("output_tokens", 0)
    return totals
