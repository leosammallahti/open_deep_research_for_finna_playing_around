class OutOfBudgetError(RuntimeError):
    """Raised when the search budget for a workflow run is exhausted."""

    def __init__(self, remaining: int = 0):  # noqa: D401, ANN001
        super().__init__(
            f"Search budget exhausted – {remaining} credits remaining. Increase the budget or reduce search depth/results."
        )
        self.remaining = remaining


class FatalModelError(RuntimeError):
    """Raised when an LLM call fails in a non-retryable way (e.g. bad API key)."""

    def __init__(self, message: str, *, provider: str | None = None):  # noqa: D401, ANN001
        prefix = f"[{provider}] " if provider else ""
        super().__init__(f"{prefix}{message}")
        self.provider = provider
