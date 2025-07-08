"""Planning, research, and report generation."""

# Expose package version
__version__ = "0.0.15"

# ---------------------------------------------------------------------------
# Early initialisation: load env settings & configure logging once so that any
# downstream imports can rely on them.
# ---------------------------------------------------------------------------

from open_deep_research.core.logging_utils import init_logger  # noqa: E402
from open_deep_research.core.settings import (
    settings,  # noqa: E402 – initialisation import
)

# Configure log level from settings (if the env var has been provided)
init_logger(level=settings.log_level)