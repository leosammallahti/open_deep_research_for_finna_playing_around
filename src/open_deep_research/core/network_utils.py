from __future__ import annotations

"""Shared network utilities (retry / back-off helpers).

The original upstream repository wraps *every* outbound HTTP or SDK call in a
`tenacity.retry` decorator so the retry/back-off policy lives in a single
place.  This file introduces the same pattern for our fork.

Usage::

    from open_deep_research.core.network_utils import async_retry

    @async_retry()
    async def fetch_page(url: str) -> str:
        ...

or customise policy::

    @async_retry(wait=wait_random_exponential(multiplier=1, max=20))
    async def call_api(...):
        ...

The decorator is thin; it keeps import-time overhead negligible and avoids
coupling utils.py to tenacity.  We will gradually apply it across search
providers in follow-up PRs.
"""

from typing import Any, Awaitable, Callable, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

_T = TypeVar("_T")


def async_retry(
    *,
    attempts: int = 3,
    wait: Any | None = None,
    retry_on: tuple[type[Exception], ...] = (Exception,),
):
    """Return a decorator that retries an async function with back-off.

    Parameters
    ----------
    attempts
        Maximum number of attempts (including first call).
    wait
        Tenacity **wait** strategy.  Defaults to `wait_random_exponential` with
        jitter between 1–10 seconds.
    retry_on
        Tuple of exception classes that trigger a retry.
    """

    if wait is None:
        wait = wait_random_exponential(min=1, max=10)

    def _decorator(func: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
        return retry(
            reraise=True,
            stop=stop_after_attempt(attempts),
            wait=wait,
            retry=retry_if_exception_type(retry_on),
        )(func)

    return _decorator


__all__ = ["async_retry"] 