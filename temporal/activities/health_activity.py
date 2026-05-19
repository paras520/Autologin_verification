"""Activity: URL health check + Playwright page extraction.

This wraps Phase 1 of the verification pipeline.  The activity is sync
(runs in a thread-pool executor) because Playwright uses its own event loop
internally and is not async-safe in the Temporal async context.
"""
from __future__ import annotations

from typing import Any

from temporalio import activity
from temporalio.exceptions import ApplicationError

from src.utils.url_health import check_url_health


@activity.defn
async def health_check_activity(url: str) -> dict[str, Any]:
    """Run URL health check + page extraction for *url*.

    Returns the raw health_result dict produced by check_url_health().
    Raises ApplicationError (non-retryable) for permanent failures such as
    clearly invalid URLs; all other exceptions are left as retryable so
    Temporal's retry policy handles transient network blips.
    """
    if not url or not url.strip():
        raise ApplicationError(
            "health_check_activity received an empty URL",
            type="InvalidInput",
            non_retryable=True,
        )

    try:
        result: dict[str, Any] = await check_url_health(url.strip())
        return result
    except ValueError as exc:
        raise ApplicationError(str(exc), type="InvalidInput", non_retryable=True) from exc
    except Exception as exc:
        # Let Temporal retry transient errors (network timeouts, Playwright crashes, etc.)
        activity.logger.warning("health_check_activity transient error for %s: %s", url, exc)
        raise
