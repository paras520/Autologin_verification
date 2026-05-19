"""Activity: country match (Phase 3 — deterministic, no LLM).

Thin wrapper so the workflow graph is consistent and the step is visible in
the Temporal UI alongside the LLM phases.
"""
from __future__ import annotations

from typing import Any

from temporalio import activity
from temporalio.exceptions import ApplicationError

from src.utils.heuristics import assess_country_match


@activity.defn
def country_match_activity(
    expected_country: str,
    page_result: dict[str, Any],
) -> dict[str, Any]:
    """Phase 3 — deterministic country match.

    Synchronous (no I/O); runs in the thread-pool executor.
    Returns the country_check dict with keys: matched, expected_country,
    expected_score, reason, notes.
    """
    if not expected_country or not expected_country.strip():
        raise ApplicationError(
            "country_match_activity requires a non-empty expected_country",
            type="InvalidInput",
            non_retryable=True,
        )

    try:
        return assess_country_match(
            expected_country=expected_country,
            page_result=page_result,
        )
    except Exception as exc:
        activity.logger.warning("country_match_activity error: %s", exc)
        raise
