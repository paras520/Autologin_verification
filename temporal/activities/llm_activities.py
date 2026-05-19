"""Activities: LLM-backed verification phases.

Two activities are defined here:
  - audience_classify_activity  — Phase 1.5 customer-facing classifier
  - full_match_activity          — Phase 2 identifier extractor + service matcher

Both are async because the underlying heuristics module routes LLM calls
through Bifrost (see ``langfuse_helper.py``). Bifrost-level retries are kept
low (``BIFROST_MAX_ATTEMPTS``) so Temporal's retry policy remains the primary
source of truth.
"""
from __future__ import annotations

from temporalio import activity
from temporalio.exceptions import ApplicationError

from src.models.request_models import CheckRequest
from src.utils.heuristics import assess_full_match, classify_customer_facing


@activity.defn
async def audience_classify_activity(
    provider: str,
    service_name: str,
    url: str,
    page_result: dict,
    session_id: str = "",
) -> dict:
    """Phase 1.5 — customer-facing audience classifier.

    Returns dict with keys: is_customer_facing, confidence, category, reason, notes.
    Never returns None (fail-open per the original service contract).
    """
    try:
        result = await classify_customer_facing(
            provider=provider,
            service_name=service_name,
            url=url,
            page_result=page_result,
            session_id=session_id,
        )
        return result or {}
    except Exception as exc:
        activity.logger.warning(
            "audience_classify_activity error for %s: %s", url, exc
        )
        # Surface as retryable so Temporal retries on transient LLM errors.
        raise


@activity.defn
async def full_match_activity(
    provider: str,
    service_name: str,
    url: str,
    page_result: dict,
    session_id: str = "",
) -> dict | None:
    """Phase 2 — identifier extractor + service matcher.

    Returns the match_result dict, or None when provider and service_name
    are both empty (nothing to match against).
    """
    if not (provider or service_name):
        activity.logger.info(
            "full_match_activity: skipping — provider and service_name both empty"
        )
        return None

    try:
        result = await assess_full_match(
            provider=provider,
            service_name=service_name,
            url=url,
            page_result=page_result,
            session_id=session_id,
        )
        return result
    except Exception as exc:
        activity.logger.warning(
            "full_match_activity error for %s: %s", url, exc
        )
        raise
