from __future__ import annotations

import logging
from typing import Any

from src.models.request_models import CheckRequest
from src.utils.heuristics import assess_full_match, classify_customer_facing

logger = logging.getLogger("autologin.analysis_service")


async def run_checks(
    payload: CheckRequest,
    url: str,
    page_result: dict[str, Any],
    session_id: str = "",
) -> dict | None:
    """Run the two-stage match pipeline (cheap extractor → final smart LLM).

    Returns the match_result dict, or None when both provider and
    service_name are empty (nothing to match against).
    """
    if not (payload.provider or payload.service_name):
        logger.info("[analysis] Skipping match — provider and service_name both empty")
        return None

    return await assess_full_match(
        provider=payload.provider,
        service_name=payload.service_name,
        url=url,
        page_result=page_result,
        session_id=session_id,
    )


async def classify_page_audience(
    payload: CheckRequest,
    url: str,
    page_result: dict[str, Any],
    session_id: str = "",
) -> dict:
    """Phase 1.5 — LLM-based classifier that decides whether the page is a
    customer-facing service portal vs HRMS / careers / internal admin /
    vendor portal / marketing-only.

    Returns a dict with keys: is_customer_facing, confidence, category,
    reason, notes. Always returns a dict (fail-open) — never None — so the
    caller can rely on a stable shape.
    """
    return await classify_customer_facing(
        provider=payload.provider or "",
        service_name=payload.service_name or "",
        url=url,
        page_result=page_result,
        session_id=session_id,
    )
