from __future__ import annotations

import logging
from typing import Any

from src.models.request_models import CheckRequest
from src.utils.heuristics import assess_full_match

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
