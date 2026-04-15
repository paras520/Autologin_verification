from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from src.models.request_models import CheckRequest
from src.models.response_models import ReturnResponse
from src.services.analysis_service import run_checks
from src.utils.heuristics import assess_country_match
from src.utils.url_health import check_url_health

logger = logging.getLogger("autologin.verification_service")


def _is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.scheme and parsed.netloc)


async def verify_url(payload: CheckRequest) -> ReturnResponse:
    url = payload.url.strip()
    request_start = datetime.now()

    logger.info(
        ">>> /check request  url=%s  provider=%s  service=%s",
        url,
        payload.provider,
        payload.service_name,
    )

    if not url:
        logger.warning("Rejected: empty URL")
        raise ValueError("Request body must include a non-empty 'url' field.")

    if not _is_valid_url(url):
        logger.warning("Rejected: invalid URL scheme/host — %s", url)
        raise ValueError("URL must include a valid scheme and host.")

    # -------------------------------------------------------------------------
    # Phase 1 — URL health + page extraction
    # -------------------------------------------------------------------------
    health_result = await check_url_health(url)

    soft_errors = health_result.get("soft_errors") or []
    page_result = health_result.get("page_result") or {}
    health_check = health_result.get("health") in {"OK", "REDIRECT"}
    token_detected = health_result.get("token_detected")

    logger.info(
        "[health] url=%s  ok=%s  status=%s  time=%sms",
        url,
        health_result.get("health"),
        health_result.get("status"),
        health_result.get("load_time_ms"),
    )

    if not health_check:
        reason = health_result.get("reason") or "URL is unreachable or returned an error"
        notes = []
        if token_detected:
            notes.append(f"token_in_url: {token_detected['summary']}")
            reason = (
                f"URL contains an embedded token that may expire — "
                f"{token_detected['summary']}"
            )
        if soft_errors:
            notes.append(f"soft_errors={', '.join(soft_errors)}")

        elapsed_ms = int((datetime.now() - request_start).total_seconds() * 1000)
        logger.info(
            "<<< /check early exit (health failed)  url=%s  reason=%s  elapsed=%dms",
            url, reason, elapsed_ms,
        )
        return ReturnResponse(
            url=url,
            inactive_flagged=True,
            reason=reason,
            health_check=False,
            page_match_score=None,
            direct_match_score=None,
            notes=" | ".join(notes) or None,
            updated_name=None,
            marked_for_human_review=True,
            marked_for_deletion=True,
            errors="",
            time=datetime.now().isoformat(),
        )

    # -------------------------------------------------------------------------
    # Phase 2 — Two-stage match pipeline (cheap extractor → final smart LLM)
    # -------------------------------------------------------------------------
    match_result = await run_checks(
        payload=payload,
        url=url,
        page_result=page_result,
        session_id=payload.cb_link_id,
    )

    detected_login_type = (
        match_result.get("login_type", payload.login_type).strip().lower()
        if match_result
        else payload.login_type.strip().lower()
    )

    bank_match_failed = bool(match_result) and not match_result.get("bank_matched", True)
    service_match_failed = bool(match_result) and not match_result.get("service_matched", True)
    provider_match_failed = bank_match_failed or service_match_failed

    # -------------------------------------------------------------------------
    # Phase 3 — Country match (deterministic, skipped if match already failed)
    # -------------------------------------------------------------------------
    prior_checks_passed = health_check and not provider_match_failed

    country_check = None
    if payload.country and payload.country.strip() and prior_checks_passed:
        country_check = assess_country_match(
            expected_country=payload.country,
            page_result=page_result,
        )
        logger.info(
            "[country-match] expected=%s  score=%s  matched=%s",
            country_check["expected_country"],
            country_check["expected_score"],
            country_check["matched"],
        )

    country_mismatch = bool(country_check) and country_check["matched"] is False
    country_uncertain = bool(country_check) and country_check["matched"] is None

    # -------------------------------------------------------------------------
    # Phase 4 — Final decision assembly
    # -------------------------------------------------------------------------
    notes = []
    if token_detected:
        notes.append(f"token_in_url: {token_detected['summary']}")
    if soft_errors:
        notes.append(f"soft_errors={', '.join(soft_errors)}")
    if match_result:
        notes.extend(match_result.get("notes", []))
    if country_check:
        notes.extend(country_check["notes"])

    final_inactive_flagged = (
        provider_match_failed
        or country_mismatch
        or bool(token_detected)
    )

    # token > bank mismatch > service mismatch > country mismatch > match reason > health reason
    if token_detected:
        final_reason = (
            f"URL contains an embedded token that may expire — "
            f"{token_detected['summary']}"
        )
    elif bank_match_failed:
        final_reason = match_result.get("reason") or "Page does not match the expected bank."
    elif service_match_failed:
        final_reason = match_result.get("reason") or "Page does not match the expected service."
    elif country_mismatch:
        final_reason = country_check["reason"]
    else:
        final_reason = (match_result.get("reason") if match_result else None) or health_result.get("reason")

    page_match_score = match_result.get("confidence_score") if match_result else None
    direct_match_score = page_match_score

    # token / bank / country mismatch → deletion candidate; name issue → inactive only, no deletion
    marked_for_deletion = (
        bool(token_detected)
        or bank_match_failed
        or country_mismatch
    )

    needs_human_review = (
        provider_match_failed
        or country_mismatch
        or country_uncertain
        or bool(token_detected)
    )

    elapsed_ms = int((datetime.now() - request_start).total_seconds() * 1000)
    logger.info(
        "<<< /check response  url=%s  inactive_flagged=%s  reason=%s  "
        "health_check=%s  page_match_score=%s  login_type=%s  "
        "country_matched=%s  elapsed=%dms",
        url,
        final_inactive_flagged,
        final_reason,
        health_check,
        page_match_score,
        detected_login_type,
        country_check["matched"] if country_check else "n/a",
        elapsed_ms,
    )

    return ReturnResponse(
        url=url,
        inactive_flagged=final_inactive_flagged,
        reason=final_reason,
        health_check=health_check,
        page_match_score=page_match_score,
        direct_match_score=direct_match_score,
        notes=" | ".join(notes) or None,
        updated_name=None,
        marked_for_human_review=needs_human_review,
        marked_for_deletion=marked_for_deletion,
        errors="",
        time=datetime.now().isoformat(),
    )


async def verify_row(row: dict[str, Any]) -> ReturnResponse:
    """Run the full verification pipeline for a single DB row."""
    cb_link_id = row.get("cb_link_id") or ""
    payload = CheckRequest(
        provider=cb_link_id,
        service_name=row.get("login_service") or "",
        login_type="direct",
        url=row.get("login_url") or "",
        country="india",
        cb_link_id=cb_link_id,
    )
    return await verify_url(payload)
