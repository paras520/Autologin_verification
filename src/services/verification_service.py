from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from src.models.request_models import CheckRequest
from src.models.response_models import ReturnResponse
from src.services.analysis_service import classify_page_audience, run_checks
from src.utils.heuristics import assess_country_match
from src.utils.url_health import check_url_health

logger = logging.getLogger("autologin.verification_service")

# Phase 1.5 — only short-circuit (delete) when the customer-facing classifier
# is at least this confident the page is non-customer-facing. Below this
# threshold the verdict is treated as uncertain and routed to human review.
AUDIENCE_DELETE_THRESHOLD = 70

# Network errors that indicate a transient technical scraping failure rather
# than a definitively dead URL. These route to human review instead of deletion.
_SCRAPING_FAIL_REASONS = frozenset({"TIMEOUT", "HTTP2_ERROR"})


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
    token_detected = None  # disabled — token detection re-enabled later
    # Use the final resolved URL for all LLM calls so domain/path matching is
    # accurate even when the original URL redirects (http→https, subdomain hops, etc.)
    effective_url = page_result.get("final_url") or url

    logger.info(
        "[health] url=%s  ok=%s  status=%s  time=%sms",
        url,
        health_result.get("health"),
        health_result.get("status"),
        health_result.get("load_time_ms"),
    )

    if not health_check:
        raw_reason = health_result.get("reason") or "URL is unreachable or returned an error"
        reason = raw_reason
        notes = []
        if token_detected:
            notes.append(f"token_in_url: {token_detected['summary']}")
            reason = (
                f"URL contains an embedded token that may expire — "
                f"{token_detected['summary']}"
            )
        if soft_errors:
            notes.append(f"soft_errors={', '.join(soft_errors)}")

        empty_server  = raw_reason == "EMPTY_SERVER_RESPONSE"
        bot_blocked   = raw_reason == "BOT_BLOCKED"
        scraping_fail = raw_reason in _SCRAPING_FAIL_REASONS

        if empty_server:
            # 2xx but 0-byte body — not actively dead, needs a human to find
            # the correct replacement host.
            reason = (
                "Server returned an empty (0-byte) response — URL likely "
                "deprecated or requires a different host. "
                "Needs human review/replacement."
            )
            notes.append("empty_server_response")
            marked_for_deletion = False
        elif bot_blocked:
            # Bank WAF detected our automation — service is functional for real
            # users. Needs human verification, not deletion.
            reason = "SERVICE_DECLINED"
            notes.append(
                "bot_detection_suspected: 403 consistent with WAF/bot-detection "
                "(Cloudflare, Akamai, etc.) — service likely functional for real users"
            )
            marked_for_deletion = False
        elif scraping_fail:
            # Transient technical failure — URL may still be live.
            reason = "SCRAPING_FAILED"
            notes.append(f"scraping_error={raw_reason}")
            marked_for_deletion = False
        else:
            marked_for_deletion = True

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
            marked_for_deletion=marked_for_deletion,
            errors="",
            time=datetime.now().isoformat(),
        )

    # -------------------------------------------------------------------------
    # Phase 1.5 — LLM customer-facing audience classifier
    # Decides whether the page is for end customers (login portal, account
    # access, etc.) or non-customer-facing (HRMS, careers, internal admin,
    # vendor portal, marketing-only).
    # -------------------------------------------------------------------------
    audience = await classify_page_audience(
        payload=payload,
        url=effective_url,
        page_result=page_result,
        session_id=payload.cb_link_id,
    )

    audience_is_non_customer = bool(audience) and audience.get("is_customer_facing") is False
    audience_confidence = int(audience.get("confidence", 0)) if audience else 0
    audience_category = audience.get("category", "unknown") if audience else "unknown"
    audience_reason = audience.get("reason", "") if audience else ""
    audience_high_confidence_delete = (
        audience_is_non_customer and audience_confidence >= AUDIENCE_DELETE_THRESHOLD
    )
    audience_uncertain_review = (
        audience_is_non_customer and audience_confidence < AUDIENCE_DELETE_THRESHOLD
    )

    logger.info(
        "[audience] is_customer_facing=%s  confidence=%d  category=%s",
        audience.get("is_customer_facing") if audience else None,
        audience_confidence,
        audience_category,
    )

    if audience_high_confidence_delete:
        reason = (
            f"Page is not a customer-facing service "
            f"({audience_category}): {audience_reason}".rstrip(": ").strip()
        )
        notes_15 = [
            f"audience={audience_category}",
            f"audience_confidence={audience_confidence}",
        ]
        if audience.get("notes"):
            notes_15.extend(audience["notes"])
        elapsed_ms = int((datetime.now() - request_start).total_seconds() * 1000)
        logger.info(
            "<<< /check early exit (non-customer-facing)  url=%s  reason=%s  elapsed=%dms",
            url, reason, elapsed_ms,
        )
        return ReturnResponse(
            url=url,
            inactive_flagged=True,
            reason=reason,
            health_check=True,
            page_match_score=None,
            direct_match_score=None,
            notes=" | ".join(notes_15) or None,
            updated_name=None,
            marked_for_human_review=False,
            marked_for_deletion=True,
            errors="",
            time=datetime.now().isoformat(),
        )

    # -------------------------------------------------------------------------
    # Phase 2 — Two-stage match pipeline (cheap extractor → final smart LLM)
    # -------------------------------------------------------------------------
    match_result = await run_checks(
        payload=payload,
        url=effective_url,
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

    # Phase 1.5 audience verdict — propagate into notes so reviewers see the
    # classifier's call even on pages that did NOT short-circuit.
    if audience_uncertain_review:
        notes.append(f"audience_uncertain={audience_category}")
        notes.append(f"audience_confidence={audience_confidence}")
        if audience_reason:
            notes.append(f"audience_reason={audience_reason}")
    elif audience and audience.get("is_customer_facing") is True:
        notes.append(
            f"audience={audience_category} (customer-facing, conf={audience_confidence})"
        )

    final_inactive_flagged = (
        provider_match_failed
        or country_mismatch
        or bool(token_detected)
        or audience_uncertain_review
    )

    # token > bank mismatch > service mismatch > country mismatch > uncertain audience > match reason > health reason
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
    elif audience_uncertain_review:
        final_reason = (
            f"Page may not be customer-facing ({audience_category}, "
            f"conf={audience_confidence}) — needs human review. {audience_reason}"
        ).strip()
    else:
        final_reason = (match_result.get("reason") if match_result else None) or health_result.get("reason")

    page_match_score   = match_result.get("confidence_score")     if match_result else None
    direct_match_score = match_result.get("url_confidence_score") if match_result else None

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
        or audience_uncertain_review
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
    # display_name (e.g. "Kotak Mutual Fund") carries the bank name; login_service
    # alone (e.g. "Mutual Fund") is too generic for the LLM to match correctly.
    provider = row.get("display_name") or row.get("login_service") or cb_link_id
    payload = CheckRequest(
        provider=provider,
        service_name=row.get("login_service") or "",
        login_type="direct",
        url=row.get("login_url") or "",
        country="india",
        cb_link_id=cb_link_id,
    )
    return await verify_url(payload)
