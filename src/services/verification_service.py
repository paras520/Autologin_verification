from __future__ import annotations

import logging
from datetime import datetime
from urllib.parse import urlparse

from src.models.request_models import CheckRequest
from src.models.response_models import ReturnResponse
from src.services.analysis_service import run_parallel_checks
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
        ">>> /check request  url=%s  provider=%s  service=%s  login_type=%s",
        url,
        payload.provider,
        payload.service_name,
        payload.login_type,
    )

    if not url:
        logger.warning("Rejected: empty URL")
        raise ValueError("Request body must include a non-empty 'url' field.")

    if not _is_valid_url(url):
        logger.warning("Rejected: invalid URL scheme/host — %s", url)
        raise ValueError("URL must include a valid scheme and host.")

    health_result = await check_url_health(url)

    soft_errors = health_result["soft_errors"]
    visible_text = health_result["page_result"]["visible_text"]
    page_result = health_result["page_result"]
    health_check = health_result.get("health") in {"OK", "REDIRECT"}
    token_detected = health_result.get("token_detected")

    logger.info(
        "Health check result: health=%s  status=%s  load_time=%sms  token_detected=%s",
        health_result.get("health"),
        health_result.get("status"),
        health_result.get("load_time_ms"),
        bool(token_detected),
    )

    login_type = payload.login_type.strip().lower()

    tasks_launched = ["main-llm"]
    if login_type in ("direct", "navigation"):
        tasks_launched.append("direct-login-check")
    if payload.provider or payload.service_name:
        tasks_launched.append("provider-service-match")
    logger.info("Launching parallel LLM tasks: %s", ", ".join(tasks_launched))

    llm_decision, direct_login_check, provider_result = await run_parallel_checks(
        payload=payload,
        url=url,
        soft_errors=soft_errors,
        visible_text=visible_text,
        page_result=page_result,
    )
    logger.info("All parallel LLM tasks completed")

    if login_type == "direct":
        direct_check_failed = (
            bool(direct_login_check) and not direct_login_check["is_login_page"]
        )
    elif login_type == "navigation":
        direct_check_failed = (
            bool(direct_login_check) and direct_login_check["is_login_page"]
        )
    else:
        direct_check_failed = False

    if direct_login_check:
        logger.info(
            "Login-type check (%s): is_login_page=%s  score=%s  failed=%s  reason=%s",
            login_type,
            direct_login_check["is_login_page"],
            direct_login_check["score"],
            direct_check_failed,
            direct_login_check["reason"],
        )

    prior_checks_passed = health_check and not direct_check_failed
    provider_service_match = (
        provider_result
        if (payload.provider or payload.service_name) and prior_checks_passed
        else None
    )

    if provider_service_match:
        logger.info(
            "Provider/service match: matched=%s  score=%s  reason=%s",
            provider_service_match["matched"],
            provider_service_match["score"],
            provider_service_match["reason"],
        )
    elif (payload.provider or payload.service_name) and not prior_checks_passed:
        logger.info("Provider/service match result discarded — prior checks failed")

    provider_match_failed = (
        bool(provider_service_match) and not provider_service_match["matched"]
    )

    country_check = None
    country_prior_passed = prior_checks_passed and not provider_match_failed
    if payload.country and payload.country.strip() and country_prior_passed:
        country_check = assess_country_match(
            expected_country=payload.country,
            page_result=page_result,
        )
        logger.info(
            "Country match: matched=%s  expected=%s  detected=%s  "
            "expected_score=%s  foreign_score=%s",
            country_check["matched"],
            country_check["expected_country"],
            country_check["detected_country"],
            country_check["expected_score"],
            country_check["best_foreign_score"],
        )
    elif payload.country and payload.country.strip() and not country_prior_passed:
        logger.info("Country match skipped — prior checks failed")

    country_mismatch = bool(country_check) and country_check["matched"] is False
    country_uncertain = bool(country_check) and country_check["matched"] is None

    notes = []
    if token_detected:
        notes.append(f"token_in_url: {token_detected['summary']}")
    if soft_errors:
        notes.append(f"soft_errors={', '.join(soft_errors)}")
    if llm_decision.session_id:
        notes.append(f"langfuse_session_id={llm_decision.session_id}")
    if llm_decision.prompt_path:
        notes.append(f"prompt_path={llm_decision.prompt_path}")
    if llm_decision.error:
        notes.append(f"llm_error={llm_decision.error}")
    if direct_login_check:
        notes.extend(direct_login_check["notes"])
    if provider_service_match:
        notes.extend(provider_service_match["notes"])
    if country_check:
        notes.extend(country_check["notes"])

    final_inactive_flagged = (
        llm_decision.inactive_flagged
        or direct_check_failed
        or provider_match_failed
        or country_mismatch
        or bool(token_detected)
    )

    if token_detected:
        final_reason = (
            f"URL contains an embedded token that may expire — "
            f"{token_detected['summary']}"
        )
    elif direct_check_failed and login_type == "navigation":
        final_reason = (
            f"Navigation URL is a direct login portal — "
            f"expected an indirect/navigation page. "
            f"{direct_login_check['reason']}"
        )
    elif direct_check_failed:
        final_reason = direct_login_check["reason"]
    elif provider_match_failed:
        final_reason = provider_service_match["reason"]
    elif country_mismatch:
        final_reason = country_check["reason"]
    else:
        final_reason = llm_decision.reason or health_result.get("reason")

    page_match_score = (
        provider_service_match["score"] if provider_service_match else None
    )
    direct_match_score = (
        direct_login_check["score"] if direct_login_check else None
    )

    needs_human_review = (
        bool(llm_decision.error)
        or direct_check_failed
        or provider_match_failed
        or country_mismatch
        or country_uncertain
        or bool(token_detected)
    )

    elapsed_ms = int((datetime.now() - request_start).total_seconds() * 1000)
    logger.info(
        "<<< /check response  url=%s  inactive_flagged=%s  reason=%s  "
        "health_check=%s  page_match_score=%s  direct_match_score=%s  "
        "country_matched=%s  elapsed=%dms",
        url,
        final_inactive_flagged,
        final_reason,
        health_check,
        page_match_score,
        direct_match_score,
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
        marked_for_deletion=final_inactive_flagged,
        errors=llm_decision.error or "",
        time=datetime.now().isoformat(),
    )
