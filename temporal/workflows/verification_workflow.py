"""Workflow: single URL verification pipeline.

This is the durable equivalent of src/services/verification_service.py:verify_url().
Each phase runs as a separate activity so individual phases can be retried
independently and are visible in the Temporal UI.

Phase 1   → health_check_activity      (URL health + Playwright extraction)
Phase 1.5 → audience_classify_activity (LLM customer-facing classifier)
Phase 2   → full_match_activity         (LLM extractor + service matcher)
Phase 3   → country_match_activity      (deterministic country check)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from temporal.activities.country_activity import country_match_activity
    from temporal.activities.health_activity import health_check_activity
    from temporal.activities.llm_activities import (
        audience_classify_activity,
        full_match_activity,
    )
    from temporal.config.settings import (
        COUNTRY_ACTIVITY_TIMEOUT,
        COUNTRY_RETRY_POLICY,
        HEALTH_ACTIVITY_TIMEOUT,
        HEALTH_RETRY_POLICY,
        LLM_ACTIVITY_TIMEOUT,
        LLM_RETRY_POLICY,
    )

AUDIENCE_DELETE_THRESHOLD = 70
_SCRAPING_FAIL_REASONS = frozenset({"TIMEOUT", "HTTP2_ERROR"})


@dataclass
class VerificationInput:
    url: str
    provider: str
    service_name: str
    login_type: str
    country: str
    cb_link_id: str = ""


@dataclass
class VerificationResult:
    url: str
    inactive_flagged: bool
    reason: str | None
    health_check: bool
    page_match_score: float | None
    direct_match_score: float | None
    notes: str | None
    updated_name: str | None
    marked_for_human_review: bool
    marked_for_deletion: bool
    errors: str
    time: str


@workflow.defn
class VerificationWorkflow:
    """Durable single-URL verification workflow."""

    @workflow.run
    async def run(self, inp: VerificationInput) -> VerificationResult:
        url = inp.url.strip()

        # ------------------------------------------------------------------
        # Phase 1 — URL health + page extraction
        # ------------------------------------------------------------------
        health_result: dict[str, Any] = await workflow.execute_activity(
            health_check_activity,
            url,
            start_to_close_timeout=HEALTH_ACTIVITY_TIMEOUT,
            retry_policy=HEALTH_RETRY_POLICY,
        )

        soft_errors: list[str] = health_result.get("soft_errors") or []
        page_result: dict[str, Any] = health_result.get("page_result") or {}
        health_ok: bool = health_result.get("health") in {"OK", "REDIRECT"}
        token_detected: dict | None = health_result.get("token_detected")

        if not health_ok:
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
                reason = (
                    "Server returned an empty (0-byte) response — URL likely "
                    "deprecated or requires a different host. "
                    "Needs human review/replacement."
                )
                notes.append("empty_server_response")
                marked_for_deletion = False
            elif bot_blocked:
                reason = "SERVICE_DECLINED"
                notes.append(
                    "bot_detection_suspected: 403 consistent with WAF/bot-detection "
                    "(Cloudflare, Akamai, etc.) — service likely functional for real users"
                )
                marked_for_deletion = False
            elif scraping_fail:
                reason = "SCRAPING_FAILED"
                notes.append(f"scraping_error={raw_reason}")
                marked_for_deletion = False
            else:
                marked_for_deletion = True

            return VerificationResult(
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

        # ------------------------------------------------------------------
        # Phase 1.5 — LLM audience classifier
        # ------------------------------------------------------------------
        audience: dict[str, Any] = await workflow.execute_activity(
            audience_classify_activity,
            args=[
                inp.provider,
                inp.service_name,
                url,
                page_result,
                inp.cb_link_id,
            ],
            start_to_close_timeout=LLM_ACTIVITY_TIMEOUT,
            retry_policy=LLM_RETRY_POLICY,
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
            return VerificationResult(
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

        # ------------------------------------------------------------------
        # Phase 2 — LLM extractor + service matcher
        # ------------------------------------------------------------------
        match_result: dict | None = await workflow.execute_activity(
            full_match_activity,
            args=[
                inp.provider,
                inp.service_name,
                url,
                page_result,
                inp.cb_link_id,
            ],
            start_to_close_timeout=LLM_ACTIVITY_TIMEOUT,
            retry_policy=LLM_RETRY_POLICY,
        )

        bank_match_failed = bool(match_result) and not match_result.get("bank_matched", True)
        service_match_failed = bool(match_result) and not match_result.get("service_matched", True)
        provider_match_failed = bank_match_failed or service_match_failed
        prior_checks_passed = health_ok and not provider_match_failed

        # ------------------------------------------------------------------
        # Phase 3 — Deterministic country check (skipped when match failed)
        # ------------------------------------------------------------------
        country_check: dict | None = None
        if inp.country and inp.country.strip() and prior_checks_passed:
            country_check = await workflow.execute_activity(
                country_match_activity,
                args=[inp.country, page_result],
                start_to_close_timeout=COUNTRY_ACTIVITY_TIMEOUT,
                retry_policy=COUNTRY_RETRY_POLICY,
            )

        country_mismatch = bool(country_check) and country_check["matched"] is False
        country_uncertain = bool(country_check) and country_check["matched"] is None

        # ------------------------------------------------------------------
        # Phase 4 — Final decision assembly (mirrors verification_service.py)
        # ------------------------------------------------------------------
        notes: list[str] = []
        if token_detected:
            notes.append(f"token_in_url: {token_detected['summary']}")
        if soft_errors:
            notes.append(f"soft_errors={', '.join(soft_errors)}")
        if match_result:
            notes.extend(match_result.get("notes", []))
        if country_check:
            notes.extend(country_check["notes"])

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
            final_reason = (
                (match_result.get("reason") if match_result else None)
                or health_result.get("reason")
            )

        page_match_score = match_result.get("confidence_score") if match_result else None
        marked_for_deletion = bool(token_detected) or bank_match_failed or country_mismatch
        needs_human_review = (
            provider_match_failed
            or country_mismatch
            or country_uncertain
            or bool(token_detected)
            or audience_uncertain_review
        )

        return VerificationResult(
            url=url,
            inactive_flagged=final_inactive_flagged,
            reason=final_reason,
            health_check=health_ok,
            page_match_score=page_match_score,
            direct_match_score=page_match_score,
            notes=" | ".join(notes) or None,
            updated_name=None,
            marked_for_human_review=needs_human_review,
            marked_for_deletion=marked_for_deletion,
            errors="",
            time=datetime.now().isoformat(),
        )
