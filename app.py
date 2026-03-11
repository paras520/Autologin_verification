from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.heuristics import assess_direct_login_page, assess_provider_service_match
from src.url_health import check_url_health

load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# Logging setup — console + rotating file
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("autologin.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("autologin.app")

app = FastAPI(title="URL Verification API")
DEFAULT_LANGFUSE_PROMPT_PATH = "autologin/login-portal-classifier"
LLM_VISIBLE_TEXT_LIMIT = 40000


class CheckRequest(BaseModel):
    provider: str
    service_name: str
    login_type: str 
    url: str


# DO NOT TOUCH THIS STRUCTURE, IT IS USED FOR THE RETURN RESPONSE
class ReturnResponse(BaseModel):
    url: str
    inactive_flagged: bool
    reason: str | None
    health_check: bool
    page_matching_score: int | None
    notes: str | None
    updated_name: str | None
    marked_for_human_review: bool
    marked_for_deletion: bool
    errors: str
    time: str



def _is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.scheme and parsed.netloc)


def _normalize_proxy_env() -> None:
    if not os.getenv("LITELLM_PROXY_URL") and os.getenv("LITELLM_PROXY"):
        os.environ["LITELLM_PROXY_URL"] = os.getenv("LITELLM_PROXY", "")


def _get_prompt_path() -> str:
    return os.getenv("LANGFUSE_PROMPT_PATH", DEFAULT_LANGFUSE_PROMPT_PATH)


def _langfuse_is_configured() -> bool:
    required_env_vars = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    ]
    return all(os.getenv(env_var) for env_var in required_env_vars)


class LLMDecision(BaseModel):
    inactive_flagged: bool
    reason: str | None
    raw_output: dict[str, Any] | str | None
    error: str | None
    session_id: str | None
    prompt_path: str


def _coerce_yes_no_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None

    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    return None


def _build_llm_variables(
    url: str,
    soft_errors: list[str],
    visible_text: str,
    page_result: dict[str, Any],
) -> tuple[dict[str, str], str]:
    llm_visible_text = visible_text[:LLM_VISIBLE_TEXT_LIMIT]
    truncated_for_llm = len(visible_text) > LLM_VISIBLE_TEXT_LIMIT

    page_metadata = {
        "title": page_result.get("title"),
        "headings": page_result.get("headings"),
        "buttons": page_result.get("buttons"),
        "login_form_present": page_result.get("login_form_present"),
        "final_url": page_result.get("final_url"),
        "visible_text_length": page_result.get("visible_text_length"),
        "visible_text_truncated": page_result.get("visible_text_truncated"),
        "visible_text_truncated_for_llm": truncated_for_llm,
    }

    variables = {
        "url": url,
        "soft_errors": json.dumps(soft_errors, ensure_ascii=False),
        "visible_text": llm_visible_text,
        "page_metadata": json.dumps(page_metadata, ensure_ascii=False),
    }

    return variables, llm_visible_text


def _load_langfuse_helpers():
    from langfuse_helper import (
        build_messages,
        call_litellm,
        get_prompts_from_langfuse,
        parse_response,
    )

    return build_messages, call_litellm, get_prompts_from_langfuse, parse_response


async def run_langfuse_analysis(
    url: str,
    soft_errors: list[str],
    visible_text: str,
    page_result: dict[str, Any],
) -> LLMDecision:
    prompt_path = _get_prompt_path()

    if not _langfuse_is_configured():
        logger.warning("Langfuse not configured — skipping main LLM analysis")
        return LLMDecision(
            inactive_flagged=False,
            reason=None,
            raw_output=None,
            error="Langfuse credentials are not configured.",
            session_id=None,
            prompt_path=prompt_path,
        )

    _normalize_proxy_env()
    session_id = f"url-check-{uuid4()}"
    variables, llm_visible_text = _build_llm_variables(url, soft_errors, visible_text, page_result)

    logger.info("[LLM-main] Calling Langfuse prompt=%s  session=%s  url=%s", prompt_path, session_id, url)

    try:
        build_messages, call_litellm, get_prompts_from_langfuse, parse_response = _load_langfuse_helpers()
        system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
            prompt_path=prompt_path,
            session_id=session_id,
            variables=variables,
        )

        messages = build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        logger.debug("[LLM-main] Sending %d messages to model=%s", len(messages), config.get("model"))

        response = await call_litellm(
            config=config,
            messages=messages,
            session_id=session_id,
            api_endpoint="/check",
            prompt=prompt_obj,
        )

        parsed_output = parse_response(response, has_functions=False, has_tools=False)
        logger.info("[LLM-main] Raw response: %s", str(parsed_output)[:500])

        decision_value = None
        decision_reason = None

        if isinstance(parsed_output, dict):
            decision_value = _coerce_yes_no_flag(parsed_output.get("inactive_flagged"))
            decision_reason = parsed_output.get("reason")

        if decision_value is None:
            logger.warning("[LLM-main] Could not extract inactive_flagged from response")
            return LLMDecision(
                inactive_flagged=False,
                reason=None,
                raw_output=parsed_output,
                error="LLM response did not contain a valid inactive_flagged yes/no value.",
                session_id=session_id,
                prompt_path=prompt_path,
            )

        logger.info("[LLM-main] Decision: inactive_flagged=%s  reason=%s", decision_value, decision_reason)

        return LLMDecision(
            inactive_flagged=decision_value,
            reason=decision_reason,
            raw_output=parsed_output,
            error=None,
            session_id=session_id,
            prompt_path=prompt_path,
        )
    except Exception as exc:
        logger.error("[LLM-main] LLM call failed: %s", exc, exc_info=True)
        return LLMDecision(
            inactive_flagged=False,
            reason=None,
            raw_output={
                "soft_errors": soft_errors,
                "visible_text_preview": llm_visible_text[:500],
            },
            error=str(exc),
            session_id=session_id,
            prompt_path=prompt_path,
        )


@app.post("/check", response_model=ReturnResponse)
async def check_url(payload: CheckRequest) -> ReturnResponse:
    url = payload.url.strip()
    request_start = datetime.now()

    logger.info(
        ">>> /check request  url=%s  provider=%s  service=%s  login_type=%s",
        url, payload.provider, payload.service_name, payload.login_type,
    )

    if not url:
        logger.warning("Rejected: empty URL")
        raise HTTPException(
            status_code=400,
            detail="Request body must include a non-empty 'url' field.",
        )

    if not _is_valid_url(url):
        logger.warning("Rejected: invalid URL scheme/host — %s", url)
        raise HTTPException(
            status_code=400,
            detail="URL must include a valid scheme and host.",
        )

    health_result = await check_url_health(url)

    soft_errors = health_result["soft_errors"]
    visible_text = health_result["page_result"]["visible_text"]
    page_result = health_result["page_result"]
    health_check = health_result.get("health") in {"OK", "REDIRECT"}

    logger.info(
        "Health check result: health=%s  status=%s  load_time=%sms",
        health_result.get("health"),
        health_result.get("status"),
        health_result.get("load_time_ms"),
    )

    is_direct = payload.login_type.strip().lower() == "direct"
    has_provider_or_service = bool(payload.provider or payload.service_name)

    tasks_launched = ["main-llm"]
    if is_direct:
        tasks_launched.append("direct-login-check")
    if has_provider_or_service:
        tasks_launched.append("provider-service-match")
    logger.info("Launching parallel LLM tasks: %s", ", ".join(tasks_launched))

    llm_coro = run_langfuse_analysis(url, soft_errors, visible_text, page_result)

    direct_coro = (
        assess_direct_login_page(
            provider=payload.provider,
            service_name=payload.service_name,
            page_result=page_result,
        )
        if is_direct
        else asyncio.sleep(0)
    )

    provider_coro = (
        assess_provider_service_match(
            provider=payload.provider,
            service_name=payload.service_name,
            page_result=page_result,
        )
        if has_provider_or_service
        else asyncio.sleep(0)
    )

    llm_decision, direct_result, provider_result = await asyncio.gather(
        llm_coro, direct_coro, provider_coro
    )

    logger.info("All parallel LLM tasks completed")

    direct_login_check = direct_result if is_direct else None
    direct_check_failed = bool(direct_login_check) and not direct_login_check["is_login_page"]

    if direct_login_check:
        logger.info(
            "Direct login check: is_login_page=%s  score=%s  reason=%s",
            direct_login_check["is_login_page"],
            direct_login_check["score"],
            direct_login_check["reason"],
        )

    prior_checks_passed = health_check and not direct_check_failed
    provider_service_match = (
        provider_result if has_provider_or_service and prior_checks_passed else None
    )

    if provider_service_match:
        logger.info(
            "Provider/service match: matched=%s  score=%s  reason=%s",
            provider_service_match["matched"],
            provider_service_match["score"],
            provider_service_match["reason"],
        )
    elif has_provider_or_service and not prior_checks_passed:
        logger.info("Provider/service match result discarded — prior checks failed")

    notes = []
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

    provider_match_failed = (
        bool(provider_service_match) and not provider_service_match["matched"]
    )

    final_inactive_flagged = (
        llm_decision.inactive_flagged
        or direct_check_failed
        or provider_match_failed
    )

    if direct_check_failed:
        final_reason = direct_login_check["reason"]
    elif provider_match_failed:
        final_reason = provider_service_match["reason"]
    else:
        final_reason = llm_decision.reason or health_result.get("reason")

    if provider_service_match:
        matching_score = provider_service_match["score"]
    elif direct_login_check:
        matching_score = direct_login_check["score"]
    else:
        matching_score = None

    elapsed_ms = int((datetime.now() - request_start).total_seconds() * 1000)

    logger.info(
        "<<< /check response  url=%s  inactive_flagged=%s  reason=%s  "
        "health_check=%s  matching_score=%s  elapsed=%dms",
        url, final_inactive_flagged, final_reason,
        health_check, matching_score, elapsed_ms,
    )

    #DO NOT TOUCH THIS STRUCTURE, IT IS USED FOR THE RETURN RESPONSE
    return ReturnResponse(
        url=url,
        inactive_flagged=final_inactive_flagged,
        reason=final_reason,
        health_check=health_check,
        page_matching_score=matching_score,
        notes=" | ".join(notes) or None,
        updated_name=page_result.get("title"),
        marked_for_human_review=(
            bool(llm_decision.error) or direct_check_failed or provider_match_failed
        ),
        marked_for_deletion=final_inactive_flagged,
        errors=llm_decision.error or "",
        time=datetime.now().isoformat(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
